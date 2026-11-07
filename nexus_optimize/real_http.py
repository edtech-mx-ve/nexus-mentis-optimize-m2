"""E12 — Comparación HTTP real M1 / E9, sin caché ni tweets inventados.

El endpoint local /api/pipeline/batch realiza exactamente el mismo trabajo RAW:
normalización + TF-IDF/regresión logística, sin reentrenamiento ni caché.
Las mediciones incluyen transporte loopback, serialización y FastAPI, lo que
distingue E12 de los experimentos de inferencia local E8-E11.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

import httpx

from .benchmark import percentile
from .integrity import audit_reference
from .real_process import RealOptions, load_real_tweets

LOG = logging.getLogger("nexus_optimize.real_http")
STRATEGIES = ("m1", "e9")
_PUBLIC_ROW_FIELDS = (
    "tweets", "concurrency", "effective_concurrency", "requests_per_round",
    "batch_size", "m1_p50_ms", "m1_p95_ms", "e9_p50_ms", "e9_p95_ms",
    "m1_throughput_tweets_s", "e9_throughput_tweets_s",
    "throughput_gain_pct", "p95_change_pct", "m1_server_processing_ms",
    "e9_server_processing_ms", "reference_control_drift_pct",
    "labels_identical", "result",
)


@dataclass(frozen=True)
class RealHTTPOptions:
    url: str = "http://127.0.0.1:8014"
    sizes: tuple[int, ...] = (1, 100, 1000, 10000)
    concurrencies: tuple[int, ...] = (1, 4, 8)
    batch_size: int = 256
    repeats: int = 5
    timeout_s: float = 30.0
    seed: int = 2026
    threshold_pct: float = 10.0

    def __post_init__(self) -> None:
        parts = urlsplit(self.url)
        if (
            parts.scheme != "http" or parts.hostname not in {"127.0.0.1", "localhost"}
            or parts.port is None or parts.path not in ("", "/")
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment
        ):
            raise ValueError("E12 solo permite HTTP local en 127.0.0.1/localhost con puerto")
        if (not self.sizes or len(set(self.sizes)) != len(self.sizes)
                or min(self.sizes) < 1 or max(self.sizes) > 10000):
            raise ValueError("sizes: números distintos de 1 a 10000")
        if (not self.concurrencies or len(set(self.concurrencies)) != len(self.concurrencies)
                or any(not 1 <= c <= 8 for c in self.concurrencies)):
            raise ValueError("concurrencies: números distintos de 1 a 8")
        if not 1 <= self.batch_size <= 256:
            raise ValueError("batch_size debe estar en 1..256")
        if not 3 <= self.repeats <= 12:
            raise ValueError("repeats debe estar en 3..12")
        if not 1 <= self.timeout_s <= 60:
            raise ValueError("timeout_s fuera de rango")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed fuera de rango")
        if not 0 < self.threshold_pct <= 50:
            raise ValueError("threshold_pct inválido")


def _batches(texts: Sequence[str], batch_size: int) -> list[list[str]]:
    """Divide textos reales sin repetir ni inventar contenido."""
    if not texts:
        raise ValueError("No hay publicaciones")
    return [list(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]


async def _one(
    client: httpx.AsyncClient, limiter: asyncio.Semaphore,
    texts: list[str], strategy: str, index: int,
) -> dict[str, Any]:
    async with limiter:
        start_ns = time.perf_counter_ns()
        response = await client.post(
            "/api/pipeline/batch",
            json={"texts": texts, "language": "en", "strategy": strategy},
        )
        rtt_ms = (time.perf_counter_ns() - start_ns) / 1e6
    if response.status_code != 200:
        raise RuntimeError(f"Error HTTP {response.status_code} con {strategy}; lote {index}")
    try:
        answer = response.json()
        labels = answer["sentiments"]
        server_ms = float(answer["processing_ms"])
        if (answer["strategy"] != strategy or answer["items"] != len(texts)
                or len(labels) != len(texts)
                or any(label not in {"positivo", "negativo"} for label in labels)
                or not 0 <= server_ms < 60000):
            raise ValueError("Respuesta incompleta")
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError("Formato de respuesta HTTP E12 inválido") from exc
    return {
        "index": index, "labels": labels, "rtt_ms": rtt_ms,
        "server_ms": server_ms,
    }


async def _round(
    client: httpx.AsyncClient, batches: list[list[str]],
    strategy: str, concurrency: int,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter_ns()
    responses = await asyncio.gather(*[
        _one(client, semaphore, batch, strategy, idx)
        for idx, batch in enumerate(batches)
    ])
    elapsed_s = (time.perf_counter_ns() - started) / 1e9
    responses.sort(key=lambda item: item["index"])
    return {
        "labels": [label for part in responses for label in part["labels"]],
        "rtts": [point["rtt_ms"] for point in responses],
        "server_ms": sum(point["server_ms"] for point in responses),
        "throughput": sum(map(len, batches)) / elapsed_s,
    }


def _pct(before: float, after: float) -> float:
    if before <= 0:
        raise ValueError("La referencia debe ser positiva")
    return (after / before - 1.0) * 100


def _summarize(groups: list[dict[str, dict[str, Any]]],
               n: int, concurrence: int, batch_size: int,
               threshold_pct: float) -> dict[str, Any]:
    baseline_a = [g["m1_a"] for g in groups]
    baseline_b = [g["m1_b"] for g in groups]
    opt = [g["e9"] for g in groups]
    reference = baseline_a + baseline_b
    control_a = statistics.median(g["throughput"] for g in baseline_a)
    control_b = statistics.median(g["throughput"] for g in baseline_b)
    drift = abs(_pct(control_a, control_b))
    m1_rate = statistics.median(g["throughput"] for g in reference)
    e9_rate = statistics.median(g["throughput"] for g in opt)
    m1_rtts = [v for g in reference for v in g["rtts"]]
    e9_rtts = [v for g in opt for v in g["rtts"]]
    m1_p95 = percentile(m1_rtts, 95)
    e9_p95 = percentile(e9_rtts, 95)
    rate_gain = _pct(m1_rate, e9_rate)
    p95_change = _pct(m1_p95, e9_p95)
    exact = all(
        group["m1_a"]["labels"] == group["e9"]["labels"] ==
        group["m1_b"]["labels"] for group in groups
    )
    verdict = (
        "inconclusive" if drift > 20 else
        "improved" if rate_gain >= threshold_pct and p95_change <= 0 else
        "regressed" if rate_gain < -5 or p95_change > 10 else
        "not_proven_improved"
    )
    return {
        "tweets": n,
        "concurrency": concurrence,
        "effective_concurrency": min(concurrence, (n + batch_size - 1) // batch_size),
        "requests_per_round": (n + batch_size - 1) // batch_size,
        "batch_size": batch_size,
        "m1_p50_ms": round(percentile(m1_rtts, 50), 4),
        "m1_p95_ms": round(m1_p95, 4),
        "e9_p50_ms": round(percentile(e9_rtts, 50), 4),
        "e9_p95_ms": round(e9_p95, 4),
        "m1_throughput_tweets_s": round(m1_rate, 2),
        "e9_throughput_tweets_s": round(e9_rate, 2),
        "throughput_gain_pct": round(rate_gain, 2),
        "p95_change_pct": round(p95_change, 2),
        "m1_server_processing_ms": round(statistics.median(
            g["server_ms"] for g in reference), 4),
        "e9_server_processing_ms": round(statistics.median(
            g["server_ms"] for g in opt), 4),
        "reference_control_drift_pct": round(drift, 2),
        "labels_identical": exact,
        "result": verdict,
    }


async def run_real_http(
    tweets: Sequence[str], options: RealHTTPOptions,
    *, transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Realiza A-B-A pareado para cada volumen y concurrencia, solo localhost."""
    if len(tweets) < max(options.sizes):
        raise ValueError("Faltan tweets reales distintos para todas las pruebas")
    # El protocolo se niega a inventar textos para completar lotes.
    if len(set(tweets[:max(options.sizes)])) != max(options.sizes):
        raise ValueError("Los tweets RAW deben ser distintos")
    rows: list[dict[str, Any]] = []
    limits = httpx.Limits(max_connections=max(options.concurrencies) + 4)
    async with httpx.AsyncClient(
        base_url=options.url, transport=transport, limits=limits,
        timeout=options.timeout_s, follow_redirects=False, trust_env=False,
    ) as client:
        health = await client.get("/healthz")
        status_resp = await client.get("/api/status")
        if health.status_code != 200 or status_resp.status_code != 200:
            raise RuntimeError("Servidor sin modelo listo")
        status = status_resp.json()
        if not (
            status.get("model_ready") is True
            and status.get("e9_api_enabled") is True
            and status.get("integrity") == "integrity_ok"
            and status.get("public_demo") is False
        ):
            raise RuntimeError("El servidor no tiene E9 local habilitado e íntegro")
        for n in options.sizes:
            batches = _batches(tweets[:n], options.batch_size)
            for concurrence in options.concurrencies:
                measurements: list[dict[str, dict[str, Any]]] = []
                for iteration in range(options.repeats):
                    a = await _round(client, batches, "m1", concurrence)
                    b = await _round(client, batches, "e9", concurrence)
                    c = await _round(client, batches, "m1", concurrence)
                    if a["labels"] != b["labels"] or c["labels"] != b["labels"]:
                        raise RuntimeError(
                            f"Predicciones NO equivalentes: n={n}, concurrencia={concurrence}"
                        )
                    measurements.append({"m1_a": a, "e9": b, "m1_b": c})
                row = _summarize(measurements, n, concurrence,
                                 options.batch_size, options.threshold_pct)
                if not row["labels_identical"]:
                    raise RuntimeError("Equivalencia de etiquetas inválida")
                rows.append(row)
                LOG.info(
                    "E12 n=%s conc=%s (efectiva=%s) | M1 %.2f vs E9 %.2f tweets/s"
                    " | Δ=%+.2f%% p95=%+.2f%% [%s]",
                    n, concurrence, row["effective_concurrency"],
                    row["m1_throughput_tweets_s"], row["e9_throughput_tweets_s"],
                    row["throughput_gain_pct"], row["p95_change_pct"], row["result"],
                )

    required = [
        row for row in rows if row["tweets"] >= 1000
        and row["concurrency"] in {1, 4, 8}
    ]
    exact = all(row["labels_identical"] for row in rows)
    stable = all(row["reference_control_drift_pct"] <= 20 for row in required)
    improved = bool(required) and all(row["result"] == "improved" for row in required)
    decision = "candidate_validated" if exact and stable and improved else (
        "inconclusive" if not stable else "not_improved"
    )
    return {
        "schema_version": 12,
        "status": "measured",
        "experiment": "E12 — API real: M1 vs E9 sin caché",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "scope": "HTTP localhost, mismo endpoint y mismo worker",
            "repeats": options.repeats, "batch_size": options.batch_size,
            "sizes": list(options.sizes),
            "concurrencies": list(options.concurrencies),
            "source": "Sentiment140 RAW, tweets naturales únicos",
            "note": "El F1 histórico NO se recalculó. AG E11 sin mejora adicional. "
                    "RTT HTTP por solicitud, throughput por volumen completo.",
            "seed": options.seed,
            "reference_controls": "M1-A / E9 / M1-B, sin caché",
        },
        "environment": {
            "python": platform.python_version(),
            "reference_integrity": audit_reference()["status"],
            "model_sha256": audit_reference()["model_sha256"],
            "e9_api_enabled": True,
        },
        "guardrails": {
            "exact_predictions": exact,
            "reference_controls_stable": stable,
            "all_required_scenarios_improved": improved,
        },
        "decision": decision,
        "results": rows,
    }


def sanitize_e12(report: dict[str, Any]) -> dict[str, Any]:
    """Salida pública segura: campos explícitos; sin tweets, rutas ni corpus."""
    if (
        report.get("schema_version") != 12
        or report.get("status") != "measured"
        or report.get("decision") not in {
            "candidate_validated", "inconclusive", "not_improved"
        }
        or not isinstance(report.get("results"), list)
        or not 1 <= len(report["results"]) <= 40
    ):
        raise ValueError("Reporte E12 inválido")
    method = report.get("methodology", {})
    if not isinstance(method, dict) or type(method.get("repeats")) is not int:
        raise ValueError("Metodología de E12 inválida")
    rows: list[dict[str, Any]] = []
    for item in report["results"]:
        if not isinstance(item, dict) or not all(key in item for key in _PUBLIC_ROW_FIELDS):
            raise ValueError("Métricas E12 incompletas")
        if type(item["tweets"]) is not int or not 1 <= item["tweets"] <= 10000:
            raise ValueError("Tamaño de carga inválido")
        if item["result"] not in {
            "improved", "inconclusive", "regressed", "not_proven_improved"
        } or type(item["labels_identical"]) is not bool:
            raise ValueError("Decisión inválida")
        if any(
            not isinstance(item[key], (int, float))
            or isinstance(item[key], bool)
            or not -1e7 <= float(item[key]) <= 1e9
            for key in (
                "m1_p50_ms", "m1_p95_ms", "e9_p50_ms", "e9_p95_ms",
                "m1_throughput_tweets_s", "e9_throughput_tweets_s",
                "throughput_gain_pct", "p95_change_pct",
            )
        ):
            raise ValueError("Métricas E12 fuera de rango")
        rows.append({key: item[key] for key in _PUBLIC_ROW_FIELDS})
    guard = report.get("guardrails", {})
    if not isinstance(guard, dict) or any(
        type(guard.get(key)) is not bool for key in (
            "exact_predictions", "reference_controls_stable",
            "all_required_scenarios_improved",
        )
    ):
        raise ValueError("Guardarraíles ausentes")
    return {
        "schema_version": 12,
        "status": "measured",
        "experiment": "E12 — API real M1 / E9, sin caché",
        "methodology": {
            "scope": "HTTP local, mediciones pareadas; no resultados de producción",
            "repeats": method["repeats"],
            "source": "Sentiment140 RAW, tweets naturales únicos",
        },
        "decision": report["decision"],
        "guardrails": {k: guard[k] for k in (
            "exact_predictions", "reference_controls_stable",
            "all_required_scenarios_improved",
        )},
        "results": rows,
    }


def _positive_list(value: str, *, name: str, max_n: int) -> tuple[int, ...]:
    try:
        result = tuple(int(i.strip()) for i in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} debe ser una lista de enteros") from exc
    if not result or len(set(result)) != len(result) or any(
        x < 1 or x > max_n for x in result
    ):
        raise ValueError(f"{name}: enteros únicos de 1 a {max_n}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8014")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=25000)
    parser.add_argument("--sizes", default="1,100,1000,10000")
    parser.add_argument("--concurrencies", default="1,4,8")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path,
                        default=Path("results/e12_real_http.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        options = RealHTTPOptions(
            url=args.url, sizes=_positive_list(args.sizes, name="sizes", max_n=10000),
            concurrencies=_positive_list(args.concurrencies,
                                         name="concurrencies", max_n=8),
            batch_size=args.batch_size,
            repeats=args.repeats, seed=args.seed,
        )
        sample_opts = RealOptions(
            sizes=(max(options.sizes),),
            sample_size=args.sample_size, seed=options.seed,
        )
        workload = load_real_tweets(args.data, sample_opts)
        report = asyncio.run(run_real_http(workload.evaluation, options))
        destination = args.output.expanduser().resolve()
        if destination.exists():
            raise FileExistsError("No se sobrescribe el reporte existente. Use otra ruta")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Escritura atómica sin incluir corpus ni tweets individuales.
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        if tmp.exists():
            raise FileExistsError("Archivo temporal E12 ya existe")
        try:
            with tmp.open("x", encoding="utf-8") as out:
                json.dump(report, out, ensure_ascii=False, indent=2)
            os.replace(tmp, destination)
        finally:
            tmp.unlink(missing_ok=True)
        print("\nE12 — HTTP LOCAL: M1 ORIGINAL vs E9 OPTIMIZADO (SIN CACHÉ)")
        print(f"{'Tweets':>7} {'Conc':>5} {'M1 p95':>10} {'E9 p95':>10}"
              f" {'Δ p95':>10} {'Δ tweets/s':>12} {'Veredicto':>22}")
        for r in report["results"]:
            print(f"{r['tweets']:>7} {r['concurrency']:>5} "
                  f"{r['m1_p95_ms']:>10.3f} {r['e9_p95_ms']:>10.3f} "
                  f"{r['p95_change_pct']:>+9.2f}% "
                  f"{r['throughput_gain_pct']:>+11.2f}% {r['result']:>22}")
        print("Decisión global:", report["decision"])
        print("AG aporta mejora adicional: no demostrada en E11; E12 compara E9, no AG")
        print("Reporte:", destination)
        return 0
    except (ValueError, OSError, RuntimeError, httpx.HTTPError) as exc:
        LOG.error("E12 interrumpido: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
