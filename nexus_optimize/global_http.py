"""E13 — Validación HTTP global con concurrencia real y dos cohortes.

No entrena ni modifica el modelo M1. Mide la misma API /api/pipeline/batch
para M1 y E9, sin caché. Corrige la interpretación de concurrencias
imposibles (p. ej., un tweet en un único lote no puede generar ocho
solicitudes simultáneas). Conserva resultados desfavorables.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import platform
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx

from .benchmark import percentile
from .integrity import audit_reference
from .nlp import normalize_tweet
from .real_http import RealHTTPOptions, _round, _pct, _positive_list
from .real_process import RealOptions, load_real_tweets

LOG = logging.getLogger("nexus_optimize.global_http")
_PUBLIC_METRICS = (
    "m1_p50_ms", "m1_p95_ms", "e9_p50_ms", "e9_p95_ms",
    "m1_throughput_tweets_s", "e9_throughput_tweets_s",
    "throughput_gain_pct", "p95_change_pct",
    "reference_control_drift_pct", "m1_server_processing_ms",
    "e9_server_processing_ms", "positive_gain_fraction",
)
_PHASES = {"primary", "exploratory", "excluded"}
_RESULTS = {"improved", "not_proven_improved", "regressed", "inconclusive",
            "not_applicable"}


@dataclass(frozen=True)
class GlobalHTTPOptions:
    """Configuración reproducible de E13; tamaños, tiempos y URL acotados."""
    url: str = "http://127.0.0.1:8014"
    sizes: tuple[int, ...] = (1, 100, 1000, 10000)
    concurrencies: tuple[int, ...] = (1, 4, 8)
    batch_size: int = 256
    repeats: int = 9
    timeout_s: float = 30.0
    seed: int = 2026
    gain_threshold_pct: float = 10.0
    drift_cap_pct: float = 20.0
    regression_cap_pct: float = 5.0
    p95_regression_cap_pct: float = 10.0

    def __post_init__(self) -> None:
        RealHTTPOptions(
            url=self.url, sizes=self.sizes,
            concurrencies=self.concurrencies, batch_size=self.batch_size,
            repeats=min(self.repeats, 12), timeout_s=self.timeout_s,
            seed=self.seed, threshold_pct=self.gain_threshold_pct,
        )
        if not 7 <= self.repeats <= 30:
            raise ValueError("repeats: E13 requiere entre 7 y 30")
        if (not 0 < self.drift_cap_pct <= 30
                or not 0 < self.regression_cap_pct <= 20
                or not 0 < self.p95_regression_cap_pct <= 30):
            raise ValueError("Umbrales de E13 fuera de rango")


def classify_case(n: int, concurrency: int, batch_size: int) -> tuple[str, str]:
    """Nunca confunde concurrencia solicitada con simultaneidad alcanzable."""
    if min(n, concurrency, batch_size) <= 0:
        raise ValueError("Tamaños y concurrencia deben ser positivos")
    requests = (n + batch_size - 1) // batch_size
    if concurrency > requests:
        return (
            "excluded",
            f"Con {n} tweets y lotes de {batch_size} solo hay {requests} "
            f"solicitudes: no se pueden medir {concurrency} simultáneas",
        )
    if n < 1000:
        return (
            "exploratory",
            "Muestra pequeña; se informa, pero no decide la validación global",
        )
    return "primary", "Condiciones de concurrencia alcanzables"


def _summarize_e13(
    groups: list[dict[str, list[dict[str, Any]]]],
    n: int, concurrency: int, batch_size: int,
    options: GlobalHTTPOptions,
) -> dict[str, Any]:
    """Medianas de pares ABBA/BAAB; p95 de RTT y guardarraíles medidos."""
    m1_first = [pair["m1"][0] for pair in groups]
    m1_second = [pair["m1"][1] for pair in groups]
    m1 = [point for pair in groups for point in pair["m1"]]
    e9 = [point for pair in groups for point in pair["e9"]]
    m1_rate = statistics.median(p["throughput"] for p in m1)
    e9_rate = statistics.median(p["throughput"] for p in e9)
    m1_p95 = percentile([ms for p in m1 for ms in p["rtts"]], 95)
    e9_p95 = percentile([ms for p in e9 for ms in p["rtts"]], 95)
    drift = abs(_pct(
        statistics.median(p["throughput"] for p in m1_first),
        statistics.median(p["throughput"] for p in m1_second),
    ))
    paired_gains: list[float] = []
    for pair in groups:
        baseline = statistics.median(p["throughput"] for p in pair["m1"])
        optimized = statistics.median(p["throughput"] for p in pair["e9"])
        paired_gains.append(_pct(baseline, optimized))
    gain = statistics.median(paired_gains)
    positive = sum(v > 0 for v in paired_gains) / len(paired_gains)
    p95_change = _pct(m1_p95, e9_p95)
    n_requests = (n + batch_size - 1) // batch_size
    observations = 2 * len(groups) * n_requests

    # Prohibido declarar una mejora con muestras p95 insuficientes.
    if drift > options.drift_cap_pct or observations < 40:
        result = "inconclusive"
    elif (
        gain >= options.gain_threshold_pct and
        p95_change <= 0 and positive >= 2 / 3
    ):
        result = "improved"
    elif (gain < -options.regression_cap_pct or
          p95_change > options.p95_regression_cap_pct):
        result = "regressed"
    else:
        result = "not_proven_improved"

    return {
        "tweets": n,
        "concurrency": concurrency,
        "effective_concurrency": min(concurrency, n_requests),
        "requests_per_round": n_requests,
        "batch_size": batch_size,
        "p95_observations_per_strategy": observations,
        "phase": classify_case(n, concurrency, batch_size)[0],
        "reason": classify_case(n, concurrency, batch_size)[1],
        "m1_p50_ms": round(percentile(
            [ms for p in m1 for ms in p["rtts"]], 50), 4),
        "m1_p95_ms": round(m1_p95, 4),
        "e9_p50_ms": round(percentile(
            [ms for p in e9 for ms in p["rtts"]], 50), 4),
        "e9_p95_ms": round(e9_p95, 4),
        "m1_throughput_tweets_s": round(m1_rate, 2),
        "e9_throughput_tweets_s": round(e9_rate, 2),
        "throughput_gain_pct": round(gain, 2),
        "p95_change_pct": round(p95_change, 2),
        "reference_control_drift_pct": round(drift, 2),
        "m1_server_processing_ms": round(
            statistics.median(p["server_ms"] for p in m1), 4),
        "e9_server_processing_ms": round(
            statistics.median(p["server_ms"] for p in e9), 4),
        "positive_gain_fraction": round(positive, 4),
        "labels_identical": True,
        "result": result,
    }


async def run_global_http(
    cohort_a: Sequence[str], cohort_b: Sequence[str],
    options: GlobalHTTPOptions,
    *, transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Benchmark A/B simétrico con cohortes disjuntas y rutas idénticas."""
    max_n = max(options.sizes)
    if len(cohort_a) < max_n or len(cohort_b) < max_n:
        raise ValueError("Cada cohorte debe tener al menos max(sizes) tweets")
    # Comprobación sobre texto normalizado: no se pueden filtrar duplicados
    # inventando caracteres o sustituyendo tweets por valores artificiales.
    keys_a = [normalize_tweet(value) for value in cohort_a[:max_n]]
    keys_b = [normalize_tweet(value) for value in cohort_b[:max_n]]
    if any(not k for k in keys_a + keys_b) or (
        len(set(keys_a)) != max_n or len(set(keys_b)) != max_n or
        bool(set(keys_a).intersection(keys_b))
    ):
        raise ValueError("Se requieren dos cohortes de tweets reales únicos y disjuntos")

    rows: list[dict[str, Any]] = []
    limit = httpx.Limits(max_connections=max(options.concurrencies) + 4)
    async with httpx.AsyncClient(
        base_url=options.url, timeout=options.timeout_s,
        transport=transport, limits=limit, follow_redirects=False,
        trust_env=False,
    ) as client:
        health = await client.get("/healthz")
        status_response = await client.get("/api/status")
        if health.status_code != 200 or status_response.status_code != 200:
            raise RuntimeError("Servidor E12 no disponible")
        status = status_response.json()
        if not (
            status.get("model_ready") is True and
            status.get("e9_api_enabled") is True and
            status.get("integrity") == "integrity_ok" and
            status.get("public_demo") is False
        ):
            raise RuntimeError("Servidor local sin M1/E9 validado; no se ejecuta E13")

        for cohort_name, tweets in (("test_a", cohort_a), ("test_b", cohort_b)):
            for n in options.sizes:
                batches = [list(tweets[i:i + options.batch_size])
                           for i in range(0, n, options.batch_size)]
                for concurrence in options.concurrencies:
                    phase, reason = classify_case(n, concurrence, options.batch_size)
                    if phase == "excluded":
                        rows.append({
                            "cohort": cohort_name, "tweets": n,
                            "concurrency": concurrence, "phase": phase,
                            "reason": reason, "result": "not_applicable",
                            "effective_concurrency": len(batches),
                            "requests_per_round": len(batches),
                            "batch_size": options.batch_size,
                            "labels_identical": None,
                        })
                        LOG.info("E13 %s n=%d conc=%d: NO APLICA, %s",
                                 cohort_name, n, concurrence, reason)
                        continue

                    # Precarga equitativa; fuera de las mediciones oficiales.
                    await _round(client, batches, "m1", min(concurrence, len(batches)))
                    await _round(client, batches, "e9", min(concurrence, len(batches)))
                    repetitions: list[dict[str, list[dict[str, Any]]]] = []
                    for iteration in range(options.repeats):
                        # ABBA / BAAB balancea el efecto del orden temporal:
                        # dos pasadas por cada estrategia y por repetición.
                        order = (
                            ("m1", "e9", "e9", "m1") if iteration % 2 == 0
                            else ("e9", "m1", "m1", "e9")
                        )
                        outcomes: dict[str, list[dict[str, Any]]] = {
                            "m1": [], "e9": []
                        }
                        for strategy in order:
                            result = await _round(client, batches, strategy, concurrence)
                            outcomes[strategy].append(result)
                        labels = outcomes["m1"][0]["labels"]
                        if any(p["labels"] != labels for name in ("m1", "e9")
                               for p in outcomes[name]):
                            raise RuntimeError(
                                f"E13 predicciones distintas, {cohort_name} "
                                f"n={n} conc={concurrence}; prueba interrumpida"
                            )
                        repetitions.append(outcomes)
                    row = _summarize_e13(
                        repetitions, n, concurrence, options.batch_size, options,
                    )
                    row["cohort"] = cohort_name
                    rows.append(row)
                    LOG.info(
                        "E13 %s n=%d conc=%d (%s) | ganancia=%+.2f%%"
                        " | p95=%+.2f%% | %s",
                        cohort_name, n, concurrence, phase,
                        row["throughput_gain_pct"], row["p95_change_pct"],
                        row["result"],
                    )

    primary = [row for row in rows if row["phase"] == "primary"]
    if not primary or len({r["cohort"] for r in primary}) != 2:
        raise RuntimeError("No existen comparaciones primarias suficientes")
    all_stable = all(r["reference_control_drift_pct"] <=
                     options.drift_cap_pct for r in primary)
    no_primary_regressions = all(
        r["result"] != "regressed" for r in primary
    )
    all_primary_improved = all(r["result"] == "improved" for r in primary)
    decision = (
        "inconclusive" if not all_stable else
        "candidate_validated" if all_primary_improved else
        "not_improved"
    )

    audit = audit_reference()
    return {
        "schema_version": 13,
        "status": "measured",
        "experiment": "E13 — M1 vs E9 HTTP; concurrencia alcanzable y dos cohortes",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "source": "Sentiment140 RAW, dos cohortes de tweets únicos disjuntos",
            "scope": "API HTTP local; mismos tweets, sin caché",
            "control_order": "ABBA / BAAB alternados, precarga excluida",
            "repeats": options.repeats,
            "sizes": list(options.sizes),
            "concurrencies": list(options.concurrencies),
            "batch_size": options.batch_size,
            "seed": options.seed,
            "eligibility": "Primaria n>=1000 y concurrencia<=solicitudes por ronda; "
                           "n<1000 exploratoria; concurrencia imposible excluida",
            "success": "Ganancia pareada >=10%; p95 no empeora; >=2/3 rondas "
                       "positivas, control referencia deriva<=20%",
            "attribution": "E9 es optimización de software; AG E11 no demostró "
                           "mejora incremental frente a E9",
        },
        "environment": {
            "python": platform.python_version(),
            "reference_integrity": audit["status"],
            "model_sha256": audit["model_sha256"],
            "e9_api_enabled": True,
        },
        "guardrails": {
            "exact_predictions": True,
            "reference_controls_stable": all_stable,
            "primary_no_regressions": no_primary_regressions,
            "all_primary_scenarios_improved": all_primary_improved,
            "two_independent_cohorts": True,
            "excluded_impossible_concurrencies": True,
        },
        "decision": decision,
        "summary": {
            "primary_total": len(primary),
            "primary_improved": sum(r["result"] == "improved" for r in primary),
            "primary_regressed": sum(r["result"] == "regressed" for r in primary),
            "primary_not_improved": sum(r["result"] == "not_proven_improved" for r in primary),
            "primary_inconclusive": sum(r["result"] == "inconclusive" for r in primary),
            "excluded": sum(r["phase"] == "excluded" for r in rows),
            "exploratory": sum(r["phase"] == "exploratory" for r in rows),
        },
        "results": rows,
    }


def sanitize_e13(report: dict[str, Any]) -> dict[str, Any]:
    """Publicación estricta de métricas. Nunca exporta tweets ni rutas."""
    if (not isinstance(report, dict) or report.get("schema_version") != 13
            or report.get("status") != "measured"
            or report.get("decision") not in {
                "not_improved", "candidate_validated", "inconclusive"
            } or not isinstance(report.get("results"), list)
            or not 1 <= len(report["results"]) <= 64):
        raise ValueError("Reporte E13 inválido")
    methods = report.get("methodology")
    guards = report.get("guardrails")
    summary = report.get("summary")
    if (not isinstance(methods, dict) or
        type(methods.get("repeats")) is not int or
        not 7 <= methods["repeats"] <= 30 or
        not isinstance(guards, dict) or
        not isinstance(summary, dict) or
        any(type(guards.get(k)) is not bool for k in (
            "exact_predictions", "reference_controls_stable",
            "primary_no_regressions", "all_primary_scenarios_improved",
            "two_independent_cohorts", "excluded_impossible_concurrencies",
        ))):
        raise ValueError("Metadatos E13 inválidos")
    safe_summary: dict[str, int] = {}
    for k in ("primary_total", "primary_improved", "primary_regressed",
              "primary_not_improved", "primary_inconclusive", "excluded",
              "exploratory"):
        v = summary.get(k)
        if type(v) is not int or not 0 <= v <= 64:
            raise ValueError("Resumen E13 inválido")
        safe_summary[k] = v
    clean: list[dict[str, Any]] = []
    for item in report["results"]:
        if not isinstance(item, dict):
            raise ValueError("Fila E13 inválida")
        n, c = item.get("tweets"), item.get("concurrency")
        phase, verdict = item.get("phase"), item.get("result")
        cohort = item.get("cohort")
        if (type(n) is not int or type(c) is not int
                or not 1 <= n <= 10000 or not 1 <= c <= 8
                or phase not in _PHASES or verdict not in _RESULTS
                or cohort not in {"test_a", "test_b"}):
            raise ValueError("Clave o decisión E13 inválida")
        row: dict[str, Any] = {
            "cohort": cohort, "tweets": n, "concurrency": c,
            "phase": phase, "result": verdict,
        }
        # No se publica el texto libre 'reason' de un archivo externo.
        if phase == "excluded":
            if verdict != "not_applicable":
                raise ValueError("Fila excluida inconsistente")
            row["reason"] = "Concurrencia solicitada superior a los lotes disponibles"
        else:
            if verdict == "not_applicable":
                raise ValueError("Falta una medición E13")
            for k in _PUBLIC_METRICS:
                value = item.get(k)
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value)
                        or not -1e7 <= value <= 1e9):
                    raise ValueError("Métrica E13 inválida")
                row[k] = value
            if type(item.get("labels_identical")) is not bool:
                raise ValueError("Comparación E13 inválida")
            row["labels_identical"] = item["labels_identical"]
            eff = item.get("effective_concurrency")
            if type(eff) is not int or not 1 <= eff <= c:
                raise ValueError("Concurrencia efectiva inválida")
            row["effective_concurrency"] = eff
        clean.append(row)
    return {
        "schema_version": 13, "status": "measured",
        "decision": report["decision"],
        "experiment": "E13 — comparación HTTP pareada M1/E9",
        "methodology": {
            "scope": "HTTP local, cohortes disjuntas; sin caché",
            "repeats": methods["repeats"],
            "control_order": "ABBA/BAAB",
        },
        "guardrails": {
            k: guards[k] for k in (
                "exact_predictions", "reference_controls_stable",
                "primary_no_regressions", "all_primary_scenarios_improved",
                "two_independent_cohorts", "excluded_impossible_concurrencies",
            )
        },
        "summary": safe_summary,
        "results": clean,
    }


def _write_report(report: dict[str, Any], location: Path) -> Path:
    target = location.expanduser().resolve()
    if target.exists():
        raise FileExistsError("No se sobrescribe un reporte E13 existente")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    if temp.exists():
        raise FileExistsError("Archivo temporal E13 existente")
    try:
        with temp.open("x", encoding="utf-8") as fd:
            json.dump(report, fd, ensure_ascii=False, indent=2)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8014")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=25000)
    parser.add_argument("--sizes", default="1,100,1000,10000")
    parser.add_argument("--concurrencies", default="1,4,8")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("results/e13_global_http.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        options = GlobalHTTPOptions(
            url=args.url,
            sizes=_positive_list(args.sizes, name="sizes", max_n=10000),
            concurrencies=_positive_list(args.concurrencies,
                                         name="concurrencies", max_n=8),
            batch_size=args.batch_size, repeats=args.repeats, seed=args.seed,
        )
        dataset_options = RealOptions(
            sizes=(max(options.sizes),),
            sample_size=args.sample_size, seed=options.seed,
        )
        workload = load_real_tweets(args.data, dataset_options)
        report = asyncio.run(run_global_http(
            workload.tuning, workload.evaluation, options,
        ))
        output = _write_report(report, args.output)
        print("\nE13 — HTTP REAL: DOS COHORTES, SIN CONCURRENCIA FICTICIA")
        print(f"{'Grupo':>7} {'Tweets':>7} {'Conc':>4} {'M1 p95':>10} "
              f"{'E9 p95':>10} {'Δ tweets/s':>12} {'Resultado':>22}")
        for row in report["results"]:
            if row["phase"] == "excluded":
                print(f"{row['cohort']:>7} {row['tweets']:>7} "
                      f"{row['concurrency']:>4} {'—':>10} {'—':>10} "
                      f"{'—':>12} {'NO APLICA':>22}")
            else:
                print(
                    f"{row['cohort']:>7} {row['tweets']:>7} {row['concurrency']:>4} "
                    f"{row['m1_p95_ms']:>10.3f} {row['e9_p95_ms']:>10.3f} "
                    f"{row['throughput_gain_pct']:>+11.2f}% "
                    f"{row['result']:>22}"
                )
        print("Primarios:", report["summary"])
        print("Decisión global:", report["decision"])
        print("Aporte AG sobre E9: no demostrado (E11); E13 evalúa E9.")
        print("Reporte:", output)
        return 0
    except (ValueError, OSError, RuntimeError, httpx.HTTPError) as exc:
        LOG.error("E13 no completado: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
