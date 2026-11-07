"""E14 — Conciliación auditada de métricas E13 sin alterar experimentos.

Una única variación principal de rendimiento: cociente entre medianas
de tweets/s M1 y E9. La mediana de ganancias pareadas de E13 se conserva
como estadístico secundario, no se confunde con la cifra visual.
Esta operación reanaliza E13: no es una nueva medición HTTP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .global_http import sanitize_e13, classify_case, _write_report
from .integrity import audit_reference, read_json_object

LOG = logging.getLogger("nexus_optimize.metrics_closure")
THRESHOLD_PCT = 10.0
DRIFT_CAP_PCT = 20.0
REGRESSION_CAP_PCT = 5.0
P95_REGRESSION_CAP_PCT = 10.0
KEYS = ("cohort", "tweets", "concurrency", "phase")
METRIC_METHOD = "ratio_of_medians"


def throughput_gain(m1_rate: float, e9_rate: float) -> float:
    """Variación de rendimiento basada en las medianas mostradas, sin efectos."""
    if not all(math.isfinite(x) for x in (m1_rate, e9_rate)) or m1_rate <= 0 or e9_rate <= 0:
        raise ValueError("Rendimientos no positivos o no finitos")
    return 100.0 * (e9_rate / m1_rate - 1.0)


def _check_derived_row(source: dict[str, Any], clean: dict[str, Any],
                       repeats: int) -> dict[str, Any]:
    """Valida campos originales necesarios y reevalúa el veredicto con métrica única."""
    row = dict(clean)
    if clean["phase"] == "excluded":
        if clean["result"] != "not_applicable":
            raise ValueError("Fila excluida inválida")
        return row

    n, concurrency = row["tweets"], row["concurrency"]
    batch = source.get("batch_size")
    requests = source.get("requests_per_round")
    observations = source.get("p95_observations_per_strategy")
    if (type(batch) is not int or batch != 256 or
            type(requests) is not int or requests != math.ceil(n / batch) or
            type(observations) is not int or observations != 2 * repeats * requests or
            row["effective_concurrency"] != concurrency or
            classify_case(n, concurrency, batch)[0] != row["phase"]):
        raise ValueError("Metadatos de concurrencia u observaciones inconsistentes")
    if (not row["labels_identical"] or
            not 0 <= row["positive_gain_fraction"] <= 1 or
            row["m1_p95_ms"] <= 0 or row["e9_p95_ms"] <= 0 or
            row["reference_control_drift_pct"] < 0 or
            row["m1_throughput_tweets_s"] <= 0 or
            row["e9_throughput_tweets_s"] <= 0):
        raise ValueError("Métricas de entrada inválidas")
    gain = throughput_gain(
        row["m1_throughput_tweets_s"], row["e9_throughput_tweets_s"])
    p95_change = 100 * (row["e9_p95_ms"] / row["m1_p95_ms"] - 1)
    # Los tiempos de E13 están redondeados a 4 decimales. Se admite
    # una tolerancia de 0,025 pp frente al porcentaje de origen.
    if abs(p95_change - row["p95_change_pct"]) > 0.025:
        raise ValueError("Variación p95 inconsistente con las latencias")
    paired = row["throughput_gain_pct"]
    row["paired_throughput_gain_pct"] = paired
    row["throughput_gain_pct"] = round(gain, 2)
    row["p95_change_pct"] = round(p95_change, 2)
    row["difference_from_paired_pp"] = round(gain - paired, 2)
    if row["reference_control_drift_pct"] > DRIFT_CAP_PCT or observations < 40:
        verdict = "inconclusive"
    elif gain >= THRESHOLD_PCT and p95_change <= 0 and row["positive_gain_fraction"] >= 2 / 3:
        verdict = "improved"
    elif gain < -REGRESSION_CAP_PCT or p95_change > P95_REGRESSION_CAP_PCT:
        verdict = "regressed"
    else:
        verdict = "not_proven_improved"
    row["original_e13_result"] = row["result"]
    row["result"] = verdict
    row["p95_observations_per_strategy"] = observations
    return row


def reconcile_e13(original: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    """Reanaliza todas las filas; no altera el objeto E13 recibido."""
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in source_sha256
    ):
        raise ValueError("Huella SHA-256 de E13 inválida")
    clean = sanitize_e13(original)
    if clean["guardrails"]["exact_predictions"] is not True:
        raise ValueError("E13 no confirma predicciones idénticas")
    if (not clean["guardrails"]["two_independent_cohorts"] or
            not clean["guardrails"]["excluded_impossible_concurrencies"]):
        raise ValueError("El protocolo E13 carece de controles obligatorios")
    repeats = clean["methodology"]["repeats"]
    source_rows = original["results"]
    if len(source_rows) != len(clean["results"]):
        raise ValueError("Filas inconsistentes")
    # Impide filas duplicadas u omisión de muestras en la conciliación.
    unique: set[tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    for source, sanit in zip(source_rows, clean["results"]):
        k = tuple(sanit[field] for field in KEYS)
        if k in unique:
            raise ValueError("Escenario E13 duplicado")
        unique.add(k)
        row = _check_derived_row(source, sanit, repeats)
        rows.append(row)
    primary = [r for r in rows if r["phase"] == "primary"]
    if not primary or {r["cohort"] for r in primary} != {"test_a", "test_b"}:
        raise ValueError("Faltan muestras primarias independientes")
    if not all(r["labels_identical"] for r in rows if r["phase"] != "excluded"):
        raise ValueError("Predicciones discordantes")
    drift_stable = all(r["reference_control_drift_pct"] <= DRIFT_CAP_PCT for r in primary)
    all_improved = all(r["result"] == "improved" for r in primary)
    decision = ("inconclusive" if not drift_stable else
                "candidate_validated" if all_improved else "not_improved")
    summary = {
        "primary_total": len(primary),
        "primary_improved": sum(r["result"] == "improved" for r in primary),
        "primary_regressed": sum(r["result"] == "regressed" for r in primary),
        "primary_not_improved": sum(r["result"] == "not_proven_improved" for r in primary),
        "primary_inconclusive": sum(r["result"] == "inconclusive" for r in primary),
        "excluded": sum(r["phase"] == "excluded" for r in rows),
        "exploratory": sum(r["phase"] == "exploratory" for r in rows),
    }
    # El estado E13 original se conserva: se declara divergencia si E14 reevalúa distinto.
    return {
        "schema_version": 14,
        "status": "measured",
        "experiment": "E14 — conciliación de E13 (no es nueva medición)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_e13_sha256": source_sha256,
        "source_e13_decision": clean["decision"],
        "decision": decision,
        "decision_changed": decision != clean["decision"],
        "methodology": {
            "repeats": repeats,
            "scope": "Reanálisis de E13: HTTP local, dos cohortes, sin caché",
            "throughput_gain_method": METRIC_METHOD,
            "paired_gain_method": "median_of_paired_gains (E13 histórico, diagnóstico)",
            "p95_method": "p95_pool_de_solicitudes_por_estrategia",
            "rounding": "tasas 2 decimales; porcentajes 2 decimales",
            "decision_rule": "ratio de medianas >=10%, p95 no aumenta, >=2/3 "
                             "parejas con mejora y controles estables",
            "attribution": "La ganancia corresponde a E9; E11 no demostró aporte AG adicional",
        },
        "guardrails": {
            "exact_predictions": True,
            "reference_controls_stable": drift_stable,
            "two_independent_cohorts": True,
            "all_primary_scenarios_improved": all_improved,
            "primary_no_regressions": all(r["result"] != "regressed" for r in primary),
        },
        "summary": summary,
        "results": rows,
    }


def public_e14(e14: dict[str, Any]) -> dict[str, Any]:
    """Lista blanca para API: sin rutas, texto ni variables de entorno."""
    if (not isinstance(e14, dict) or e14.get("schema_version") != 14 or
            e14.get("status") != "measured" or
            e14.get("decision") not in {"candidate_validated", "not_improved", "inconclusive"} or
            e14.get("source_e13_decision") not in {"candidate_validated", "not_improved", "inconclusive"} or
            type(e14.get("decision_changed")) is not bool or
            type(e14.get("guardrails")) is not dict or
            not isinstance(e14.get("results"), list) or
            not 1 <= len(e14["results"]) <= 64):
        raise ValueError("Informe E14 inválido")
    doc = {
        "schema_version": 14, "status": "measured",
        "experiment": "E14: conciliación de E13, sin nuevas mediciones",
        "decision": e14["decision"], "decision_changed": e14["decision_changed"],
        "source_e13_decision": e14["source_e13_decision"],
        "methodology": {"repeats": None, "throughput_gain_method": METRIC_METHOD,
                        "scope": "HTTP local; mismos tweets; sin caché; E14 reanaliza E13"},
        "guardrails": {},
        "summary": {},
        "results": [],
    }
    methodology = e14.get("methodology", {})
    rep = methodology.get("repeats") if isinstance(methodology, dict) else None
    if type(rep) is not int or not 7 <= rep <= 30 or methodology.get(
        "throughput_gain_method") != METRIC_METHOD:
        raise ValueError("Método E14 inválido")
    doc["methodology"]["repeats"] = rep
    for key in ("exact_predictions", "reference_controls_stable",
                "two_independent_cohorts", "all_primary_scenarios_improved",
                "primary_no_regressions"):
        v = e14["guardrails"].get(key)
        if type(v) is not bool:
            raise ValueError("Guardarraíles E14 inválidos")
        doc["guardrails"][key] = v
    summary = e14.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Resumen E14 ausente")
    for key in ("primary_total", "primary_improved", "primary_regressed",
                "primary_not_improved", "primary_inconclusive", "excluded", "exploratory"):
        val = summary.get(key)
        if type(val) is not int or not 0 <= val <= 64:
            raise ValueError("Resumen E14 inválido")
        doc["summary"][key] = val
    # El frontend necesita sólo estos campos. Otros (rutas, textos) se descartan.
    metrics = (
        "m1_p50_ms", "m1_p95_ms", "e9_p50_ms", "e9_p95_ms",
        "m1_throughput_tweets_s", "e9_throughput_tweets_s",
        "throughput_gain_pct", "paired_throughput_gain_pct",
        "difference_from_paired_pp", "p95_change_pct",
        "reference_control_drift_pct", "positive_gain_fraction",
    )
    for entry in e14["results"]:
        if not isinstance(entry, dict):
            raise ValueError("Fila E14 inválida")
        cohort, n, concurrency = entry.get("cohort"), entry.get("tweets"), entry.get("concurrency")
        phase, verdict = entry.get("phase"), entry.get("result")
        if (cohort not in {"test_a", "test_b"} or type(n) is not int or
                type(concurrency) is not int or not 1 <= n <= 10000 or
                not 1 <= concurrency <= 8 or phase not in {
                    "primary", "exploratory", "excluded"} or verdict not in {
                    "improved", "regressed", "not_proven_improved", "inconclusive",
                    "not_applicable"}):
            raise ValueError("Clave E14 inválida")
        row = {"cohort": cohort, "tweets": n,
               "concurrency": concurrency, "phase": phase, "result": verdict}
        if phase == "excluded":
            if verdict != "not_applicable":
                raise ValueError("Fila excluida inválida")
            row["reason"] = "No hay suficientes lotes para esa concurrencia"
        else:
            if verdict == "not_applicable" or type(entry.get("labels_identical")) is not bool:
                raise ValueError("Fila medida inválida")
            row["labels_identical"] = entry["labels_identical"]
            eff = entry.get("effective_concurrency")
            if type(eff) is not int or not 1 <= eff <= concurrency:
                raise ValueError("Concurrencia efectiva inválida")
            row["effective_concurrency"] = eff
            for key in metrics:
                v = entry.get(key)
                if (type(v) not in (int, float) or not math.isfinite(v) or
                        abs(v) > 1e9):
                    raise ValueError(f"Valor E14 inválido: {key}")
                row[key] = v
            if row["m1_throughput_tweets_s"] <= 0 or row["e9_throughput_tweets_s"] <= 0:
                raise ValueError("Rendimientos E14 inválidos")
            expected = throughput_gain(row["m1_throughput_tweets_s"],
                                       row["e9_throughput_tweets_s"])
            if abs(round(expected, 2) - row["throughput_gain_pct"]) > 0.011:
                raise ValueError("Porcentaje E14 inconsistente")
        doc["results"].append(row)
    return doc


def load_and_reconcile(path: Path) -> dict[str, Any]:
    """Lee E13 con cota de tamaño, calcula SHA-256 y concilia sus métricas."""
    source = path.expanduser().resolve()
    if not source.is_file() or source.stat().st_size > 2_000_000:
        raise ValueError("E13 no existe o supera el límite permitido")
    raw = source.read_bytes()
    parsed = json.loads(raw.decode("utf-8"))
    return reconcile_e13(parsed, hashlib.sha256(raw).hexdigest())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/e13_global_http.json"))
    parser.add_argument("--output", type=Path, default=Path("results/e14_metrics_closure.json"))
    args = parser.parse_args()
    logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)
    try:
        audit = audit_reference()
        if audit["status"] != "integrity_ok":
            raise ValueError("Modelo M1 no íntegro; se detiene E14")
        report = load_and_reconcile(args.source)
        public_e14(report)
        path = _write_report(report, args.output)
        print("\nE14 — CONCILIACIÓN DEL EXPERIMENTO E13")
        print("Método oficial: (mediana E9 tweets/s / mediana M1 tweets/s − 1) × 100")
        print("Se conserva E13 original; no se repite ni se modifica el modelo.")
        print(f"{'Muestra':>8} {'Tweets':>7} {'Conc':>4} {'M1/s':>10} "
              f"{'E9/s':>10} {'Ganancia':>10} {'Pareada E13':>12} {'Decisión':>21}")
        for row in report["results"]:
            if row["phase"] == "excluded":
                continue
            print(f"{row['cohort']:>8} {row['tweets']:>7} {row['concurrency']:>4} "
                  f"{row['m1_throughput_tweets_s']:>10.0f} "
                  f"{row['e9_throughput_tweets_s']:>10.0f} "
                  f"{row['throughput_gain_pct']:>+9.2f}% "
                  f"{row['paired_throughput_gain_pct']:>+11.2f}% "
                  f"{row['result']:>21}")
        print("Resultados primarios:", report["summary"])
        print("Decisión E13 original:", report["source_e13_decision"])
        print("Decisión conciliada E14:", report["decision"])
        print("Reporte E14:", path)
        return 0
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        LOG.error("E14 no completado: %s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
