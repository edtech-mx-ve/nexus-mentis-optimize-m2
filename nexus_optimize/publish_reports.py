"""Publica versiones mínimas E0/E1/E2/E3, sin textos ni rutas locales."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .integrity import BASE_DIR, read_json_object

LOG = logging.getLogger(__name__)

SPEC: dict[str, tuple[int, tuple[str, ...]]] = {
    "baseline_real.json": (
        1, ("batch_size", "request_latency_p95_ms",
            "throughput_median_tweets_per_s", "amortized_median_ms_per_tweet")
    ),
    "comparison_e1_real.json": (
        2, ("scenario", "batch_size", "strategy",
            "throughput_median_tweets_per_s", "speedup_vs_paired_reference_x",
            "request_latency_p95_ms", "exact_prediction_equivalence")
    ),
    "comparison_e2_http.json": (
        3, ("scenario", "concurrency", "strategy",
            "latency_client_p95_ms", "throughput_median_tweets_per_s",
            "server_rss_observed_mb", "exact_prediction_equivalence")
    ),
    "comparison_e4_ga_http.json": (
        3, ("scenario", "concurrency", "strategy",
            "latency_client_p95_ms", "throughput_median_tweets_per_s",
            "exact_prediction_equivalence")
    ),
    "e5_selective_http.json": (
        5, ("scenario", "concurrency", "strategy",
            "latency_client_p95_ms", "throughput_median_tweets_per_s",
            "execution_path_counts", "exact_prediction_equivalence")
    ),
    "e6_stability_http.json": (
        6, ("scenario", "concurrency", "strategy",
            "latency_client_p95_ms", "throughput_median_tweets_per_s",
            "execution_path_counts", "exact_prediction_equivalence")
    ),
    "e7_stratified_http.json": (
        7, ("scenario", "concurrency", "strategy",
            "latency_client_p95_ms", "throughput_median_tweets_per_s",
            "execution_path_counts", "expected_path_counts",
            "exact_prediction_equivalence")
    ),
}


def sanitize_evolution(report: dict[str, Any]) -> dict[str, Any]:
    """Reduce E3 a métricas de validación; no exporta corpus ni trazas privadas."""
    if (report.get("schema_version") != 4
            or report.get("status") not in {"candidate_validated", "not_improved"}):
        raise ValueError("Esquema E3 desconocido")
    try:
        genome = report["train_best"]["genome"]
        holdout = report["holdout"]
        safe_rows = {}
        for split in ("baseline", "candidate"):
            safe_rows[split] = {
                scenario: {
                    name: holdout[split][scenario][name]
                    for name in ("latency_p95_ms", "throughput_tweets_s",
                                 "rss_sampled_mb", "exact_prediction_equivalence")
                }
                for scenario in ("original", "repeated")
            }
        permitted = ("batch_size", "strategy", "cache_capacity", "concurrency")
        selected = {name: genome[name] for name in permitted}
        guardrails = {k: bool(v) for k, v in holdout["guardrails"].items()}
        score = float(holdout["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Reporte E3 incompleto") from exc
    return {
        "schema_version": 4, "status": report["status"],
        "experiment": "E3 - búsqueda genética local, sin reentrenamiento",
        "generated_utc": report.get("generated_utc"),
        "methodology": {"scope": "inferencia Python local (NO HTTP)",
                        "no_training": True},
        "train_best": {"genome": selected},
        "holdout": {**safe_rows, "score": score, "guardrails": guardrails},
        "recommendation": report.get("recommendation", "Verificar HTTP antes de desplegar"),
    }


def sanitize_report(filename: str, report: dict[str, Any]) -> dict[str, Any]:
    """Crea un subconjunto exclusivamente de métricas para el panel público."""
    if filename == "e3_genetic.json":
        return sanitize_evolution(report)
    if filename not in SPEC:
        raise ValueError("Nombre de reporte desconocido")
    version, columns = SPEC[filename]
    if report.get("schema_version") != version or report.get("status") != "measured":
        raise ValueError(f"Reporte inválido o sin ejecutar: {filename}")
    if not isinstance(report.get("results"), list) or not report["results"]:
        raise ValueError("No hay resultados para publicar")
    if len(report["results"]) > 100:
        raise ValueError("Reporte demasiado extenso para exposición pública")
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(report["results"]):
        if not isinstance(row, dict) or not all(k in row for k in columns):
            raise ValueError(f"Fila {index} incompleta en {filename}")
        if version in (5, 6, 7):
            dictionaries = [row["execution_path_counts"]]
            if version == 7:
                dictionaries.append(row["expected_path_counts"])
            for counts in dictionaries:
                if not isinstance(counts, dict) or any(
                    path not in ("reference", "lru", "dedupe", "unknown")
                    or type(count) is not int or count < 0
                    for path, count in counts.items()
                ):
                    raise ValueError("Indicadores de rutas HTTP inválidos")
        rows.append({key: row[key] for key in columns})
    method = report.get("methodology", {})
    if not isinstance(method, dict) or not isinstance(method.get("repeats"), int):
        raise ValueError("Falta el número de repeticiones")
    result: dict[str, Any] = {
        "schema_version": version,
        "status": "measured",
        "experiment": str(report.get("experiment", f"E{version-1}")),
        "generated_utc": report.get("generated_utc"),
        "methodology": {"repeats": method["repeats"]},
        "results": rows,
    }
    if version == 1:
        provenance = report.get("data_provenance", {})
        result["data_provenance"] = {
            "source_type": provenance.get("source_type", "desconocido"),
            "warning": provenance.get("warning", "Medición local"),
        }
    if version == 2:
        scenarios = report.get("scenarios", [])
        result["scenarios"] = [
            {k: r[k] for k in ("name", "unique_texts", "observed_duplicate_fraction")
             if k in r}
            for r in scenarios if isinstance(r, dict)
        ]
    if version in (3, 5, 6, 7):
        result["methodology"].update({
            "cache_note": str(method.get("cache_note", "Caché local")),
            "network_scope": str(method.get("network_scope", "HTTP local")),
        })
    if version in (5, 6, 7):
        decision = report.get("decision")
        guardrails = report.get("guardrails")
        routes = report.get("execution_paths")
        if decision not in ("candidate_validated", "not_improved", "inconclusive"):
            raise ValueError("Decisión E5/E6 inválida")
        if not isinstance(guardrails, dict) or not all(
            isinstance(k, str) and isinstance(v, bool) for k, v in guardrails.items()
        ):
            raise ValueError("Guardarraíles E5/E6 inválidos")
        if not isinstance(routes, dict) or any(
            k not in ("reference", "lru") or type(v) is not int or v < 0
            for k, v in routes.items()
        ):
            raise ValueError("Rutas E5/E6 inválidas")
        result["decision"] = decision
        result["guardrails"] = guardrails
        result["execution_paths"] = routes
    if version == 7:
        cohort_stats = report.get("cohort_stats")
        if not isinstance(cohort_stats, dict) or set(cohort_stats) != {
            "unique", "natural", "repeated"
        }:
            raise ValueError("E7 requiere tres cohortes")
        safe_stats = {}
        for key, values in cohort_stats.items():
            if not isinstance(values, dict):
                raise ValueError("Cohorte E7 inválida")
            safe_stats[key] = {}
            for name in ("items", "distinct_texts", "duplicate_positions",
                         "reference_batches_expected", "lru_batches_expected"):
                value = values.get(name)
                if type(value) is not int or value < 0:
                    raise ValueError("Conteo E7 inválido")
                safe_stats[key][name] = value
        result["cohort_stats"] = safe_stats
    return result


def export_reports(
    source_dir: Path, destination_dir: Path, *, only: str | None = None,
) -> list[Path]:
    """Crea reportes públicos y evita sobrescribir archivos anteriores."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Falta carpeta de resultados: {source_dir}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    if only is not None and only not in (*SPEC, "e3_genetic.json"):
        raise ValueError("Nombre de reporte no admitido")
    names = (only,) if only else (*SPEC, "e3_genetic.json")
    for filename in names:
        source = source_dir / filename
        if not source.is_file():
            LOG.warning("Sin reporte local %s: se omite", filename)
            continue
        destination = destination_dir / filename
        safe = sanitize_report(filename, read_json_object(source))
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(safe, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        exported.append(destination)
        LOG.info("Reporte público generado: %s", destination.name)
    return exported


def main() -> int:
    """Exporta reportes locales y registra errores sin imprimir datos sensibles."""
    parser = argparse.ArgumentParser(description="Exportar E0/E1/E2/E3 sin datos brutos")
    parser.add_argument("--source", type=Path, default=BASE_DIR / "results")
    parser.add_argument("--dest", type=Path, default=BASE_DIR / "public_reports")
    parser.add_argument("--only", choices=(*SPEC, "e3_genetic.json"),
                        help="Exportar solo un experimento y conservar E0-E2 existentes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        output = export_reports(args.source, args.dest, only=args.only)
    except (OSError, ValueError, TypeError) as exc:
        LOG.error("No fue posible exportar reportes: %s", exc)
        return 1
    print(f"Reportes públicos generados: {len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
