"""CLI de auditoría, benchmark E0 y comparación optimizada E1; sin entrenar."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .benchmark import BenchmarkOptions, run_benchmark
from .compare import CompareOptions, run_comparison
from .data import load_workload
from .integrity import BASE_DIR, audit_reference, load_trusted_model, read_json_object

LOG = logging.getLogger("nexus_optimize")


def _batch_sizes(value: str) -> tuple[int, ...]:
    """Convierte lista CSV de tamaños de lote en enteros positivos."""
    try:
        sizes = tuple(int(p.strip()) for p in value.split(","))
        if not sizes or any(p <= 0 for p in sizes):
            raise ValueError
        return sizes
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Indique lotes separados por coma, por ejemplo 1,16,64,256"
        ) from exc


def _write_report(path: Path, payload: dict, overwrite: bool = False) -> None:
    """Escribe JSON de modo exclusivo; --overwrite autoriza reemplazarlo."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if overwrite:
        temp = path.with_name(path.name + ".tmp")
        try:
            with temp.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    else:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)


def build_parser() -> argparse.ArgumentParser:
    """Construye CLI didáctica con validación declarativa."""
    parser = argparse.ArgumentParser(
        prog="nexus_optimize", description="Módulo 2 - benchmark y optimización reproducibles"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Audita metadatos e integridad sin deserializar")
    benchmark = sub.add_parser("benchmark", help="Mide inferencia real del modelo M1")
    benchmark.add_argument("--trust-local-model", action="store_true",
                           help="Autorizo cargar el archivo joblib de mi proyecto")
    benchmark.add_argument("--source", choices=("demo", "sentiment140"),
                           default="demo")
    benchmark.add_argument("--data", type=Path, help="Ruta a un CSV Sentiment140")
    benchmark.add_argument("--sample-size", type=int, default=2000)
    benchmark.add_argument("--items", type=int, default=1000)
    benchmark.add_argument("--batch-sizes", type=_batch_sizes, default=(1, 16, 64, 256))
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--warmup", type=int, default=2)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--output", type=Path,
                           default=BASE_DIR / "results" / "baseline.json")
    benchmark.add_argument("--overwrite", action="store_true")

    compare = sub.add_parser(
        "compare", help="E1: compara referencia, deduplicación y caché SIN reentrenar"
    )
    compare.add_argument("--trust-local-model", action="store_true")
    compare.add_argument("--source", choices=("demo", "sentiment140"), default="demo")
    compare.add_argument("--data", type=Path)
    compare.add_argument("--sample-size", type=int, default=2000)
    compare.add_argument("--items", type=int, default=2000)
    compare.add_argument("--batch-sizes", type=_batch_sizes, default=(16, 64, 256))
    compare.add_argument("--repeats", type=int, default=3)
    compare.add_argument("--warmup", type=int, default=1)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--cache-capacity", type=int, default=4096)
    compare.add_argument("--repeat-fraction", type=float, default=0.5)
    compare.add_argument("--output", type=Path,
                         default=BASE_DIR / "results" / "comparison_e1.json")
    compare.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ejecuta auditoría o benchmark con manejo de errores y trazabilidad."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        audit = audit_reference()
        if args.command == "audit":
            print(json.dumps(audit, indent=2, ensure_ascii=False))
            import sklearn
            if sklearn.__version__ != audit["expected_sklearn"]:
                LOG.warning(
                    "Versión local %s distinta de la requerida %s: "
                    "no ejecute benchmark en este entorno.",
                    sklearn.__version__, audit["expected_sklearn"],
                )
            return 0

        if not args.trust_local_model:
            raise PermissionError(
                "Use --trust-local-model solo si reconoce y confía "
                "en el modelo joblib proporcionado."
            )
        if args.source == "sentiment140" and args.data is None:
            raise ValueError("--data es obligatorio al usar --source sentiment140")
        if args.source == "demo" and args.data is not None:
            raise ValueError("--data solo se admite con --source sentiment140")

        if args.command == "benchmark":
            options = BenchmarkOptions(
                items=args.items, batch_sizes=args.batch_sizes,
                repeats=args.repeats, warmup=args.warmup, seed=args.seed,
            )
        else:
            options = CompareOptions(
                items=args.items, batch_sizes=args.batch_sizes,
                repeats=args.repeats, warmup=args.warmup, seed=args.seed,
                cache_capacity=args.cache_capacity,
                repeat_fraction=args.repeat_fraction,
            )
        source_path = (
            args.data if args.source == "sentiment140"
            else BASE_DIR / "fixtures" / "english_demo.csv"
        )
        LOG.info("Leyendo corpus %s; semilla %s", args.source, args.seed)
        workload = load_workload(source_path, args.source, args.sample_size, args.seed)
        LOG.info("Cargando artefacto local autorizado y verificando versión")
        model = load_trusted_model()
        LOG.info("Ejecutando %d configuraciones; %d repeticiones",
                 len(options.batch_sizes), options.repeats)
        if args.command == "compare":
            measured = run_comparison(model, workload, options)
            measured["model_sha256"] = audit["model_sha256"]
            _write_report(args.output, measured, args.overwrite)
            print("\nRESULTADOS E1 — mismo modelo, control pareado en cada lote")
            print(f"{'Escenario':>10} {'Lote':>6} {'Estrategia':>12} "
                  f"{'tweets/s':>12} {'Cambio':>10} {'p95 ms':>10}")
            for row in measured["results"]:
                print(f"{row['scenario']:>10} {row['batch_size']:>6} "
                      f"{row['strategy']:>12} "
                      f"{row['throughput_median_tweets_per_s']:>12.2f} "
                      f"{row['speedup_vs_paired_reference_x']:>9.3f}x "
                      f"{row['request_latency_p95_ms']:>10.3f}")
            print("\nVerificación: todas las estrategias devolvieron etiquetas "
                  "exactamente iguales a la referencia.")
        else:
            measured = run_benchmark(model, workload, options)
            measured["model_sha256"] = audit["model_sha256"]
            _write_report(args.output, measured, args.overwrite)
            print("\nRESULTADOS — mismo modelo original, distinta agrupación")
            print(f"{'Lote':>6} {'p95 lote (ms)':>16} {'tweets/s':>14} {'ms/tweet':>12}")
            for item in measured["results"]:
                print(f"{item['batch_size']:>6} "
                      f"{item['request_latency_p95_ms']:>16.4f} "
                      f"{item['throughput_median_tweets_per_s']:>14.2f} "
                      f"{item['amortized_median_ms_per_tweet']:>12.6f}")
        print(f"\\nReporte creado: {args.output.resolve()}")
        return 0
    except (OSError, ValueError, TypeError, RuntimeError, PermissionError) as error:
        LOG.error("%s: %s", type(error).__name__, error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
