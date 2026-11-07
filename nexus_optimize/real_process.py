"""E8 — Benchmark de la tarea REAL de inferencia de NEXUS ANALYZE M1.

Compara (mismos tweets reales, mismo modelo original, misma máquina):
1) Flujo nativo de classify() M1: normalizar y predecir tweet por tweet.
2) Capacidad nativa de Pipeline.predict() M1: normalizar y predecir todo el lote.
3) Candidato M2: normalizar y predecir en bloques acotados SIN caché ni repetición.

El tamaño de bloque se selecciona sólo con datos de ajuste de *ejecución*;
las métricas se calculan en otro grupo disjunto de tweets naturales únicos.
NO mide HTTP, NO recalcula F1, NO reentrena.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import psutil

from .benchmark import percentile
from .fast_nlp import normalize_tweet_fast
from .integrity import (
    audit_reference, load_trusted_model, sha256_file,
)
from .nlp import normalize_tweet
from .optimization import validated_labels

LOG = logging.getLogger("nexus_optimize.real_process")
_MAX_CSV_BYTES = 350_000_000


@dataclass(frozen=True)
class RealOptions:
    """Opciones acotadas y reproducibles para el benchmark de inferencia."""
    sizes: tuple[int, ...] = (1, 100, 1000, 10000)
    candidate_batches: tuple[int, ...] = (16, 64, 256, 1024)
    sample_size: int = 25_000
    repeats: int = 5
    tune_repeats: int = 2
    individual_max: int = 1000
    seed: int = 42
    improvement_threshold_pct: float = 10.0

    def __post_init__(self) -> None:
        if (not self.sizes or len(set(self.sizes)) != len(self.sizes)
                or min(self.sizes) < 1 or max(self.sizes) > 30_000):
            raise ValueError("sizes: enteros distintos entre 1 y 30000")
        if (not self.candidate_batches
                or len(set(self.candidate_batches)) != len(self.candidate_batches)
                or min(self.candidate_batches) < 1
                or max(self.candidate_batches) > 30_000):
            raise ValueError("candidate_batches: enteros distintos entre 1 y 30000")
        if not 2 * max(self.sizes) <= self.sample_size <= 100_000:
            raise ValueError("sample_size debe cubrir dos grupos disjuntos de tamaño max(sizes)")
        if not 3 <= self.repeats <= 30 or not 1 <= self.tune_repeats <= 10:
            raise ValueError("repeats debe estar en 3..30 y tune_repeats en 1..10")
        if not 0 <= self.individual_max <= 30_000:
            raise ValueError("individual_max fuera de rango")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed fuera de rango")
        if not 0 < self.improvement_threshold_pct <= 100:
            raise ValueError("Umbral de mejora fuera de rango")


@dataclass(frozen=True)
class RawWorkload:
    """Muestra RAW de origen real: las entradas aún no se han normalizado."""
    tuning: tuple[str, ...]
    evaluation: tuple[str, ...]
    file_sha256: str
    source_valid_rows: int
    invalid_rows: int
    natural_duplicate_positions: int
    source_sample_size: int

    def __post_init__(self) -> None:
        if not self.tuning or not self.evaluation:
            raise ValueError("Se requieren muestras de ajuste y evaluación")
        a = {normalize_tweet(s) for s in self.tuning}
        b = {normalize_tweet(s) for s in self.evaluation}
        if len(a) != len(self.tuning) or len(b) != len(self.evaluation) or a & b:
            raise ValueError("Las muestras deben ser únicas y disjuntas")


def load_real_tweets(path: Path, options: RealOptions) -> RawWorkload:
    """Muestreo uniforme de Sentiment140, sin transformar ni duplicar textos.

    El lector reserva filas originales RAW y sólo normaliza para descartar
    duplicados. CSV con formato Sentiment140 (6 campos y etiquetas 0/4).
    """
    resolved = path.expanduser().resolve()
    if resolved.suffix.lower() != ".csv" or not resolved.is_file():
        raise FileNotFoundError("Indica un CSV Sentiment140 local válido")
    if not 0 < resolved.stat().st_size <= _MAX_CSV_BYTES:
        raise ValueError("Dataset vacío o mayor de 350 MB")
    rng = random.Random(options.seed)
    reservoir: list[str] = []
    valid = invalid = 0
    with resolved.open("r", encoding="latin-1", newline="") as source:
        for row in csv.reader(source):
            if len(row) != 6 or row[0].strip() not in {"0", "4"}:
                invalid += 1
                continue
            text = row[5]
            # Mantener la entrada RAW. No se filtra por la clase predicha.
            if not 1 <= len(text) <= 1000 or not normalize_tweet(text):
                invalid += 1
                continue
            valid += 1
            if len(reservoir) < options.sample_size:
                reservoir.append(text)
            else:
                j = rng.randrange(valid)
                if j < options.sample_size:
                    reservoir[j] = text
    if len(reservoir) < options.sample_size:
        raise ValueError("El dataset no contiene suficientes publicaciones válidas")
    rng.shuffle(reservoir)
    seen: set[str] = set()
    unique: list[str] = []
    duplicates = 0
    for raw in reservoir:
        key = normalize_tweet(raw)
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
            unique.append(raw)
    needed = 2 * max(options.sizes)
    if len(unique) < needed:
        raise ValueError(
            f"Se requieren {needed} tweets distintos para ajuste y evaluación, "
            f"pero se encontraron {len(unique)}. Aumenta --sample-size."
        )
    size = max(options.sizes)
    return RawWorkload(
        tuning=tuple(unique[:size]),
        evaluation=tuple(unique[size:2 * size]),
        file_sha256=sha256_file(resolved),
        source_valid_rows=valid,
        invalid_rows=invalid,
        natural_duplicate_positions=duplicates,
        source_sample_size=options.sample_size,
    )


def native_m1_individual(model: Any, texts: Sequence[str]) -> list[int]:
    """Reproduce classify() M1: normalize_tweet y model.predict([text])."""
    output: list[int] = []
    for raw in texts:
        normalized = normalize_tweet(raw)
        if not normalized:
            raise ValueError("Tweet vacío tras normalizar")
        output.extend(validated_labels(model.predict([normalized]), 1))
    return output


def native_m1_batch(model: Any, texts: Sequence[str]) -> list[int]:
    """Control competitivo: la capacidad de lote YA existe en Pipeline M1."""
    normalized = [normalize_tweet(raw) for raw in texts]
    if not normalized or any(not text for text in normalized):
        raise ValueError("Lote vacío o con textos vacíos")
    return validated_labels(model.predict(normalized), len(normalized))


def optimized_bounded_batch(model: Any, texts: Sequence[str], size: int) -> list[int]:
    """Procesa todo el flujo M1 en bloques de memoria acotada, sin cachear."""
    if not 1 <= size <= 30_000 or not texts:
        raise ValueError("Tamaño de bloque inválido o entrada vacía")
    out: list[int] = []
    for pos in range(0, len(texts), size):
        raw_chunk = texts[pos:pos + size]
        normalized = [normalize_tweet_fast(raw) for raw in raw_chunk]
        if any(not text for text in normalized):
            raise ValueError("Tweet vacío después de la normalización")
        out.extend(validated_labels(model.predict(normalized), len(normalized)))
    return out


def _measure(func: Callable[[], list[int]]) -> tuple[list[int], float, float]:
    """Cronometra toda la tarea; el RSS observado no es pico continuo."""
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / (1024**2)
    started = time.perf_counter_ns()
    labels = func()
    ms = (time.perf_counter_ns() - started) / 1e6
    after = process.memory_info().rss / (1024**2)
    if ms <= 0:
        raise RuntimeError("Resolución insuficiente del cronómetro")
    return labels, ms, max(before, after)


def _run_paired(
    original: Callable[[], list[int]],
    candidate: Callable[[], list[int]],
    repeats: int,
) -> dict[str, Any]:
    """Orden AB/BA alternado; comprueba equivalencia en CADA repetición."""
    orig_times: list[float] = []
    cand_times: list[float] = []
    rss_samples: list[float] = []
    wins = 0
    for turn in range(repeats):
        if turn % 2 == 0:
            a, a_ms, a_rss = _measure(original)
            b, b_ms, b_rss = _measure(candidate)
        else:
            b, b_ms, b_rss = _measure(candidate)
            a, a_ms, a_rss = _measure(original)
        if a != b:
            raise RuntimeError("Las predicciones del candidato difieren de M1")
        orig_times.append(a_ms)
        cand_times.append(b_ms)
        rss_samples.extend([a_rss, b_rss])
        wins += int(b_ms < a_ms)
    a_median, b_median = statistics.median(orig_times), statistics.median(cand_times)
    change_pct = (b_median / a_median - 1) * 100
    return {
        "reference_median_ms": round(a_median, 5),
        "candidate_median_ms": round(b_median, 5),
        "reference_job_p95_ms": round(percentile(orig_times, 95), 5),
        "candidate_job_p95_ms": round(percentile(cand_times, 95), 5),
        "runtime_change_pct": round(change_pct, 3),
        "throughput_improvement_pct": round((a_median / b_median - 1) * 100, 3),
        "candidate_faster_pairs": wins,
        "pairs": repeats,
        "exact_predictions": True,
        "rss_observed_max_mb": round(max(rss_samples), 2),
        "reference_times_ms": [round(v, 5) for v in orig_times],
        "candidate_times_ms": [round(v, 5) for v in cand_times],
    }


def _profile_stages(model: Any, raw: Sequence[str]) -> dict[str, Any]:
    """Aísla normalización, TF-IDF y regresión logística del pipeline M1."""
    if not hasattr(model, "named_steps"):
        return {"status": "unavailable", "reason": "Modelo sin named_steps"}
    transformer = model.named_steps.get("tfidf")
    classifier = model.named_steps.get("classifier")
    if transformer is None or classifier is None:
        return {"status": "unavailable", "reason": "No se identificaron las etapas TF-IDF y clasificador"}

    t0 = time.perf_counter_ns()
    normalized = [normalize_tweet(text) for text in raw]
    t1 = time.perf_counter_ns()
    vectors = transformer.transform(normalized)
    t2 = time.perf_counter_ns()
    labels = validated_labels(classifier.predict(vectors), len(raw))
    t3 = time.perf_counter_ns()
    if labels != native_m1_batch(model, raw):
        raise RuntimeError("El perfil separado no reproduce Pipeline.predict()")
    fast_start = time.perf_counter_ns()
    fast_normalized = [normalize_tweet_fast(text) for text in raw]
    fast_ms = (time.perf_counter_ns() - fast_start) / 1e6
    if fast_normalized != normalized:
        raise RuntimeError("Normalización rápida no equivale a normalización M1")
    stages = {
        "normalization_ms": round((t1 - t0) / 1e6, 5),
        "tfidf_ms": round((t2 - t1) / 1e6, 5),
        "logistic_regression_ms": round((t3 - t2) / 1e6, 5),
    }
    bottleneck = max(stages, key=stages.get)
    return {
        "status": "measured",
        "items": len(raw),
        **stages,
        "normalization_fast_ms": round(fast_ms, 5),
        "normalization_exact_equivalence": True,
        "slowest_stage": bottleneck,
        "notice": "Diagnóstico por etapas; no es el tiempo exacto del Pipeline completo",
    }


def run_real_process(
    model: Any, workload: RawWorkload, options: RealOptions,
) -> dict[str, Any]:
    """Ajusta tamaños de bloque en tuning y evalúa en tweets disjuntos.

    No emplea caché, no construye duplicados artificiales y no modifica M1.
    """
    if not callable(getattr(model, "predict", None)):
        raise TypeError("El modelo debe implementar predict")
    if len(workload.tuning) < max(options.sizes) or len(workload.evaluation) < max(options.sizes):
        raise ValueError("Las muestras son menores que la configuración solicitada")
    rows: list[dict[str, Any]] = []

    for n in sorted(options.sizes):
        tune = workload.tuning[:n]
        evaluation = workload.evaluation[:n]
        available = sorted(set(min(n, b) for b in options.candidate_batches) | {n})
        # Este valor procede exclusivamente de las muestras de ajuste.
        best_chunk, best_tune_ms = n, float("inf")
        tune_results: list[dict[str, float | int]] = []
        for chunk in available:
            func = lambda ch=chunk: optimized_bounded_batch(model, tune, ch)
            func()
            times = [_measure(func)[1] for _ in range(options.tune_repeats)]
            candidate_ms = statistics.median(times)
            tune_results.append({"chunk": chunk, "median_ms": round(candidate_ms, 5)})
            if candidate_ms < best_tune_ms:
                best_chunk, best_tune_ms = chunk, candidate_ms
        ref = lambda: native_m1_batch(model, evaluation)
        opt = lambda: optimized_bounded_batch(model, evaluation, best_chunk)
        ref()
        opt()
        paired = _run_paired(ref, opt, options.repeats)
        paired["reference_throughput_tweets_s"] = round(n * 1000 / paired["reference_median_ms"], 2)
        paired["candidate_throughput_tweets_s"] = round(n * 1000 / paired["candidate_median_ms"], 2)

        # La prueba individual es la semántica nativa de classify() de M1,
        # pero NO es un control válido para atribuir un algoritmo más rápido.
        single_median: float | None = None
        if n <= options.individual_max:
            individual = lambda: native_m1_individual(model, evaluation)
            indiv_labels = individual()
            if indiv_labels != ref():
                raise RuntimeError("La inferencia individual M1 difiere del lote M1")
            single_times = [_measure(individual)[1] for _ in range(options.repeats)]
            single_median = round(statistics.median(single_times), 5)
        improved = (
            paired["throughput_improvement_pct"] >= options.improvement_threshold_pct
            and paired["candidate_faster_pairs"] >= (options.repeats // 2 + 1)
        )
        rows.append({
            "items": n,
            "unique_input": True,
            "selected_chunk_from_tuning": best_chunk,
            "candidate_variants_tuned": tune_results,
            "m1_native_single_median_ms": single_median,
            "m1_native_full_batch_median_ms": paired["reference_median_ms"],
            "m2_bounded_batch_median_ms": paired["candidate_median_ms"],
            "evaluation": paired,
            "decision": "improved_vs_native_m1_batch" if improved else "not_proven_improved",
        })
        LOG.info(
            "E8 n=%d chunk=%d | M1 lote=%.3f ms | M2=%.3f ms | Δthroughput=%+.2f%% | %s",
            n, best_chunk, paired["reference_median_ms"], paired["candidate_median_ms"],
            paired["throughput_improvement_pct"], rows[-1]["decision"],
        )

    by_n = {row["items"]: row for row in rows}
    # Un resultado favorable en sólo uno de los tamaños NO es éxito global.
    target_sizes = [n for n in (1000, 10000) if n in by_n]
    global_improved = (
        len(target_sizes) == 2
        and all(by_n[n]["decision"] == "improved_vs_native_m1_batch" for n in target_sizes)
        and all(row["evaluation"]["runtime_change_pct"] <= 10 for row in rows)
    )
    profile = _profile_stages(model, workload.evaluation[:max(options.sizes)])
    return {
        "schema_version": 8,
        "status": "measured",
        "experiment": "E8 - proceso real M1 vs normalizacion rapida y lotes M2, SIN cache",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "candidate_validated_local" if global_improved else "not_improved",
        "data": {
            "source": "sentiment140",
            "source_sha256": workload.file_sha256,
            "valid_rows_seen": workload.source_valid_rows,
            "invalid_rows_seen": workload.invalid_rows,
            "sampled_raw_positions": workload.source_sample_size,
            "natural_duplicates_excluded": workload.natural_duplicate_positions,
            "tuning_unique_items": len(workload.tuning),
            "evaluation_unique_items": len(workload.evaluation),
            "synthetic_duplication": False,
            "tuning_and_evaluation_disjoint": True,
        },
        "methodology": {
            "pipeline": "normalizacion M1 -> TF-IDF M1 -> Regresion logistica M1",
            "comparison": "mismo pipeline M1, mismo equipo, mismos datos, A/B alternado",
            "primary_baseline": "M1 ya admite Pipeline.predict(lista); comparación nativa por lotes",
            "secondary_baseline": "classify() M1 por tweet; NO prueba mejora sobre la capacidad batch M1",
            "candidate": "normalizador equivalente acelerado + chunks ajustados en grupo separado; sin cache",
            "benchmark_scope": "INFERENCIA LOCAL COMPLETA; excluye CSV, HTTP, traduccion y navegador",
            "latency": "p95 del trabajo completo entre repeticiones, NO latencia HTTP por solicitud",
            "throughput": "n / mediana del tiempo total del trabajo",
            "memory": "RSS antes/después; no pico continuo de memoria",
            "tune_repeats": options.tune_repeats,
            "evaluation_repeats": options.repeats,
            "improvement_threshold_pct": options.improvement_threshold_pct,
            "f1": "NO recalculado: sólo equivalencia exacta de etiquetas",
            "selection_bias_control": "ajuste y evaluación con tweets disjuntos, ambos del dataset real",
            "limitations": (
                "La muestra puede incluir tweets usados al entrenar M1; aquí NO se mide F1. "
                "La variabilidad del sistema operativo y CPU exige repetir el experimento."
            ),
        },
        "profile": profile,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "sklearn": __import__("sklearn").__version__,
        },
        "results": rows,
    }


def _list_int(value: str) -> tuple[int, ...]:
    try:
        items = tuple(int(s.strip()) for s in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Se requiere lista de enteros separados por comas") from exc
    if not items:
        raise argparse.ArgumentTypeError("La lista no puede estar vacía")
    return items


def _write_no_overwrite(path: Path, report: dict[str, Any]) -> None:
    """Crea reporte nuevo sin sobrescribir mediciones históricas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    try:
        with path.open("x", encoding="utf-8") as target:
            target.write(payload + "\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Ya existe el reporte: {path.name}. Elige otro nombre; no se sobrescribe."
        ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="CSV real de Sentiment140")
    parser.add_argument("--sample-size", type=int, default=25_000)
    parser.add_argument("--sizes", type=_list_int, default=(1, 100, 1000, 10000))
    parser.add_argument("--candidate-batches", type=_list_int, default=(16, 64, 256, 1024))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tune-repeats", type=int, default=2)
    parser.add_argument("--individual-max", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/e8_real_m1.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        options = RealOptions(
            sizes=args.sizes,
            candidate_batches=args.candidate_batches,
            sample_size=args.sample_size,
            repeats=args.repeats,
            tune_repeats=args.tune_repeats,
            individual_max=args.individual_max,
            seed=args.seed,
        )
        report_path = args.output.expanduser().resolve()
        if report_path.exists():
            raise FileExistsError("El reporte solicitado ya existe; usa otro --output")
        audit = audit_reference()
        LOG.info("Modelo original auditado: %s", audit["status"])
        # El joblib original solo se carga con permiso explícito.
        if os.getenv("NEXUS_TRUST_LOCAL_MODEL") != "1":
            raise PermissionError("Requiere NEXUS_TRUST_LOCAL_MODEL=1 para cargar el modelo M1 local")
        model = load_trusted_model()
        if tuple(model.named_steps) != ("tfidf", "classifier"):
            raise ValueError("El modelo original no es el pipeline M1 esperado")
        texts = load_real_tweets(args.data, options)
        report = run_real_process(model, texts, options)
        report["model_sha256"] = audit["model_sha256"]
        _write_no_overwrite(report_path, report)
        print("\nE8 — MISMO PROCESO M1, SIN CACHÉ NI DATOS ARTIFICIALES")
        print(f"{'Tweets':>7} {'Chunk':>7} {'M1 lote ms':>13} {'M2 ms':>12} {'Δ rend.':>10} {'Conclusión':>23}")
        for row in report["results"]:
            e = row["evaluation"]
            print(
                f"{row['items']:7d} {row['selected_chunk_from_tuning']:7d} "
                f"{e['reference_median_ms']:13.3f} {e['candidate_median_ms']:12.3f} "
                f"{e['throughput_improvement_pct']:+9.2f}% {row['decision']:>23}"
            )
        print("Etapa más lenta:", report["profile"].get("slowest_stage", "no disponible"))
        print("Decisión GLOBAL:", report["decision"])
        print("Reporte:", report_path)
        return 0
    except (OSError, ValueError, RuntimeError, PermissionError, TypeError) as exc:
        LOG.error("E8 abortado: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
