"""Benchmark reproducible: idénticos textos, variando tamaño del lote."""
from __future__ import annotations

import os
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import psutil

from .data import Workload, make_equal_workload

MB = 1024 * 1024


@dataclass(frozen=True)
class BenchmarkOptions:
    """Parámetros del experimento; todos los lotes procesan el mismo volumen."""
    items: int = 1000
    batch_sizes: tuple[int, ...] = (1, 16, 64, 256)
    warmup: int = 2
    repeats: int = 5
    seed: int = 42

    def __post_init__(self) -> None:
        if not 1 <= self.items <= 100_000:
            raise ValueError("items fuera de rango")
        if not 1 <= self.repeats <= 30 or not 0 <= self.warmup <= 20:
            raise ValueError("Número inválido de repeticiones o calentamientos")
        if not self.batch_sizes or len(set(self.batch_sizes)) != len(self.batch_sizes):
            raise ValueError("Se requiere al menos un lote, sin duplicados")
        if any(not 1 <= b <= 10_000 for b in self.batch_sizes):
            raise ValueError("Los tamaños de lote deben estar entre 1 y 10.000")


def percentile(values: Sequence[float], percent: float) -> float:
    """Percentil lineal interpolado sin dependencias científicas adicionales."""
    if not values or not 0 <= percent <= 100:
        raise ValueError("Percentil inválido o muestra vacía")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def _chunks(texts: Sequence[str], size: int) -> list[Sequence[str]]:
    """Divide en lotes; no transforma los textos ni modifica el modelo."""
    return [texts[start:start + size] for start in range(0, len(texts), size)]


def _predict_checked(model: Any, batch: Sequence[str]) -> None:
    """Asegura que el modelo realmente devuelve una predicción por entrada."""
    prediction = model.predict(list(batch))
    if len(prediction) != len(batch):
        raise ValueError("El modelo devolvió una cantidad de predicciones inválida")


def run_benchmark(
    model: Any,
    workload: Workload,
    options: BenchmarkOptions,
) -> dict[str, Any]:
    """Mide latencia de solicitudes/lotes, throughput y RSS observado.

    No registra textos ni utiliza etiquetas para calcular métricas de calidad.
    """
    if not callable(getattr(model, "predict", None)):
        raise TypeError("El modelo no implementa predict")
    texts = make_equal_workload(workload, options.items)
    process = psutil.Process(os.getpid())
    results: list[dict[str, Any]] = []

    for batch_size in options.batch_sizes:
        batches = _chunks(texts, batch_size)
        for _ in range(options.warmup):
            _predict_checked(model, batches[0])
        all_call_ms: list[float] = []
        throughputs: list[float] = []
        whole_run_ms: list[float] = []
        observed_peak = process.memory_info().rss / MB
        rss_start = observed_peak

        for _ in range(options.repeats):
            total_ns = 0
            for batch in batches:
                started = time.perf_counter_ns()
                _predict_checked(model, batch)
                duration_ns = time.perf_counter_ns() - started
                total_ns += duration_ns
                all_call_ms.append(duration_ns / 1_000_000)
                observed_peak = max(observed_peak, process.memory_info().rss / MB)
            if total_ns <= 0:
                raise RuntimeError("El temporizador devolvió tiempo no positivo")
            duration_s = total_ns / 1_000_000_000
            throughputs.append(options.items / duration_s)
            whole_run_ms.append(duration_s * 1000)

        results.append({
            "batch_size": batch_size,
            "calls_per_repeat": len(batches),
            "items_per_repeat": options.items,
            "total_time_median_ms": round(statistics.median(whole_run_ms), 4),
            "throughput_median_tweets_per_s": round(statistics.median(throughputs), 2),
            "amortized_median_ms_per_tweet": round(
                statistics.median(whole_run_ms) / options.items, 6
            ),
            "request_latency_p50_ms": round(percentile(all_call_ms, 50), 5),
            "request_latency_p95_ms": round(percentile(all_call_ms, 95), 5),
            "rss_before_mb": round(rss_start, 2),
            "rss_observed_peak_mb": round(observed_peak, 2),
        })

    import sklearn
    return {
        "schema_version": 1,
        "status": "measured",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E0 - inferencia original, sin optimizar",
        "data_provenance": {
            "source_type": workload.source_type,
            "source_file_sha256": workload.file_sha256,
            "source_valid_rows_seen": workload.valid_rows_seen,
            "source_invalid_rows_seen": workload.invalid_rows_seen,
            "sampled_unique_positions": len(workload.texts),
            "items_per_repeat": options.items,
            "reused_sample": options.items > len(workload.texts),
            "warning": (
                "Datos didácticos sintéticos: NO son evidencia sobre Sentiment140"
                if workload.source_type == "demo"
                else "Muestra de Sentiment140 para latencia; NO se calcula F1"
            ),
        },
        "methodology": {
            "repeats": options.repeats,
            "warmup_per_batch": options.warmup,
            "batch_sizes": list(options.batch_sizes),
            "seed": options.seed,
            "timed_scope": "model.predict() local; no incluye CSV, red, HTTP ni GUI",
            "latency_definition": "p95 por solicitud de inferencia (un lote)",
            "throughput_definition": "mediana de items/segundo por repetición",
            "memory_definition": "RSS observado después de cada llamada, no pico continuo",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "sklearn": sklearn.__version__,
            "psutil": psutil.__version__,
        },
        "results": results,
    }
