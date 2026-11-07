"""E1: experimento pareado sobre inferencia original y optimizaciones sin entrenar.

La comparación directa E0/E1 se ejecuta en el MISMO proceso, dataset, tamaño
de lote, número de tweets y entorno para evitar comparar máquinas distintas.
"""
from __future__ import annotations

import hashlib
import os
import platform
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

import psutil

from .benchmark import percentile
from .data import Workload, make_equal_workload
from .optimization import InferenceEngine, validated_labels

_MB = 1024 * 1024
STRATEGIES = ("reference", "dedupe", "cache_cold", "cache_warm")
SCENARIOS = ("original", "repeated")


@dataclass(frozen=True)
class CompareOptions:
    """Configuración reproducible de las comparaciones pareadas."""
    items: int = 2000
    batch_sizes: tuple[int, ...] = (16, 64, 256)
    repeats: int = 3
    warmup: int = 1
    cache_capacity: int = 4096
    repeat_fraction: float = 0.5
    seed: int = 42
    scenarios: tuple[str, ...] = SCENARIOS

    def __post_init__(self) -> None:
        """Rechaza configuraciones ambiguas o excesivas."""
        if not 1 <= self.items <= 100_000:
            raise ValueError("items fuera de rango (1..100.000)")
        if not 1 <= self.repeats <= 20 or not 0 <= self.warmup <= 10:
            raise ValueError("repeats o warmup fuera de rango")
        if not self.batch_sizes or len(self.batch_sizes) != len(set(self.batch_sizes)):
            raise ValueError("Se requieren lotes positivos distintos")
        if any(not 1 <= value <= 10_000 for value in self.batch_sizes):
            raise ValueError("Lote fuera de rango (1..10.000)")
        if not 0 <= self.cache_capacity <= 100_000:
            raise ValueError("cache_capacity fuera de rango")
        if not isinstance(self.repeat_fraction, float) or not 0 <= self.repeat_fraction < 1:
            raise ValueError("repeat_fraction debe ser decimal en [0, 1)")
        if not self.scenarios or any(item not in SCENARIOS for item in self.scenarios):
            raise ValueError("Escenario desconocido")


def create_scenario(
    texts: Sequence[str], *, fraction: float, seed: int,
) -> tuple[str, ...]:
    """Construye cargas idénticas en longitud con repetición explícita.

    La fracción indica qué proporción de posiciones se sustituirá por una
    copia de textos existentes. No constituye tráfico real observado.
    """
    if not texts or not 0 <= fraction < 1:
        raise ValueError("Secuencia vacía o proporción inválida")
    rng = random.Random(seed)
    n = len(texts)
    n_copies = int(n * fraction)
    source = list(texts[: n - n_copies])
    scenario = source + [rng.choice(source) for _ in range(n_copies)]
    rng.shuffle(scenario)
    return tuple(scenario)


def chunks(texts: Sequence[str], size: int) -> list[Sequence[str]]:
    """Separa una secuencia en lotes preservando el orden original."""
    if size <= 0:
        raise ValueError("Tamaño de lote inválido")
    return [texts[i:i + size] for i in range(0, len(texts), size)]


def _create_engine(model: Any, strategy: str, capacity: int) -> InferenceEngine:
    """Selecciona el tratamiento; el estimador original nunca se altera."""
    if strategy not in STRATEGIES or strategy == "reference":
        raise ValueError("La estrategia requiere un motor optimizado")
    return InferenceEngine(
        model,
        deduplicate=True,
        cache_capacity=0 if strategy == "dedupe" else capacity,
    )


def _execute_repeat(
    model: Any,
    texts: tuple[str, ...],
    batches: list[Sequence[str]],
    strategy: str,
    options: CompareOptions,
) -> tuple[list[int], list[float], float, float, dict[str, int]]:
    """Mide exclusivamente las llamadas de inferencia y verifica salidas."""
    engine = (
        None if strategy == "reference"
        else _create_engine(model, strategy, options.cache_capacity)
    )
    if engine is not None and strategy == "cache_warm":
        for part in batches:
            engine.predict(part)
    elif options.warmup:
        for _ in range(options.warmup):
            if engine is None:
                model.predict(list(batches[0]))
            else:
                engine.predict(batches[0])
        if engine is not None:
            engine.reset()  # Garantiza caché fría tras calentamiento

    process = psutil.Process(os.getpid())
    observed = process.memory_info().rss / _MB
    predictions: list[int] = []
    call_latencies: list[float] = []
    calls_to_model = 0
    model_items = 0
    cache_hits = 0
    duplicates = 0
    total_ns = 0

    for part in batches:
        t0 = time.perf_counter_ns()
        if engine is None:
            out = model.predict(list(part))
            metrics = None
        else:
            out, metrics = engine.predict_with_metrics(part)
        elapsed = time.perf_counter_ns() - t0
        if elapsed <= 0:
            raise RuntimeError("Temporizador sin resolución suficiente")
        # Validación después de detener el reloj, idéntica para todos.
        predictions.extend(validated_labels(out, len(part)))
        call_latencies.append(elapsed / 1_000_000)
        total_ns += elapsed
        if metrics is None:
            calls_to_model += 1
            model_items += len(part)
        else:
            calls_to_model += metrics.model_calls
            model_items += metrics.model_items
            cache_hits += metrics.cache_hits
            duplicates += metrics.duplicate_reuses

    observed = max(observed, process.memory_info().rss / _MB)
    if len(predictions) != len(texts):
        raise ValueError("La estrategia omitió predicciones")
    return (
        predictions, call_latencies, total_ns / 1e9, observed,
        {
            "model_calls": calls_to_model, "model_items": model_items,
            "cache_hits": cache_hits, "batch_duplicates_reused": duplicates,
        },
    )


def run_comparison(
    model: Any, workload: Workload, options: CompareOptions,
) -> dict[str, Any]:
    """Ejecuta controles pareados y exige igualdad exacta de predicciones.

    Retorna solo estadísticas agregadas y huellas. No persiste textos.
    """
    if not callable(getattr(model, "predict", None)):
        raise TypeError("El modelo no implementa predict")
    base = make_equal_workload(workload, options.items)
    rows: list[dict[str, Any]] = []
    scenario_metadata: list[dict[str, Any]] = []

    for scenario in options.scenarios:
        texts = (
            base if scenario == "original"
            else create_scenario(base, fraction=options.repeat_fraction, seed=options.seed)
        )
        observed_duplicates = 1 - len(set(texts)) / len(texts)
        scenario_metadata.append({
            "name": scenario, "items": len(texts),
            "unique_texts": len(set(texts)),
            "observed_duplicate_fraction": round(observed_duplicates, 4),
            "constructed_repetition": scenario == "repeated",
        })
        for batch_size in options.batch_sizes:
            batches = chunks(texts, batch_size)
            reference_labels: list[int] | None = None
            reference_speed: float | None = None

            for strategy in STRATEGIES:
                all_latencies: list[float] = []
                throughputs: list[float] = []
                observed_rss = 0.0
                totals = {
                    "model_calls": 0, "model_items": 0,
                    "cache_hits": 0, "batch_duplicates_reused": 0,
                }
                for _ in range(options.repeats):
                    labels, latencies, seconds, rss, counters = _execute_repeat(
                        model, texts, batches, strategy, options,
                    )
                    if reference_labels is None:
                        reference_labels = labels
                    elif labels != reference_labels:
                        raise RuntimeError(
                            f"Predicciones distintas: {scenario}/{batch_size}/{strategy}"
                        )
                    all_latencies.extend(latencies)
                    throughputs.append(len(texts) / seconds)
                    observed_rss = max(observed_rss, rss)
                    for key in totals:
                        totals[key] += counters[key]

                median_rate = statistics.median(throughputs)
                if strategy == "reference":
                    reference_speed = median_rate
                assert reference_speed is not None and reference_labels is not None
                rows.append({
                    "scenario": scenario,
                    "batch_size": batch_size,
                    "strategy": strategy,
                    "items_per_repeat": len(texts),
                    "calls_per_repeat": len(batches),
                    "request_latency_p50_ms": round(percentile(all_latencies, 50), 5),
                    "request_latency_p95_ms": round(percentile(all_latencies, 95), 5),
                    "throughput_median_tweets_per_s": round(median_rate, 2),
                    "amortized_ms_per_tweet": round(1000 / median_rate, 6),
                    "speedup_vs_paired_reference_x": round(median_rate / reference_speed, 4),
                    "observed_rss_mb": round(observed_rss, 2),
                    "exact_prediction_equivalence": True,
                    "predictions_sha256": hashlib.sha256(
                        bytes(reference_labels)
                    ).hexdigest(),
                    "counters_total": totals,
                })

    import sklearn
    return {
        "schema_version": 2,
        "status": "measured",
        "experiment": "E1 - deduplicacion y cache LRU sin reentrenar",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_provenance": {
            "source_type": workload.source_type,
            "source_file_sha256": workload.file_sha256,
            "valid_rows_seen": workload.valid_rows_seen,
            "sampled_positions": len(workload.texts),
            "reused_sample": options.items > len(workload.texts),
            "warning": "demo sintética, no evidencia de Sentiment140"
            if workload.source_type == "demo"
            else "muestra Sentiment140; no equivale a tráfico real ni recalcula F1",
        },
        "scenarios": scenario_metadata,
        "methodology": {
            "repeats": options.repeats, "warmup": options.warmup,
            "batch_sizes": list(options.batch_sizes),
            "seed": options.seed, "cache_capacity": options.cache_capacity,
            "constructed_repeat_fraction": options.repeat_fraction,
            "strategies": list(STRATEGIES),
            "timed_scope": "solo llamadas locales de inferencia, sin CSV/HTTP/UI",
            "latency_definition": "p95 por llamada (lote); no p95 por tweet",
            "throughput_definition": "mediana de tweets/s por repeticion",
            "cache_cold_definition": "caché vacía al inicio de cada repetición",
            "cache_warm_definition": "se precargan todos los lotes antes de medir",
            "memory_definition": "RSS muestreado antes/después, no pico continuo",
            "quality_definition": "equivalencia exacta de etiquetas frente al control; F1 histórica no recalculada",
            "fairness": "control original cronometrado en la misma ejecución, mismo dataset y lotes",
            "limitation": "orden fijo entre estrategias; cachés de CPU, frecuencia y temperatura pueden variar",
        },
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "logical_cpus": os.cpu_count(), "sklearn": sklearn.__version__,
        },
        "results": rows,
    }
