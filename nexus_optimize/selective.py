"""Enrutador selectivo de inferencia: referencia para lotes únicos, LRU si hay duplicados.

El modelo, el vectorizador y sus etiquetas quedan intactos. La decisión se basa
solo en estadísticas por lote: no usa identidad del autor ni datos externos.
La política es conservadora: no deduce que una caché caliente mejore la p95 HTTP.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from .optimization import CallMetrics, InferenceEngine, validate_texts, validated_labels


@dataclass(frozen=True)
class SelectiveConfig:
    """Umbral de duplicados para activar LRU en un lote de dos o más elementos."""
    cache_capacity: int = 4096
    min_duplicates: int = 2
    min_duplicate_fraction: float = 0.01

    def __post_init__(self) -> None:
        if isinstance(self.cache_capacity, bool) or not isinstance(self.cache_capacity, int):
            raise ValueError("cache_capacity debe ser entero")
        if not 1 <= self.cache_capacity <= 100_000:
            raise ValueError("cache_capacity debe estar entre 1 y 100000")
        if isinstance(self.min_duplicates, bool) or not isinstance(self.min_duplicates, int):
            raise ValueError("min_duplicates debe ser entero")
        if not 1 <= self.min_duplicates <= 256:
            raise ValueError("min_duplicates fuera de rango")
        if not isinstance(self.min_duplicate_fraction, (int, float)) or isinstance(self.min_duplicate_fraction, bool):
            raise ValueError("min_duplicate_fraction debe ser un número")
        if not 0 < self.min_duplicate_fraction <= 1:
            raise ValueError("min_duplicate_fraction debe estar en (0,1]")


def config_from_env(getenv: Callable[[str, str], str]) -> SelectiveConfig:
    """Valida todos los parámetros, sin leer ni registrar valores sensibles."""
    try:
        return SelectiveConfig(
            cache_capacity=int(getenv("NEXUS_CACHE_CAPACITY", "4096")),
            min_duplicates=int(getenv("NEXUS_ADAPTIVE_MIN_DUPLICATES", "2")),
            min_duplicate_fraction=float(
                getenv("NEXUS_ADAPTIVE_MIN_DUPLICATE_FRACTION", "0.01")
            ),
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Configuración adaptativa inválida") from exc


def choose_inference_path(texts: Sequence[str], config: SelectiveConfig) -> str:
    """Función pura: selecciona 'reference' o 'lru' sin ejecutar predicciones.

    Para una solicitud individual usamos LRU y permitimos recalcular/recordar
    su etiqueta. Para lotes, activamos LRU solamente si hay suficientes textos
    repetidos DENTRO de la misma solicitud. Si son distintos, referencia.
    """
    clean = validate_texts(texts, maximum=256)
    if len(clean) == 1:
        return "lru"
    duplicate_count = len(clean) - len(set(clean))
    if (duplicate_count >= config.min_duplicates
            and duplicate_count / len(clean) >= config.min_duplicate_fraction):
        return "lru"
    return "reference"


class SelectiveInferenceEngine:
    """Compone modelo y caché LRU, con estadísticas y rutas independientes.

    La LRU protege sus estructuras con un candado breve y ejecuta el modelo
    fuera de él; puede repetir inferencias concurrentes de una misma clave.
    No promete reducción automática de p95 HTTP.
    """

    def __init__(self, model: Any, config: SelectiveConfig | None = None):
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model debe implementar predict")
        self._model = model
        self._config = config or SelectiveConfig()
        self._lru = InferenceEngine(
            model, cache_capacity=self._config.cache_capacity, lock_scope="short"
        )
        self._lock = threading.Lock()
        self._routes = {"reference": 0, "lru": 0}

    def predict_with_decision(
        self, texts: Sequence[str],
    ) -> tuple[list[int], CallMetrics, str]:
        """Devuelve etiquetas, métricas agregadas y ruta efectivamente ejecutada."""
        clean = validate_texts(texts, maximum=256)
        path = choose_inference_path(clean, self._config)
        if path == "lru":
            labels, metrics = self._lru.predict_with_metrics(clean)
        else:
            labels = validated_labels(self._model.predict(list(clean)), len(clean))
            metrics = CallMetrics(len(clean), 0, 0, len(clean), 1)
        with self._lock:
            self._routes[path] += 1
        return labels, metrics, path

    def predict_with_metrics(
        self, texts: Sequence[str],
    ) -> tuple[list[int], CallMetrics]:
        """Mantiene el contrato público de los motores de sprints previos."""
        labels, metrics, _ = self.predict_with_decision(texts)
        return labels, metrics

    def predict(self, texts: Sequence[str]) -> list[int]:
        labels, _, _ = self.predict_with_decision(texts)
        return labels

    def snapshot(self) -> dict[str, Any]:
        """Muestra solo contadores; no registra mensajes ni hashes de tweets."""
        with self._lock:
            route_counts = dict(self._routes)
        result = self._lru.snapshot()
        return {
            **result,
            "policy": "adaptive",
            "routes_reference": route_counts["reference"],
            "routes_lru": route_counts["lru"],
            "min_duplicates": self._config.min_duplicates,
            "min_duplicate_fraction": self._config.min_duplicate_fraction,
            "scope": "decisión por lote; no mide mejora HTTP",
        }
