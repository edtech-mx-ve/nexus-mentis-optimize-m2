"""Motor de inferencia sin reentrenamiento: deduplicación y caché LRU acotada.

Entrada: textos YA normalizados con la función del Módulo 1.
Salida: exactamente una etiqueta original (0/1) por texto, conservando el orden.
El caché nunca persiste textos: mantiene huellas SHA-256 en RAM.
"""
from __future__ import annotations

import hashlib
import threading
from typing import Literal
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Sequence


def text_fingerprint(text: str) -> bytes:
    """Devuelve una huella SHA-256 determinista del texto normalizado."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Los textos deben ser cadenas no vacías")
    return hashlib.sha256(text.encode("utf-8")).digest()


def validate_texts(texts: Sequence[str], maximum: int = 10_000) -> tuple[str, ...]:
    """Comprueba tipo y cantidad sin modificar ni limpiar el contenido."""
    if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
        raise TypeError("Se requiere una secuencia de textos")
    if not 1 <= len(texts) <= maximum:
        raise ValueError(f"Se requieren entre 1 y {maximum} textos")
    result = tuple(texts)
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError("Hay textos vacíos o con tipos inválidos")
    return result


def validated_labels(values: Any, expected: int) -> list[int]:
    """Convierte predicciones en etiquetas 0/1 y exige longitud esperada."""
    try:
        if len(values) != expected:
            raise ValueError("El modelo cambió el número de predicciones")
        labels = [int(value) for value in values]
    except (TypeError, OverflowError) as exc:
        raise ValueError("El modelo devolvió predicciones inválidas") from exc
    if any(value not in (0, 1) for value in labels):
        raise ValueError("El modelo devolvió etiquetas no reconocidas")
    return labels


@dataclass(frozen=True)
class CallMetrics:
    """Estadísticas por llamada; no contienen textos ni huellas."""
    items: int
    cache_hits: int
    duplicate_reuses: int
    model_items: int
    model_calls: int

    def to_dict(self) -> dict[str, int]:
        """Serializa únicamente contadores no sensibles."""
        return {
            "items": self.items, "cache_hits": self.cache_hits,
            "duplicate_reuses": self.duplicate_reuses,
            "model_items": self.model_items, "model_calls": self.model_calls,
        }


class InferenceEngine:
    """Servicio con deduplicación opcional y caché LRU por proceso.

    El candado cubre la inferencia para garantizar resultados y métricas
    coherentes ante solicitudes concurrentes. Esta decisión puede limitar el
    paralelismo; se evaluará su costo antes de desplegar.
    """

    def __init__(
        self, model: Any, *, cache_capacity: int = 0, deduplicate: bool = True,
        lock_scope: Literal["full", "short"] = "full",
    ) -> None:
        """Inicializa una instancia independiente sin alterar el modelo."""
        if not callable(getattr(model, "predict", None)):
            raise TypeError("El modelo debe implementar predict")
        if not isinstance(cache_capacity, int) or not 0 <= cache_capacity <= 100_000:
            raise ValueError("cache_capacity debe estar entre 0 y 100.000")
        if lock_scope not in ("full", "short"):
            raise ValueError("lock_scope debe ser full o short")
        self._model = model
        self._lock_scope = lock_scope
        self._capacity = cache_capacity
        self._deduplicate = bool(deduplicate)
        self._entries: OrderedDict[bytes, int] = OrderedDict()
        self._lock = threading.RLock()
        self._total_calls = 0
        self._total_hits = 0
        self._total_model_items = 0

    def reset(self) -> None:
        """Vacía el caché y reinicia contadores; útil entre repeticiones."""
        with self._lock:
            self._entries.clear()
            self._total_calls = 0
            self._total_hits = 0
            self._total_model_items = 0

    def snapshot(self) -> dict[str, int | bool]:
        """Devuelve contadores agregados sin mostrar datos privados."""
        with self._lock:
            return {
                "capacity": self._capacity, "cache_entries": len(self._entries),
                "deduplicate": self._deduplicate, "lock_scope": self._lock_scope,
                "calls": self._total_calls,
                "cache_hits": self._total_hits,
                "items_sent_to_model": self._total_model_items,
            }

    def predict(self, texts: Sequence[str]) -> list[int]:
        """Predice una lista de textos normalizados en su orden original."""
        result, _ = self.predict_with_metrics(texts)
        return result

    def predict_with_metrics(
        self, texts: Sequence[str],
    ) -> tuple[list[int], CallMetrics]:
        """Ejecuta inferencia sin cambiar etiquetas ni orden de los textos.

        full: contrato histórico, lock protege también model.predict.
        short: lock solo protege la LRU/contadores; permite que aciertos de
        caché progresen mientras otro hilo calcula misses. En caso de misses
        concurrentes del mismo texto puede haber cómputo redundante.
        """
        clean = validate_texts(texts)
        if self._lock_scope == "short":
            return self._predict_short_lock(clean)
        return self._predict_full_lock(clean)

    def _predict_full_lock(
        self, clean: tuple[str, ...],
    ) -> tuple[list[int], CallMetrics]:
        """Implementación original preservada para otros experimentos."""
        with self._lock:
            self._total_calls += 1
            if not self._deduplicate and self._capacity == 0:
                labels = validated_labels(self._model.predict(list(clean)), len(clean))
                self._total_model_items += len(clean)
                return labels, CallMetrics(len(clean), 0, 0, len(clean), 1)

            keys = tuple(text_fingerprint(text) for text in clean)
            seen: dict[bytes, int] = {}
            pending: dict[bytes, str] = {}
            hits = 0
            in_batch = 0
            for key, text in zip(keys, clean):
                if key in self._entries:
                    seen[key] = self._entries[key]
                    self._entries.move_to_end(key)
                    hits += 1
                elif key in seen or key in pending:
                    in_batch += 1
                else:
                    pending[key] = text

            if pending:
                predicted = validated_labels(
                    self._model.predict(list(pending.values())), len(pending)
                )
                for key, label in zip(pending, predicted):
                    seen[key] = label
                    if self._capacity:
                        self._entries[key] = label
                        self._entries.move_to_end(key)
                        while len(self._entries) > self._capacity:
                            self._entries.popitem(last=False)

            labels = [seen[key] for key in keys]
            self._total_hits += hits
            self._total_model_items += len(pending)
            return labels, CallMetrics(
                len(clean), hits, in_batch, len(pending), int(bool(pending))
            )

    def _predict_short_lock(
        self, clean: tuple[str, ...],
    ) -> tuple[list[int], CallMetrics]:
        """No mantiene un candado global durante la inferencia.

        Se toman instantáneas de aciertos y misses bajo lock. Los misses
        se calculan fuera del lock; al terminar se fusionan las etiquetas.
        Las claves SHA-256 son las únicas que se retienen en el LRU.
        """
        keys = tuple(text_fingerprint(text) for text in clean)
        seen: dict[bytes, int] = {}
        pending: dict[bytes, str] = {}
        hits = 0
        in_batch = 0
        with self._lock:
            self._total_calls += 1
            for key, text in zip(keys, clean):
                if key in self._entries:
                    seen[key] = self._entries[key]
                    self._entries.move_to_end(key)
                    hits += 1
                elif key in seen or key in pending:
                    in_batch += 1
                else:
                    pending[key] = text

        predicted: list[int] = []
        if pending:
            predicted = validated_labels(
                self._model.predict(list(pending.values())), len(pending)
            )
            for key, label in zip(pending, predicted):
                seen[key] = label

        labels = [seen[key] for key in keys]
        with self._lock:
            if self._capacity:
                for key, label in zip(pending, predicted):
                    # Un hilo concurrente pudo insertar la misma clave.
                    # El modelo de clasificación original es inmutable;
                    # conservamos la primera etiqueta ya presente.
                    if key not in self._entries:
                        self._entries[key] = label
                    self._entries.move_to_end(key)
                    while len(self._entries) > self._capacity:
                        self._entries.popitem(last=False)
            self._total_hits += hits
            self._total_model_items += len(pending)
        return labels, CallMetrics(
            len(clean), hits, in_batch, len(pending), int(bool(pending))
        )
