"""Observabilidad acotada: estadísticas agregadas, nunca contenido de mensajes."""
from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import psutil

from .benchmark import percentile


@dataclass
class MetricsCollector:
    """Agrega latencias HTTP sin conservar textos, IP, rutas externas ni secretos.

    Por proceso: no consolida métricas de múltiples workers/instancias.
    """

    capacity: int = 2000
    _latencies: Deque[float] = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _count: int = field(default=0, init=False)
    _errors: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.capacity, int) or not 10 <= self.capacity <= 20000:
            raise ValueError("Capacidad de métricas inválida")
        self._latencies = deque(maxlen=self.capacity)

    def observe(self, duration_ms: float, status_code: int) -> None:
        """Añade una duración no negativa y registra códigos HTTP fallidos."""
        if not 0 <= duration_ms < 3600000:
            raise ValueError("Duración HTTP inválida")
        if not 100 <= status_code <= 599:
            raise ValueError("Estado HTTP inválido")
        with self._lock:
            self._count += 1
            self._errors += int(status_code >= 400)
            self._latencies.append(duration_ms)

    def snapshot(self) -> dict[str, int | float | str | None]:
        """Devuelve contadores y percentiles de la ventana reciente."""
        with self._lock:
            values = list(self._latencies)
            count, errors = self._count, self._errors
        memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1_048_576
        return {
            "scope": "HTTP servidor, desde middleware hasta respuesta, solo /api/predict*",
            "requests_total": count,
            "errors_total": errors,
            "errors_percent": round(100 * errors / count, 2) if count else 0,
            "window_samples": len(values),
            "p50_ms": round(percentile(values, 50), 3) if values else None,
            "p95_ms": round(percentile(values, 95), 3) if values else None,
            "p99_ms": round(percentile(values, 99), 3) if values else None,
            "rss_now_mb": round(memory_mb, 2),
            "memory_limitation": "RSS instantánea, no pico continuo",
        }
