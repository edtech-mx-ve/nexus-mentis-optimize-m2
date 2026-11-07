"""Guardarraíl para la demostración pública: límites globales en un solo proceso.

No registra IP, publicaciones ni identificadores. No sustituye un WAF
o un control de cuota distribuido, y debe usarse solo para demostraciones.
"""
from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Callable


class PublicRequestGuard:
    """Limita solicitudes de inferencia por ventana temporal deslizante."""

    def __init__(
        self, max_requests: int = 60, *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Construye el control con límites comprobados y reloj inyectable."""
        if not isinstance(max_requests, int) or not 1 <= max_requests <= 600:
            raise ValueError("max_requests debe estar entre 1 y 600")
        if not callable(clock):
            raise TypeError("clock debe ser invocable")
        self._max_requests = max_requests
        self._clock = clock
        self._recent: deque[float] = deque()
        self._lock = Lock()

    def allow(self) -> bool:
        """Reserva una solicitud si hay cuota global en los últimos 60 s."""
        with self._lock:
            now = self._clock()
            while self._recent and now - self._recent[0] >= 60:
                self._recent.popleft()
            if len(self._recent) >= self._max_requests:
                return False
            self._recent.append(now)
            return True


def public_config_from_env(getenv: Callable[[str, str], str]) -> tuple[bool, int, int]:
    """Lee configuración pública explícita sin depender de variables globales."""
    if getenv("NEXUS_PUBLIC_DEMO", "0") != "1":
        return False, 0, 0
    limit = int(getenv("NEXUS_PUBLIC_REQUESTS_PER_MINUTE", "60"))
    batch = int(getenv("NEXUS_PUBLIC_MAX_BATCH", "16"))
    if not 1 <= limit <= 600:
        raise ValueError("Límite público por minuto fuera de rango")
    if not 1 <= batch <= 256:
        raise ValueError("Máximo lote público fuera de rango")
    return True, limit, batch
