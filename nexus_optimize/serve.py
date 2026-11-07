"""Servidor único de desarrollo/Render: usa el puerto del entorno."""
from __future__ import annotations

import logging
import os

import uvicorn

LOG = logging.getLogger("nexus_optimize.serve")


def main() -> None:
    """Inicia una instancia con concurrencia acotada y sin recarga automática."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raw_port = os.getenv("PORT", "8000")
    try:
        port = int(raw_port)
        if not 1 <= port <= 65535:
            raise ValueError("Puerto fuera de rango")
    except ValueError as exc:
        raise SystemExit(f"PORT inválido: {raw_port!r}") from exc

    host = os.getenv("NEXUS_HOST", "127.0.0.1")
    if host not in ("127.0.0.1", "0.0.0.0"):
        raise SystemExit("NEXUS_HOST debe ser 127.0.0.1 o 0.0.0.0")
    LOG.info("Iniciando NEXUS OPTIMIZE sobre %s:%s (un worker)", host, port)
    uvicorn.run(
        "nexus_optimize.app:app",
        host=host,
        port=port,
        workers=1,             # Evita duplicar modelo y caché en servicios de poca RAM.
        limit_concurrency=32,  # Capacidad de conexiones en prueba controlada.
        timeout_keep_alive=5,
        access_log=False,      # No almacena rutas/IP; errores agregados vía telemetría.
    )


if __name__ == "__main__":
    main()
