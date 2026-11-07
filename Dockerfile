# Demostración académica M2: un worker y modelo original verificado.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEXUS_HOST=0.0.0.0 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app
COPY requirements.deploy.txt /app/requirements.deploy.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.deploy.txt && python -m pip check

COPY nexus_optimize /app/nexus_optimize
COPY reference /app/reference
COPY public_reports /app/public_reports

# Fallo explícito si falta evidencia conciliada o el artefacto original fue cambiado.
RUN python -m nexus_optimize.cli audit > /dev/null && \
    python -c "from pathlib import Path; from nexus_optimize.integrity import read_json_object; from nexus_optimize.metrics_closure import public_e14; p=Path('/app/public_reports/e14_metrics_closure_public.json'); assert p.is_file(), 'Falta E14 público'; public_e14(read_json_object(p))" && \
    useradd --system --uid 10001 appuser && \
    chown -R appuser:appuser /app

USER appuser
EXPOSE 10000
CMD ["python", "-m", "nexus_optimize.serve"]
