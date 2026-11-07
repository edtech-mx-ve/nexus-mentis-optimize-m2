"""API didáctica M2: comparación local y demostración pública acotada."""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .integrity import BASE_DIR, audit_reference, load_trusted_model, read_json_object
from .nlp import normalize_tweet
from .fast_nlp import normalize_tweet_fast
from .fused_tfidf import FusedTfidfLogistic
from .optimization import InferenceEngine, validated_labels
from .selective import SelectiveInferenceEngine, config_from_env
from .public_guard import PublicRequestGuard, public_config_from_env
from .telemetry import MetricsCollector

LOG = logging.getLogger("nexus_optimize.api")
WEB_DIR = Path(__file__).resolve().parent / "web"


class PredictionRequest(BaseModel):
    """Un mensaje en inglés; español/traducción son alcance de sprints posteriores."""
    text: str = Field(min_length=1, max_length=500)
    language: Literal["en"] = "en"


class PredictionResponse(BaseModel):
    """Etiqueta y tiempo local de procesamiento, sin devolver el texto enviado."""
    sentiment: Literal["negativo", "positivo"]
    processing_ms: float
    scope: str = "normalizacion_y_prediccion_local_sin_transporte_http"
    cache_hit: bool = False


class BatchPredictionRequest(BaseModel):
    """Solicitud acotada a 256 textos, solo inglés y sin traducción."""
    texts: list[str] = Field(min_length=1, max_length=256)
    language: Literal["en"] = "en"
    strategy: Literal["reference", "optimized"] = "optimized"


class BatchPredictionResponse(BaseModel):
    """Predicciones con métricas por solicitud, sin incluir textos del usuario."""
    sentiments: list[Literal["negativo", "positivo"]]
    processing_ms: float
    items: int
    model_items: int
    cache_hits: int
    duplicate_reuses: int
    strategy: str
    scope: str = "normalizacion_y_prediccion_local_sin_transporte_http"
    execution_path: Literal["reference", "dedupe", "lru"] | None = None


class PipelineBatchRequest(BaseModel):
    """Prueba local sin caché sobre el flujo RAW de M1 y E9."""
    texts: list[str] = Field(min_length=1, max_length=256)
    language: Literal["en"] = "en"
    strategy: Literal["m1", "e9"] = "m1"


class PipelineBatchResponse(BaseModel):
    """Solo resultados agregados de procesamiento; no expone los tweets."""
    sentiments: list[Literal["negativo", "positivo"]]
    processing_ms: float
    items: int
    strategy: Literal["m1", "e9"]
    scope: str = "normalizacion_y_inferencia_servidor_sin_transporte_http"


def create_app() -> FastAPI:
    """Construye una aplicación sin cargar joblib salvo autorización explícita."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = None
        app.state.engine = None
        app.state.e9_pipeline = None
        app.state.inference_policy = "disabled"
        if app.state.require_public_reports:
            from .metrics_closure import public_e14
            path = BASE_DIR / "public_reports" / "e14_metrics_closure_public.json"
            if not path.is_file():
                raise RuntimeError(
                    "Falta public_reports/e14_metrics_closure_public.json; "
                    "exporta y valida E14 antes de publicar M2"
                )
            try:
                public_e14(read_json_object(path))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise RuntimeError("El reporte público E14 es inválido") from exc
        if os.environ.get("NEXUS_TRUST_LOCAL_MODEL") == "1":
            LOG.info("Autorizada carga local del modelo para inferencia")
            app.state.model = load_trusted_model()
            policy = os.getenv("NEXUS_INFERENCE_STRATEGY", "lru").strip().lower()
            if policy not in {"reference", "dedupe", "lru", "adaptive"}:
                raise ValueError("NEXUS_INFERENCE_STRATEGY inválida")
            if policy == "adaptive":
                app.state.engine = SelectiveInferenceEngine(
                    app.state.model, config_from_env(os.getenv)
                )
            else:
                capacity = int(os.getenv("NEXUS_CACHE_CAPACITY", "4096"))
                app.state.engine = InferenceEngine(
                    app.state.model,
                    cache_capacity=capacity if policy == "lru" else 0,
                    deduplicate=policy != "reference",
                )
            app.state.inference_policy = policy
            if ((not app.state.public_demo
                 and os.getenv("NEXUS_ENABLE_E9_API", "0") == "1")
                    or app.state.public_compare):
                candidate = FusedTfidfLogistic(app.state.model)
                probes = [
                    "I love this fantastic product!",
                    "This is terrible and disappointing.",
                    "Absolutely amazing concert, thank you!",
                    "I do not like this.",
                    "I feel fine", "Neutral statement.",
                    "Great! Really great!", "bad bad bad",
                    "café muy bien 😀", "http://example.com",
                    "RT @user: I enjoyed this", "not sure what to say",
                ]
                a = validated_labels(
                    app.state.model.predict([normalize_tweet(t) for t in probes]),
                    len(probes),
                )
                b = candidate.predict_raw(probes, fast_normalization=True)
                if a != b:
                    raise RuntimeError("E9 no iguala las etiquetas de M1 en los controles de arranque")
                app.state.e9_pipeline = candidate
                LOG.info("Ruta optimizada habilitada: equivalencia inicial verificada")
        else:
            LOG.warning("Modelo deshabilitado: falta NEXUS_TRUST_LOCAL_MODEL=1")
        yield
        app.state.model = None
        app.state.engine = None
        app.state.e9_pipeline = None
        app.state.inference_policy = "disabled"

    public, per_minute, max_batch = public_config_from_env(os.getenv)
    public_compare = public and os.getenv("NEXUS_PUBLIC_ENABLE_COMPARE", "0") == "1"
    require_public_reports = public and os.getenv("NEXUS_REQUIRE_PUBLIC_REPORTS", "0") == "1"
    app = FastAPI(
        title="NEXUS MENTIS — NEXUS OPTIMIZE (M2)",
        version="0.4.17",
        description=("API de análisis de sentimientos y comparación del tiempo de "
                     "respuesta entre el clasificador original y su ejecución optimizada."),
        openapi_url=None if public else "/openapi.json",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
    )
    app.state.inference_policy = "disabled"
    app.state.e9_pipeline = None
    app.state.telemetry = MetricsCollector(capacity=2000)
    app.state.public_compare = public_compare
    app.state.require_public_reports = require_public_reports
    app.state.public_demo = public
    app.state.public_max_batch = max_batch
    app.state.public_guard = (
        PublicRequestGuard(max_requests=per_minute) if public else None
    )
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")

    @app.middleware("http")
    async def monitor(request: Request, call_next: Any) -> Any:
        """Mide tiempo de servidor, sin leer ni almacenar el cuerpo HTTP."""
        started = time.perf_counter_ns()
        if app.state.public_demo and (
            request.url.path in {"/docs", "/redoc", "/openapi.json"}
            or (request.url.path.startswith("/api/") and request.url.path not in {
                "/api/status", "/api/history", "/api/metrics-closure/results",
                "/api/predict", "/api/predict/batch", "/api/pipeline/batch",
            })
        ):
            response = JSONResponse(status_code=404, content={"detail": "No disponible"})
        elif app.state.public_demo and request.url.path in {
            "/api/predict", "/api/predict/batch", "/api/pipeline/batch"
        }:
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    size = int(content_length)
                except ValueError:
                    size = -1
                if size < 0 or size > 16384:
                    response = JSONResponse(
                        status_code=413, content={"detail": "Solicitud demasiado grande"}
                    )
                elif not app.state.public_guard.allow():
                    response = JSONResponse(
                        status_code=429, content={"detail": "Límite temporal de demostración"}
                    )
                else:
                    response = await call_next(request)
            else:
                response = JSONResponse(
                    status_code=411,
                    content={"detail": "Content-Length obligatorio en la demostración"},
                )
        else:
            response = await call_next(request)
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        if request.url.path in {"/api/predict", "/api/predict/batch", "/api/pipeline/batch"}:
            app.state.telemetry.observe(elapsed_ms, response.status_code)
        # Server-Timing estima middleware y función; NO incluye RTT cliente.
        response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.3f}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; base-uri 'none'; object-src 'none'; "
            "frame-ancestors 'none'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; connect-src 'self'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/docs", include_in_schema=False)
    def local_api_docs() -> HTMLResponse:
        """Swagger UI sin CDN ni scripts inline: compatible con CSP `self`."""
        bundle = WEB_DIR / "swagger-ui" / "swagger-ui-bundle.js"
        stylesheet = WEB_DIR / "swagger-ui" / "swagger-ui.css"
        if not bundle.is_file() or not stylesheet.is_file():
            return HTMLResponse(
                "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
                "<title>Documentación de la API</title></head><body>"
                "<h1>Swagger UI requiere archivos locales</h1>"
                "<p>Ejecuta en PowerShell: "
                "<code>.\\scripts\\install_swagger_assets.ps1</code></p>"
                "<p>La especificación continúa disponible en "
                "<a href='/openapi.json'>/openapi.json</a>.</p>"
                "</body></html>",
                status_code=503,
                headers={"Cache-Control": "no-store"},
            )
        return HTMLResponse(
            "<!doctype html><html lang='es'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>NEXUS OPTIMIZE M2 · API</title>"
            "<link rel='icon' href='/assets/nexus-logo.webp'>"
            "<link rel='stylesheet' href='/assets/swagger-ui/swagger-ui.css'>"
            "</head><body>"
            "<div id='swagger-ui'></div>"
            "<script src='/assets/swagger-ui/swagger-ui-bundle.js' defer></script>"
            "<script src='/assets/swagger-init.js' defer></script>"
            "</body></html>",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/redoc", include_in_schema=False)
    def docs_redirect() -> RedirectResponse:
        """La documentación secundaria remite a Swagger local."""
        return RedirectResponse(url="/docs", status_code=307)

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        """Presenta el laboratorio didáctico responsivo."""
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        """Informa sin revelar rutas locales ni datos textuales."""
        audit = audit_reference()
        if app.state.public_demo:
            return {
                "service": "nexus-optimize-m2",
                "model_ready": app.state.model is not None,
                "e9_api_enabled": app.state.e9_pipeline is not None,
                "public_demo": True,
                "integrity": audit["status"],
                "dataset": "Sentiment140 (histórico)",
            }
        return {
            "service": "nexus-optimize-m2",
            "sprint": 4,
            "model_ready": app.state.model is not None,
            "optimized_ready": app.state.engine is not None,
            "inference_policy": app.state.inference_policy,
            "selective_revision": (
                "s3.7" if isinstance(app.state.engine, SelectiveInferenceEngine)
                else None
            ),
            "protocol_revision": "e7-s3.8",
            "global_http_revision": "e13-s4.5",
            "metric_closure_revision": "e14-s4.6",
            "selective_config": ({
                "min_duplicates": app.state.engine.snapshot()["min_duplicates"],
                "min_duplicate_fraction":
                    app.state.engine.snapshot()["min_duplicate_fraction"],
                "cache_capacity": app.state.engine.snapshot()["capacity"],
            } if isinstance(app.state.engine, SelectiveInferenceEngine) else None),
            "integrity": audit["status"],
            "dataset": "Sentiment140 (histórico)",
            "public_demo": app.state.public_demo,
            "e9_api_enabled": app.state.e9_pipeline is not None,
            "e9_revision": "s4.4",
        }

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        """Readiness para despliegue; 503 cuando predicción no está habilitada."""
        ready = app.state.model is not None
        return JSONResponse(
            status_code=200 if ready else 503,
            content={"status": "ready" if ready else "model_disabled"},
        )

    @app.get("/api/history")
    def history() -> dict[str, Any]:
        """Métricas históricas del Módulo 1, distintas del benchmark nuevo."""
        audit = audit_reference()
        if app.state.public_demo:
            history = audit["historical"]
            return {
                "status": "historical_reference",
                "f1_macro": history["f1_macro"],
                "test_rows": history["test_rows"],
            }
        return {"status": "historical_reference", **audit["historical"]}

    @app.get("/api/benchmark")
    def benchmark() -> dict[str, Any]:
        """Entrega un archivo de resultados local, nunca ejecuta cargas remotas."""
        selected = Path(os.getenv(
            "NEXUS_BENCHMARK_FILE", str(BASE_DIR / "results" / "baseline.json")
        ))
        if not selected.is_file():
            return {"status": "not_run", "message": "Ejecute el benchmark desde CLI"}
        try:
            record = read_json_object(selected)
        except (OSError, ValueError) as exc:
            LOG.warning("Reporte inválido: %s", type(exc).__name__)
            raise HTTPException(status_code=422, detail="Reporte inválido") from exc
        if record.get("schema_version") != 1:
            raise HTTPException(status_code=422, detail="Esquema no reconocido")
        return record

    @app.get("/api/compare/results")
    def comparison() -> dict[str, Any]:
        """Lee el resultado local de E1; nunca inicia benchmarks remotos."""
        selected = Path(os.getenv(
            "NEXUS_COMPARE_FILE", str(BASE_DIR / "results" / "comparison_e1.json")
        ))
        if not selected.is_file():
            return {"status": "not_run", "message": "Ejecute compare desde la CLI"}
        try:
            record = read_json_object(selected)
        except (OSError, ValueError) as exc:
            LOG.warning("Comparación inválida: %s", type(exc).__name__)
            raise HTTPException(status_code=422, detail="Reporte E1 inválido") from exc
        if record.get("schema_version") != 2:
            raise HTTPException(status_code=422, detail="Esquema E1 no reconocido")
        return record

    @app.get("/api/http/results")
    def http_results() -> dict[str, Any]:
        """Muestra E2 desde disco; nunca ejecuta pruebas de carga por HTTP."""
        selected = Path(os.getenv(
            "NEXUS_HTTP_REPORT_FILE", str(BASE_DIR / "results" / "comparison_e2_http.json")
        ))
        if not selected.is_file():
            return {"status": "not_run", "message": "Ejecute el experimento E2 en PowerShell"}
        try:
            record = read_json_object(selected)
        except (OSError, ValueError) as exc:
            LOG.warning("Reporte E2 inválido: %s", type(exc).__name__)
            raise HTTPException(status_code=422, detail="Reporte E2 inválido") from exc
        if record.get("schema_version") != 3 or record.get("status") != "measured":
            raise HTTPException(status_code=422, detail="Esquema E2 no reconocido")
        return record

    @app.get("/api/evolution/results")
    def evolution_results() -> dict[str, Any]:
        """Consulta E3 de forma pasiva: nunca ejecuta el AG desde la web."""
        selected = Path(os.getenv(
            "NEXUS_E3_REPORT_FILE", str(BASE_DIR / "results" / "e3_genetic.json")
        ))
        if not selected.is_file():
            return {"status": "not_run", "message": "Ejecute E3 desde PowerShell"}
        try:
            report = read_json_object(selected)
        except (OSError, ValueError) as exc:
            LOG.warning("Reporte E3 inválido: %s", type(exc).__name__)
            raise HTTPException(status_code=422, detail="Reporte E3 inválido") from exc
        if report.get("schema_version") != 4 or report.get("status") not in {
            "candidate_validated", "not_improved"
        } or not isinstance(report.get("holdout"), dict):
            raise HTTPException(status_code=422, detail="Esquema E3 no reconocido")
        return report

    @app.get("/api/e4/results")
    def e4_results() -> dict[str, Any]:
        """Lee el resultado E4 sin ejecutar pruebas de carga."""
        from .publish_reports import sanitize_report
        selected = Path(os.getenv(
            "NEXUS_E4_REPORT_FILE",
            str(BASE_DIR / "results" / "comparison_e4_ga_http.json"),
        ))
        if not selected.is_file():
            return {"status": "not_run", "message": "E4 no publicado"}
        try:
            record = read_json_object(selected)
            if record.get("schema_version") != 3 or record.get("status") != "measured":
                raise ValueError("Esquema E4 inválido")
            return sanitize_report("comparison_e4_ga_http.json", record) if app.state.public_demo else record
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Reporte E4 inválido") from exc

    @app.get("/api/selective/results")
    def e5_results() -> dict[str, Any]:
        """Muestra la evaluación E5 sin ejecutar ni alterar rutas de inferencia."""
        from .publish_reports import sanitize_report
        selected = Path(os.getenv(
            "NEXUS_E5_REPORT_FILE",
            str(BASE_DIR / "results" / "e5_selective_http.json"),
        ))
        if not selected.is_file():
            return {"status": "not_run", "message": "Ejecute E5 localmente"}
        try:
            record = read_json_object(selected)
            if record.get("schema_version") != 5 or record.get("status") != "measured":
                raise ValueError("Esquema E5 inválido")
            return sanitize_report("e5_selective_http.json", record) if app.state.public_demo else record
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Reporte E5 inválido") from exc

    @app.get("/api/stability/results")
    def e6_results() -> dict[str, Any]:
        """Expone E6 solo como reporte leído, nunca ejecuta carga HTTP."""
        from .publish_reports import sanitize_report
        selected = Path(os.getenv(
            "NEXUS_E6_REPORT_FILE",
            str(BASE_DIR / "results" / "e6_stability_http.json"),
        ))
        if not selected.is_file():
            return {"status": "not_run", "message": "Ejecute E6 desde PowerShell"}
        try:
            record = read_json_object(selected)
            if record.get("schema_version") != 6 or record.get("status") != "measured":
                raise ValueError("Esquema E6 inválido")
            return (
                sanitize_report("e6_stability_http.json", record)
                if app.state.public_demo else record
            )
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Reporte E6 inválido") from exc

    @app.get("/api/stratified/results")
    def e7_results() -> dict[str, Any]:
        """E7: entrega únicamente un reporte ya medido, sin ejecutar pruebas."""
        from .publish_reports import sanitize_report
        selected = Path(os.getenv(
            "NEXUS_E7_REPORT_FILE",
            str(BASE_DIR / "results" / "e7_stratified_http.json"),
        ))
        if not selected.is_file():
            return {"status": "not_run", "message": "Ejecute E7 desde PowerShell"}
        try:
            record = read_json_object(selected)
            if record.get("schema_version") != 7 or record.get("status") != "measured":
                raise ValueError("Esquema E7 inválido")
            return (
                sanitize_report("e7_stratified_http.json", record)
                if app.state.public_demo else record
            )
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Reporte E7 inválido") from exc

    @app.get("/api/real-http/results")
    def e12_results() -> dict[str, Any]:
        """Lee únicamente métricas permitidas de E12; jamás ejecuta el benchmark."""
        from .real_http import sanitize_e12
        selected = Path(os.getenv(
            "NEXUS_E12_REPORT_FILE",
            str(BASE_DIR / "results" / "e12_real_http.json"),
        ))
        if not selected.is_file():
            return {
                "status": "not_run",
                "message": "Ejecute E12 en Terminal 2 para comparar M1 frente a E9",
            }
        try:
            return sanitize_e12(read_json_object(selected))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOG.warning("Reporte E12 inválido: %s", type(exc).__name__)
            raise HTTPException(status_code=422, detail="Reporte E12 inválido") from exc

    @app.get("/api/global-http/results")
    def e13_results() -> dict[str, Any]:
        """E13: reporte pasivo con cohortes y escenarios no aplicables visibles."""
        from .global_http import sanitize_e13
        selected = Path(os.getenv(
            "NEXUS_E13_REPORT_FILE",
            str(BASE_DIR / "results" / "e13_global_http.json"),
        ))
        if not selected.is_file():
            return {
                "status": "not_run",
                "message": "Ejecute E13 en Terminal 2; conserva E12 como referencia histórica",
            }
        try:
            return sanitize_e13(read_json_object(selected))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            LOG.warning("Reporte E13 inválido: %s", type(exc).__name__)
            raise HTTPException(status_code=422, detail="Reporte E13 inválido") from exc

    @app.get("/api/metrics-closure/results")
    def e14_results() -> dict[str, Any]:
        """E14: conciliación pasiva verificable contra el E13 original."""
        from .metrics_closure import load_and_reconcile, public_e14
        if app.state.public_demo:
            selected_public = BASE_DIR / "public_reports" / "e14_metrics_closure_public.json"
            if not selected_public.is_file():
                return {"status": "not_run", "message": "Comparación no publicada"}
            try:
                return public_e14(read_json_object(selected_public))
            except (OSError, ValueError, TypeError, KeyError) as exc:
                LOG.warning("E14 público inválido: %s", type(exc).__name__)
                raise HTTPException(status_code=422, detail="Reporte público inválido") from exc
        e14_path = Path(os.getenv(
            "NEXUS_E14_REPORT_FILE",
            str(BASE_DIR / "results" / "e14_metrics_closure.json"),
        ))
        if not e14_path.is_file():
            return {
                "status": "not_run",
                "message": "Ejecute E14 en Terminal 2; el E13 original se conserva",
            }
        e13_path = Path(os.getenv(
            "NEXUS_E13_REPORT_FILE",
            str(BASE_DIR / "results" / "e13_global_http.json"),
        ))
        try:
            if not e13_path.is_file():
                raise ValueError("Reporte E13 de origen no disponible")
            source_reanalysis = load_and_reconcile(e13_path)
            saved_closure = read_json_object(e14_path)
            if saved_closure.get("source_e13_sha256") != source_reanalysis["source_e13_sha256"]:
                raise ValueError("Huella de E13 distinta: E14 está desactualizado")
            expected = public_e14(source_reanalysis)
            actual = public_e14(saved_closure)
            if actual != expected:
                raise ValueError("E14 no coincide con el reporte E13 original")
            return actual
        except (OSError, ValueError, TypeError, KeyError) as exc:
            LOG.warning("Conciliación E14 inválida: %s", type(exc).__name__)
            raise HTTPException(
                status_code=422, detail="E14 no coincide con E13; regenere E14 sin borrar E13"
            ) from exc

    @app.post("/api/pipeline/batch", response_model=PipelineBatchResponse)
    def pipeline_batch(payload: PipelineBatchRequest, request: Request) -> PipelineBatchResponse:
        """E12: mismos tweets RAW para M1 y E9; sin caché ni búsqueda genética."""
        if app.state.public_demo:
            if not app.state.public_compare:
                raise HTTPException(status_code=404, detail="Demostración no habilitada")
            if len(payload.texts) > min(8, app.state.public_max_batch):
                raise HTTPException(status_code=422, detail="Máximo 8 textos en demostración")
        elif (os.getenv("NEXUS_ENABLE_E9_API", "0") != "1"
              or request.client is None
              or request.client.host not in {"127.0.0.1", "::1", "testclient"}):
            raise HTTPException(status_code=404, detail="Comparación local no habilitada")
        if app.state.model is None or app.state.e9_pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline E9 no disponible")
        if any(not isinstance(t, str) or not 1 <= len(t) <= 500
               for t in payload.texts):
            raise HTTPException(status_code=422, detail="Tweets: entre 1 y 500 caracteres")
        start = time.perf_counter_ns()
        try:
            if payload.strategy == "m1":
                normalized = [normalize_tweet(value) for value in payload.texts]
                if any(not v for v in normalized):
                    raise ValueError("Texto vacío tras normalización")
                labels = validated_labels(app.state.model.predict(normalized), len(normalized))
            else:
                normalized = [normalize_tweet_fast(value) for value in payload.texts]
                if any(not v for v in normalized):
                    raise ValueError("Texto vacío tras normalización")
                labels = validated_labels(
                    app.state.e9_pipeline.predict_normalized(normalized),
                    len(normalized),
                )
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail="Entrada o inferencia inválida") from exc
        duration_ms = (time.perf_counter_ns() - start) / 1e6
        return PipelineBatchResponse(
            sentiments=["negativo" if n == 0 else "positivo" for n in labels],
            processing_ms=round(duration_ms, 5),
            items=len(labels),
            strategy=payload.strategy,
        )

    @app.get("/api/telemetry")
    def telemetry() -> dict[str, Any]:
        """Permite consultar estadísticas agregadas solo con permiso del entorno."""
        if app.state.public_demo or os.getenv("NEXUS_EXPOSE_METRICS", "0") != "1":
            raise HTTPException(status_code=404, detail="Telemetría no habilitada")
        engine = app.state.engine
        return {
            "status": "ready",
            "http": app.state.telemetry.snapshot(),
            "cache": engine.snapshot() if engine is not None else None,
            "limitation": "contadores de un solo proceso; no sustituye benchmark controlado",
        }

    @app.get("/api/cache/stats")
    def cache_stats() -> dict[str, Any]:
        """Estadísticas agregadas, sin claves de caché ni mensajes."""
        engine = app.state.engine
        return {"status": "ready" if engine is not None else "disabled",
                **(engine.snapshot() if engine is not None else {})}

    @app.post("/api/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(payload: BatchPredictionRequest) -> BatchPredictionResponse:
        """Compara dos rutas de inferencia; máximo 256 tweets por solicitud."""
        model = app.state.model
        if model is None:
            raise HTTPException(status_code=503, detail="Modelo no habilitado")
        if app.state.public_demo and len(payload.texts) > app.state.public_max_batch:
            raise HTTPException(
                status_code=422, detail="El lote supera el máximo de demostración pública"
            )
        if any(len(text) > 500 for text in payload.texts):
            raise HTTPException(status_code=422, detail="Cada texto admite hasta 500 caracteres")
        started = time.perf_counter_ns()
        try:
            normalized = [normalize_tweet(text) for text in payload.texts]
            if any(not value for value in normalized):
                raise ValueError("Hay textos vacíos después de normalización")
            if payload.strategy == "optimized" and app.state.engine is not None:
                engine = app.state.engine
                if isinstance(engine, SelectiveInferenceEngine):
                    labels, metrics, executed = engine.predict_with_decision(normalized)
                else:
                    labels, metrics = engine.predict_with_metrics(normalized)
                    executed = app.state.inference_policy
                    if executed not in {"reference", "dedupe", "lru"}:
                        executed = "lru"
                model_items = metrics.model_items
                cache_hits = metrics.cache_hits
                duplicate_reuses = metrics.duplicate_reuses
            else:
                labels = validated_labels(model.predict(normalized), len(normalized))
                model_items = len(normalized)
                cache_hits = duplicate_reuses = 0
                executed = "reference"
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail="Entrada o predicción inválida") from exc
        elapsed = (time.perf_counter_ns() - started) / 1e6
        return BatchPredictionResponse(
            sentiments=["negativo" if label == 0 else "positivo" for label in labels],
            processing_ms=round(elapsed, 4), items=len(normalized),
            model_items=model_items, cache_hits=cache_hits,
            duplicate_reuses=duplicate_reuses, strategy=payload.strategy,
            execution_path=executed,
        )

    @app.post("/api/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        """Predice sin guardar texto, y mide solo normalización + inferencia."""
        if app.state.model is None:
            raise HTTPException(status_code=503, detail="Modelo no habilitado")
        started = time.perf_counter_ns()
        text = normalize_tweet(payload.text)
        if not text:
            raise HTTPException(status_code=422, detail="Texto vacío tras normalización")
        try:
            cache_hit = False
            if app.state.engine is not None:
                labels, metrics = app.state.engine.predict_with_metrics([text])
                label = labels[0]
                cache_hit = bool(metrics.cache_hits)
            else:
                label = int(app.state.model.predict([text])[0])
        except (IndexError, TypeError, ValueError) as exc:
            LOG.error("Inferencia inválida: %s", type(exc).__name__)
            raise HTTPException(status_code=500, detail="Falló la inferencia") from exc
        if label not in {0, 1}:
            raise HTTPException(status_code=500, detail="Etiqueta inesperada")
        return PredictionResponse(
            sentiment="negativo" if label == 0 else "positivo",
            processing_ms=round((time.perf_counter_ns() - started) / 1e6, 4),
            cache_hit=cache_hit,
        )

    return app


app = create_app()
