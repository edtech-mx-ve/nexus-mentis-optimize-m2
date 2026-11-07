"""Verificación del artefacto original. Nunca carga el pickle durante la auditoría."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = BASE_DIR / "reference"
MANIFEST_FILE = REFERENCE_DIR / "manifest.json"
MODEL_FILE = REFERENCE_DIR / "model.joblib"
REPORT_FILE = REFERENCE_DIR / "report_m1.json"


def sha256_file(path: Path) -> str:
    """Calcula el SHA-256 de un archivo local existente, en bloques."""
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    """Lee un objeto JSON con tamaño acotado y errores explícitos."""
    if not path.is_file():
        raise FileNotFoundError(f"Falta {path.name}")
    if path.stat().st_size > 2_000_000:
        raise ValueError(f"Archivo JSON excesivamente grande: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"JSON inválido: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Se esperaba un objeto JSON en {path.name}")
    return payload


def audit_reference(reference_dir: Path = REFERENCE_DIR) -> dict[str, Any]:
    """Comprueba integridad, procedencia y metadatos sin deserializar modelos."""
    manifest = read_json_object(reference_dir / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise ValueError("Versión de manifiesto no reconocida")
    model_hash = sha256_file(reference_dir / "model.joblib")
    report_hash = sha256_file(reference_dir / "report_m1.json")
    if model_hash != manifest.get("model_sha256"):
        raise ValueError("La huella del modelo difiere del manifiesto")
    if report_hash != manifest.get("report_sha256"):
        raise ValueError("La huella del reporte difiere del manifiesto")

    report = read_json_object(reference_dir / "report_m1.json")
    expected = manifest.get("expected_scikit_learn")
    recorded = report.get("environment", {}).get("library_sklearn")
    if not isinstance(expected, str) or expected != recorded:
        raise ValueError("La versión de scikit-learn no coincide con el reporte")
    test = report.get("test", {})
    baseline = test.get("local_inference_benchmark", {})
    return {
        "status": "integrity_ok",
        "model_sha256": model_hash,
        "report_sha256": report_hash,
        "expected_sklearn": expected,
        "expected_python_minor": manifest.get("expected_python_minor"),
        "current_python": platform.python_version(),
        "historical": {
            "clean_rows": report["audit"]["clean_rows"],
            "test_rows": report["splits"]["test"],
            "f1_macro": test["f1_macro"],
            "batch_throughput_tweets_per_s": test["batch_throughput_tweets_per_s"],
            "local_p95_ms": baseline["p95_ms"],
            "local_scope": baseline["scope"],
        },
        "notice": (
            "La coincidencia SHA-256 verifica integridad frente al manifiesto "
            "incluido, no garantiza seguridad de archivos pickle/joblib."
        ),
    }


def load_trusted_model(reference_dir: Path = REFERENCE_DIR) -> Any:
    """Carga un modelo SOLO cuando el invocador autorizó una fuente confiable.

    Rechaza diferencias de versión: los archivos joblib pueden ejecutar código.
    """
    audit = audit_reference(reference_dir)
    import sklearn
    import joblib

    if sklearn.__version__ != audit["expected_sklearn"]:
        raise RuntimeError(
            f"Versión incompatible: requerida scikit-learn {audit['expected_sklearn']}, "
            f"instalada {sklearn.__version__}. Cree el entorno indicado en README."
        )
    required_minor = str(audit["expected_python_minor"])
    if not platform.python_version().startswith(required_minor + "."):
        raise RuntimeError(
            f"Python incompatible: se requiere {required_minor}.x; "
            f"actual {platform.python_version()}"
        )
    model = joblib.load(reference_dir / "model.joblib")
    if not callable(getattr(model, "predict", None)):
        raise TypeError("El artefacto no implementa predict()")
    return model
