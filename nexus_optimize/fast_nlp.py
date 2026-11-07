"""Normalización de tweets optimizada sin cambiar la semántica de NEXUS ANALYZE.

La ruta rápida evita conversiones innecesarias cuando el texto es ASCII sin
entidades HTML. Siempre comprueba patrones antes de ejecutar sustituciones.
Los casos Unicode y HTML se delegan a la misma normalización del M1.
"""
from __future__ import annotations

import html
import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@\w+")
_WS = re.compile(r"\s+")


def normalize_tweet_fast(value: str) -> str:
    """Devuelve exactamente la normalización M1 con atajos seguros para ASCII.

    Input no confiable: valida tipo, no persiste texto ni efectúa llamadas de red.
    """
    if not isinstance(value, str):
        raise TypeError("El texto debe ser una cadena")
    if value.isascii() and "&" not in value:
        text = value.strip().lower()
    else:
        text = unicodedata.normalize("NFKC", html.unescape(value)).strip().lower()
    if "http://" in text or "https://" in text or "www." in text:
        text = _URL.sub(" urltoken ", text)
    if "@" in text:
        text = _MENTION.sub(" usermention ", text)
    return _WS.sub(" ", text).strip()
