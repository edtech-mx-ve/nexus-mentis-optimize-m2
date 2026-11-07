"""Normalización de NEXUS ANALYZE M1, preservada para medir el mismo pipeline."""
from __future__ import annotations

import html
import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION = re.compile(r"(?<!\w)@\w+")
_WS = re.compile(r"\s+")


def normalize_tweet(value: str) -> str:
    """Normaliza NFKC, entidades HTML, URLs, menciones y espacios, sin stemming."""
    if not isinstance(value, str):
        raise TypeError("El texto debe ser una cadena")
    text = unicodedata.normalize("NFKC", html.unescape(value)).strip().lower()
    text = _URL.sub(" urltoken ", text)
    text = _MENTION.sub(" usermention ", text)
    text = _WS.sub(" ", text)
    return text.strip()
