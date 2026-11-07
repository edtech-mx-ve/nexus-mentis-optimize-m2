"""Carga de trabajo: CSV real Sentiment140 o conjunto didáctico sintético."""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .integrity import sha256_file
from .nlp import normalize_tweet

MAX_CSV_BYTES = 350_000_000


@dataclass(frozen=True)
class Workload:
    """Mensajes para medir inferencia; no equivale a un conjunto de evaluación."""
    texts: tuple[str, ...]
    source_type: str
    file_sha256: str
    valid_rows_seen: int
    invalid_rows_seen: int

    def __post_init__(self) -> None:
        if not self.texts:
            raise ValueError("No hay textos válidos para medir")


def _rows(path: Path, source_type: str) -> Iterator[str | None]:
    """Extrae texto respetando el formato y las etiquetas del origen."""
    if source_type == "sentiment140":
        encoding = "latin-1"
    elif source_type == "demo":
        encoding = "utf-8-sig"
    else:
        raise ValueError("Origen no admitido: use 'demo' o 'sentiment140'")
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        if source_type == "demo":
            header = next(reader, [])
            if header != ["text"]:
                raise ValueError("El CSV didáctico debe tener encabezado 'text'")
        for row in reader:
            if source_type == "sentiment140":
                valid = len(row) == 6 and row[0] in {"0", "4"}
                value = row[5].strip() if valid else ""
            else:
                value = row[0].strip() if len(row) == 1 else ""
            cleaned = normalize_tweet(value)
            yield cleaned if 1 <= len(cleaned) <= 2000 else None


def load_workload(path: Path, source_type: str, sample_size: int, seed: int) -> Workload:
    """Muestrea uniformemente por reservorio y registra únicamente metadatos."""
    if not 1 <= sample_size <= 100_000:
        raise ValueError("sample_size debe estar entre 1 y 100.000")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("La semilla debe ser un entero entre 0 y 2^32-1")
    path = path.expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise FileNotFoundError("Indique un archivo CSV local existente")
    if path.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("Archivo CSV mayor al máximo permitido")
    if source_type not in {"demo", "sentiment140"}:
        raise ValueError("Tipo de origen no reconocido")

    rng = random.Random(seed)
    sampled: list[tuple[int, str]] = []
    valid_count, invalid_count = 0, 0
    for value in _rows(path, source_type):
        if value is None:
            invalid_count += 1
            continue
        valid_count += 1
        if len(sampled) < sample_size:
            sampled.append((valid_count, value))
        else:
            index = rng.randrange(valid_count)
            if index < sample_size:
                sampled[index] = (valid_count, value)
    sampled.sort(key=lambda row: row[0])
    if not sampled:
        raise ValueError("El archivo no contiene tweets válidos")
    return Workload(
        texts=tuple(text for _, text in sampled),
        source_type=source_type,
        file_sha256=sha256_file(path),
        valid_rows_seen=valid_count,
        invalid_rows_seen=invalid_count,
    )


def make_equal_workload(source: Workload, total_items: int) -> tuple[str, ...]:
    """Replica cíclicamente el muestreo para comparar lotes del mismo tamaño."""
    if not 1 <= total_items <= 100_000:
        raise ValueError("total_items debe estar entre 1 y 100.000")
    items = source.texts
    return tuple(items[i % len(items)] for i in range(total_items))
