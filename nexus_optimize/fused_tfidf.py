"""E9: proyección TF-IDF + regresión logística sin matriz dispersa intermedia.

Sólo activa la ruta para el pipeline M1 de dos clases con TF-IDF:
unigramas/bigramas, idf, tf sublineal, norma L2 y regresión logística binaria.
La representación del conocimiento (vocabulario, IDF y coeficientes) se reutiliza
sin alterar los parámetros ajustados. Se verifica equivalencia de etiquetas.

NO se utiliza automáticamente para servir usuarios: es candidato experimental.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .fast_nlp import normalize_tweet_fast
from .nlp import normalize_tweet
from .optimization import validated_labels


class FusedTfidfLogistic:
    """Puntuación binaria de los vectores TF-IDF sin construir una matriz CSR.

    Encapsula únicamente los parámetros *leídos* del pipeline original.
    Con texto cercano a la frontera, recurre a predict() del M1 para evitar
    diferencias de clasificación debidas al orden de suma en punto flotante.
    """

    def __init__(self, model: Any, *, boundary_margin: float = 1e-5) -> None:
        if not math.isfinite(boundary_margin) or not 0 < boundary_margin <= 0.01:
            raise ValueError("boundary_margin debe estar en (0, 0.01]")
        steps = getattr(model, "named_steps", None)
        if steps is None or tuple(steps) != ("tfidf", "classifier"):
            raise ValueError("Se requiere el Pipeline M1: tfidf -> classifier")
        vec, clf = steps["tfidf"], steps["classifier"]
        if not isinstance(vec, TfidfVectorizer) or not isinstance(clf, LogisticRegression):
            raise TypeError("Se requiere TfidfVectorizer y LogisticRegression originales")
        if (
            vec.input != "content"
            or vec.analyzer != "word"
            or vec.ngram_range != (1, 2)
            or vec.stop_words is not None
            or vec.binary
            or not vec.sublinear_tf
            or vec.norm != "l2"
            or not vec.use_idf
            or np.dtype(vec.dtype) != np.dtype("float64")
            or vec.tokenizer is not None
            or vec.preprocessor is not None
        ):
            raise ValueError("Parámetros TF-IDF distintos del perfil M1 certificado")
        if (
            clf.coef_.ndim != 2 or clf.coef_.shape[0] != 1
            or clf.intercept_.shape != (1,)
            or tuple(int(c) for c in clf.classes_) != (0, 1)
            or len(vec.idf_) != clf.coef_.shape[1]
            or len(vec.vocabulary_) != clf.coef_.shape[1]
        ):
            raise ValueError("El clasificador no corresponde al perfil binario M1")
        weighted = np.asarray(clf.coef_[0], dtype=np.float64) * vec.idf_
        if not (
            np.all(np.isfinite(weighted)) and np.all(np.isfinite(vec.idf_))
            and math.isfinite(float(clf.intercept_[0]))
        ):
            raise ValueError("Parámetros no finitos")
        self._model = model
        self._vocabulary: dict[str, int] = vec.vocabulary_
        self._preprocess: Callable[[str], str] = vec.build_preprocessor()
        self._tokenize: Callable[[str], list[str]] = vec.build_tokenizer()
        self._weighted = weighted.tolist()
        self._idf = vec.idf_.tolist()
        self._intercept = float(clf.intercept_[0])
        self._boundary_margin = boundary_margin

    def _score(self, text: str) -> float:
        """Puntuación binaria: b + Σ(tf*idf*coef)/sqrt(Σ(tf*idf)^2)."""
        terms = self._tokenize(self._preprocess(text))
        lookup = self._vocabulary.get
        counts: dict[int, int] = {}
        previous: str | None = None
        for token in terms:
            index = lookup(token)
            if index is not None:
                counts[index] = counts.get(index, 0) + 1
            if previous is not None:
                index = lookup(previous + " " + token)
                if index is not None:
                    counts[index] = counts.get(index, 0) + 1
            previous = token
        numerator = 0.0
        norm_squared = 0.0
        weights, idfs = self._weighted, self._idf
        for index, frequency in counts.items():
            tf = 1.0 + math.log(frequency) if frequency > 1 else 1.0
            numerator += tf * weights[index]
            weight = tf * idfs[index]
            norm_squared += weight * weight
        return self._intercept + (
            numerator / math.sqrt(norm_squared) if norm_squared else 0.0
        )

    def predict_normalized(self, texts: Sequence[str]) -> list[int]:
        if not texts or any(not isinstance(t, str) or not t for t in texts):
            raise ValueError("El lote normalizado debe contener textos no vacíos")
        results: list[int] = []
        uncertain_indices: list[int] = []
        uncertain_texts: list[str] = []
        for index, value in enumerate(texts):
            score = self._score(value)
            if not math.isfinite(score):
                raise RuntimeError("Puntuación no finita del clasificador")
            if abs(score) <= self._boundary_margin:
                uncertain_indices.append(index)
                uncertain_texts.append(value)
            results.append(1 if score >= 0 else 0)
        if uncertain_indices:
            reliable = validated_labels(
                self._model.predict(uncertain_texts), len(uncertain_texts)
            )
            for index, label in zip(uncertain_indices, reliable):
                results[index] = label
        return results

    def predict_raw(self, texts: Sequence[str], *, fast_normalization: bool) -> list[int]:
        """Tarea completa RAW → normalización → predicción; no usa caché."""
        normalizer = normalize_tweet_fast if fast_normalization else normalize_tweet
        normalized = [normalizer(t) for t in texts]
        return self.predict_normalized(normalized)
