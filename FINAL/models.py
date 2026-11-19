"""Entrada de compatibilidad para la fábrica de clasificadores de Aitor."""

from classifiers import (
    BOOSTING_NAMES,
    CLASSIFIER_NAMES,
    NAIVE_BAYES_NAMES,
    build_classifier,
    is_boosting,
    is_naive_bayes,
)

__all__ = [
    "BOOSTING_NAMES",
    "CLASSIFIER_NAMES",
    "NAIVE_BAYES_NAMES",
    "build_classifier",
    "is_boosting",
    "is_naive_bayes",
]
