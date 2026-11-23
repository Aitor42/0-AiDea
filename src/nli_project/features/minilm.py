"""Representación de un par a partir de sus dos embeddings MiniLM."""

import numpy as np


def full_features(premise: np.ndarray, hypothesis: np.ndarray) -> np.ndarray:
    return np.hstack(
        (premise, hypothesis, np.abs(premise - hypothesis), premise * hypothesis)
    )
