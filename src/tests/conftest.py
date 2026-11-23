"""Fixtures compartidas por la suite TDD del pipeline de entrenamiento."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
LANGUAGES = ["en", "es", "fr"]


def make_synthetic_pairs(n_per_class: int = 60, seed: int = 0) -> pd.DataFrame:
    """DataFrame sintético ya preprocesado, con señal léxica aprendible.

    Mantiene las columnas originales (``id``, ``premise``, ``hypothesis``,
    ``lang_abv``, ``language``, ``label``) y añade las columnas
    ``<columna>_processed`` que consume ``NLITextVectorizer``.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for label in (0, 1, 2):
        for i in range(n_per_class):
            group = i % 25
            premise = (
                f"shared premise {group} says the movie was released "
                f"with review number {i}"
            )

            if label == 0:
                hypothesis = f"yes the movie was released and reviewed {i}"
            elif label == 1:
                hypothesis = f"the movie exists and maybe has review {i}"
            else:
                hypothesis = f"no the movie was not released nor reviewed {i}"

            language = LANGUAGES[(label + i) % len(LANGUAGES)]
            rows.append(
                {
                    "id": f"id-{label}-{i}",
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "lang_abv": language,
                    "language": language,
                    "label": label,
                    "premise_processed": premise,
                    "hypothesis_processed": hypothesis,
                }
            )

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def synthetic_pairs() -> pd.DataFrame:
    return make_synthetic_pairs()


@pytest.fixture
def synthetic_pairs_small() -> pd.DataFrame:
    return make_synthetic_pairs(n_per_class=24, seed=7)


@pytest.fixture
def preprocessor_params() -> dict:
    """Configuración barata y determinista para tests de integración."""
    return {
        "columns": ("premise", "hypothesis"),
        "language": "en",
        "language_column": "lang_abv",
        "lowercase": True,
        "strip_accents": False,
        "tokenizer": "whitespace",
        "stopword_method": "none",
        "keep_negations": True,
        "reduction": "none",
        "ngram_orders": (),
        "use_pos": False,
    }
