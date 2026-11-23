"""Configuración y caché del preprocesado fila a fila."""

from __future__ import annotations

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from nli_project.preprocessing.preprocesser import NLITextPreprocesser


# Configuración final del preprocesado: tokenización y lematización con spaCy,
# sin stopwords (se conservan las negaciones) y POS activado para
# ``POSRatioFeatures``. El tokenizador de spaCy separa puntuación y
# contracciones, lo que reduce el ruido del vocabulario.
PREPROCESSING_PARAMS = {
    "columns": ("premise", "hypothesis"),
    "language": None,
    "language_column": "lang_abv",
    "lowercase": True,
    "strip_accents": False,
    "tokenizer": "spacy",
    "stopword_method": "none",
    "keep_negations": True,
    "reduction": "spacy_lemma",
    "use_pos": True,
    "ngram_orders": (),
    "top_n": 20,
}

# El baseline histórico se conserva para reproducir los experimentos anteriores.
# Esta receta evita tokenizadores, reducciones, stopwords y POS ingleses.
MULTILINGUAL_PREPROCESSING_PARAMS = {
    **PREPROCESSING_PARAMS,
    "tokenizer": "whitespace",
    "reduction": "none",
    "use_pos": False,
}

ENGLISH_PREPROCESSING_PARAMS = {**PREPROCESSING_PARAMS, "language": "en"}

# Versión del esquema de caché de preprocesado: al incrementarla se invalidan
# todos los .pkl previos sin borrarlos.
PREPROCESS_CACHE_VERSION = 1


def _canonical_repr(value) -> str:
    """Representación canónica y estable de valores de configuración.

    Ordena claves de diccionarios y elementos de conjuntos para que el mismo
    contenido produzca siempre la misma cadena, independientemente del orden
    de inserción o de la semilla de hash de la sesión.
    """
    if isinstance(value, dict):
        items = sorted((str(key), _canonical_repr(item)) for key, item in value.items())
        return "{" + ",".join(f"{key!r}:{item}" for key, item in items) + "}"
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(sorted(_canonical_repr(item) for item in value)) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canonical_repr(item) for item in value) + "]"
    return repr(value)


def _dataframe_digest(df: pd.DataFrame) -> str:
    """Calcula un hash SHA-256 estable del contenido de un DataFrame.

    Solo depende de los nombres de columna y de los valores (con ``str``),
    no del índice ni de la representación interna.
    """
    hasher = hashlib.sha256()
    hasher.update((",".join(map(str, df.columns))).encode("utf-8"))
    for column in df.columns:
        hasher.update("\x1f".join(map(str, df[column].tolist())).encode("utf-8"))
    return hasher.hexdigest()


def _partition_digest(params: dict, df: pd.DataFrame) -> str:
    """Identificador de caché para (parámetros de preprocesado, DataFrame)."""
    payload = (
        f"v{PREPROCESS_CACHE_VERSION}|{_canonical_repr(params)}|"
        f"{_dataframe_digest(df)}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def preprocess_partition(
    df: pd.DataFrame,
    preprocessor_params: dict | None = None,
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Preprocesa una partición con ``NLITextPreprocesser`` y cachea el resultado.

    El preprocesado es una transformación fila a fila (no aprende
    vocabulario ni estadísticas), por lo que puede aplicarse de forma
    independiente a cada partición sin fugas de datos.

    Args:
        df: DataFrame de entrada (no se muta).
        preprocessor_params: Argumentos del constructor de
            ``NLITextPreprocesser``; ``None`` usa ``PREPROCESSING_PARAMS``.
        cache_dir: Si no es ``None``, directorio donde se guarda un ``.pkl``
            identificado por un hash estable de (versión de caché, parámetros,
            contenido del DataFrame). Llamadas idénticas reutilizan la caché.

    Returns:
        DataFrame procesado, con entre otras las columnas
        ``<columna>_processed`` que consume ``NLITextVectorizer``.
    """
    params = dict(
        PREPROCESSING_PARAMS if preprocessor_params is None else preprocessor_params
    )

    if cache_dir is None:
        return NLITextPreprocesser(**params).preprocess_dataset(df)

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"preprocess_{_partition_digest(params, df)}.pkl"

    if cache_file.exists():
        return pd.read_pickle(cache_file)

    processed = NLITextPreprocesser(**params).preprocess_dataset(df)
    with TemporaryDirectory(prefix=".preprocess-", dir=cache_path) as temporary:
        staged = Path(temporary) / cache_file.name
        processed.to_pickle(staged)
        staged.replace(cache_file)
    return processed
