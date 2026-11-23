"""Carga de datos de Kaggle, particiones y auditoría de grupos."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# Etiquetas oficiales de la competición, por identificador entero.
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_IDS = tuple(LABEL_NAMES)  # (0, 1, 2)
VALID_PREDICTION_LABELS = frozenset(LABEL_IDS)

# Columnas del CSV original de entrenamiento, en su orden canónico.
TRAIN_COLUMNS = ("id", "premise", "hypothesis", "lang_abv", "language", "label")


def load_train_data(path: str | Path = "experiments/data/train.csv") -> pd.DataFrame:
    """Carga el CSV de entrenamiento de la competición.

    Args:
        path: Ruta del CSV (por defecto ``experiments/data/train.csv``).

    Returns:
        DataFrame con las columnas ``id``, ``premise``, ``hypothesis``,
        ``lang_abv``, ``language`` y ``label`` (entero), en ese orden.

    Raises:
        FileNotFoundError: Si el fichero no existe.
        ValueError: Si falta alguna columna obligatoria o ``label`` tiene nulos.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"No existe el fichero de entrenamiento: {data_path.resolve()}"
        )

    df = pd.read_csv(data_path)

    missing = [column for column in TRAIN_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en {data_path}: {missing}")

    df = df.loc[:, list(TRAIN_COLUMNS)].copy()

    if df["label"].isna().any():
        raise ValueError("La columna 'label' contiene valores nulos.")
    df["label"] = df["label"].astype("int64")
    return df



def make_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
    group_by_premise: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en entrenamiento y validación.

    Args:
        df: DataFrame de entrada (no se muta).
        val_size: Proporción de validación (entre 0 y 1).
        seed: Semilla que controla la aleatoriedad de la partición.
        stratify: Si es True, la partición simple estratifica por ``label``.
            Se ignora en modo agrupado, que ya es estratificado por naturaleza.
        group_by_premise: Si es True, ninguna premisa aparece en las dos
            particiones: se usa ``StratifiedGroupKFold`` con ``1 / val_size``
            bloques (5 por defecto) y el primer bloque queda como validación.

    Returns:
        Tupla ``(train_df, val_df)`` con índices reiniciados.

    Raises:
        ValueError: Si ``val_size`` no está en (0, 1).
    """
    if not 0.0 < float(val_size) < 1.0:
        raise ValueError(f"val_size debe estar en (0, 1), pero es {val_size}.")

    if group_by_premise:
        n_splits = max(2, min(10, int(round(1.0 / float(val_size)))))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        train_idx, val_idx = next(
            iter(splitter.split(df, df["label"], groups=df["premise"]))
        )
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=seed,
            stratify=df["label"] if stratify else None,
        )

    return (
        train_df.copy().reset_index(drop=True),
        val_df.copy().reset_index(drop=True),
    )



def split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Resume tamaños, distribuciones y solape de premisas de una partición.

    Args:
        train_df: Partición de entrenamiento.
        val_df: Partición de validación.

    Returns:
        Diccionario con ``n_train``, ``n_val``, ``class_distribution_train``,
        ``class_distribution_val`` (diccionarios con las tres etiquetas
        enteras, contando 0 si no aparece), ``language_distribution_train``,
        ``language_distribution_val`` (lenguas presentes y sus recuentos),
        ``premise_overlap`` (número de premisas compartidas) y
        ``premise_overlap_ratio`` (proporción de solape respecto a la
        partición con menos premisas únicas).
    """

    def class_distribution(series: pd.Series) -> dict:
        counts = series.value_counts()
        return {label: int(counts.get(label, 0)) for label in LABEL_IDS}

    def language_distribution(series: pd.Series | None) -> dict:
        if series is None:
            return {}
        return {
            str(key): int(value) for key, value in series.value_counts().items()
        }

    premises_train = set(train_df["premise"])
    premises_val = set(val_df["premise"])
    overlap = len(premises_train & premises_val)
    smallest = min(len(premises_train), len(premises_val))
    ratio = overlap / smallest if smallest else 0.0

    return {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "class_distribution_train": class_distribution(train_df["label"]),
        "class_distribution_val": class_distribution(val_df["label"]),
        "language_distribution_train": language_distribution(
            _language_series(train_df)
        ),
        "language_distribution_val": language_distribution(
            _language_series(val_df)
        ),
        "premise_overlap": int(overlap),
        "premise_overlap_ratio": float(ratio),
    }



def _language_series(frame: pd.DataFrame) -> pd.Series | None:
    """Devuelve la columna de lengua disponible ('language' o 'lang_abv')."""
    if "language" in frame.columns:
        return frame["language"]
    if "lang_abv" in frame.columns:
        return frame["lang_abv"]
    return None
