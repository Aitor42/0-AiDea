"""Métricas agregadas, por clase y por idioma para NLI."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from nli_project.data.dataset import LABEL_IDS, LABEL_NAMES, _language_series

# Decimales con los que se guardan los tiempos (0.1 s de resolución). La
# cuantización evita que informes de ejecuciones idénticas difieran por ruido
# del reloj; la medición subyacente siempre usa ``time.perf_counter``.
SECONDS_DECIMALS = 1

# Columnas de la tabla de métricas por lengua.
PER_LANGUAGE_COLUMNS = ("language", "n", "accuracy", "macro_f1", "n_classes")


def _quantize_seconds(seconds: float) -> float:
    """Redondea un tiempo medido a la resolución fijada en ``SECONDS_DECIMALS``."""
    return round(float(seconds), SECONDS_DECIMALS)



def _accuracy_and_macro_f1(y_true, y_pred) -> tuple[float, float]:
    """Accuracy y macro-F1 sobre las tres clases oficiales."""
    return (
        float(accuracy_score(y_true, y_pred)),
        float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_IDS,
                average="macro",
                zero_division=0,
            )
        ),
    )



def _zero_classification_report() -> dict:
    """Informe de clasificación nulo, para validaciones sin filas.

    scikit-learn rechaza métricas sobre entradas vacías, así que se define
    explícitamente el resultado: precision, recall y F1 nulos y soporte 0 en
    las tres clases, manteniendo el mismo esquema que ``output_dict=True``.
    """
    zero = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
    report = {LABEL_NAMES[label]: dict(zero) for label in LABEL_IDS}
    report["accuracy"] = 0.0
    report["macro avg"] = dict(zero)
    report["weighted avg"] = dict(zero)
    return report



def _empty_per_language_table() -> pd.DataFrame:
    """Tabla de métricas por lengua vacía pero con sus columnas completas."""
    return pd.DataFrame(columns=list(PER_LANGUAGE_COLUMNS))



def per_language_metrics(
    y_true,
    y_pred,
    languages,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Métricas de clasificación agregadas por lengua.

    Args:
        y_true: Etiquetas verdaderas (0, 1, 2).
        y_pred: Etiquetas predichas.
        languages: Lengua de cada ejemplo.
        min_samples: Tamaño mínimo del grupo para incluirlo en la tabla.

    Returns:
        DataFrame con columnas ``language``, ``n``, ``accuracy``,
        ``macro_f1`` (macro-F1 sobre las tres clases con ``zero_division=0``)
        y ``n_classes`` (clases presentes entre real y predicho). Los grupos
        con menos de ``min_samples`` ejemplos se excluyen.
    """
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            "y_pred": np.asarray(y_pred).astype(int),
            "language": np.asarray(languages, dtype=object),
        }
    )

    rows = []
    for language, group in frame.groupby("language", sort=True):
        if len(group) < min_samples:
            continue
        y_true_group = group["y_true"].to_numpy()
        y_pred_group = group["y_pred"].to_numpy()
        accuracy, macro_f1 = _accuracy_and_macro_f1(y_true_group, y_pred_group)
        rows.append(
            {
                "language": language,
                "n": int(len(group)),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "n_classes": len(set(y_true_group) | set(y_pred_group)),
            }
        )

    return pd.DataFrame(rows, columns=list(PER_LANGUAGE_COLUMNS))



def evaluate_holdout(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """Ajusta el pipeline con ``train_df`` y lo evalúa sobre ``val_df``.

    El pipeline recibido se ajusta en el propio objeto (útil para inspeccionar
    después vocabulario, IDF o features) y solo con las filas de entrenamiento:
    la validación jamás participa en el ajuste.

    Args:
        pipeline: Pipeline sin ajustar de ``build_pipeline``.
        train_df: Partición de entrenamiento con columna ``label``.
        val_df: Partición de validación con columna ``label`` (puede estar
            vacía; en ese caso se devuelven métricas nulas).

    Returns:
        Diccionario con ``accuracy``, ``macro_f1``, ``classification_report``
        (``output_dict`` con las clases ``entailment``, ``neutral`` y
        ``contradiction``), ``confusion_matrix`` 3x3, ``per_language``
        (DataFrame de ``per_language_metrics`` con ``min_samples=30``),
        ``runtime_seconds``, ``n_train`` y ``n_val``.
    """
    started = time.perf_counter()
    pipeline.fit(train_df, train_df["label"])

    n_val = int(len(val_df))
    if n_val > 0:
        y_true = val_df["label"].to_numpy()
        predictions = pipeline.predict(val_df)
        accuracy, macro_f1 = _accuracy_and_macro_f1(y_true, predictions)
        report = classification_report(
            y_true,
            predictions,
            labels=LABEL_IDS,
            target_names=[LABEL_NAMES[label] for label in LABEL_IDS],
            output_dict=True,
            zero_division=0,
        )
        confusion = confusion_matrix(y_true, predictions, labels=LABEL_IDS)
        language_series = _language_series(val_df)
        if language_series is None:
            per_language = _empty_per_language_table()
        else:
            per_language = per_language_metrics(
                y_true, predictions, language_series.to_numpy()
            )
    else:
        accuracy = 0.0
        macro_f1 = 0.0
        report = _zero_classification_report()
        confusion = np.zeros((len(LABEL_IDS), len(LABEL_IDS)), dtype=int)
        per_language = _empty_per_language_table()

    runtime_seconds = _quantize_seconds(time.perf_counter() - started)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": confusion,
        "per_language": per_language,
        "runtime_seconds": runtime_seconds,
        "n_train": int(len(train_df)),
        "n_val": n_val,
    }
