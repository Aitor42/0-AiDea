"""Summarize dense embedding out-of-fold predictions and model complementarity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


CLASSIFIERS = (
    "decision_tree",
    "knn",
    "mlp",
    "xgboost",
    "catboost",
    "lightgbm",
    "random_forest",
)
ENSEMBLE_MEMBERS = ("random_forest", "xgboost", "lightgbm", "catboost", "mlp")
METADATA_COLUMNS = ("id", "label", "lang_abv", "fold")


def _predictions(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return probabilities.argmax(axis=1).astype(labels.dtype, copy=False)


def _fold_metrics(labels: np.ndarray, predictions: np.ndarray, folds: np.ndarray):
    accuracy, macro_f1 = [], []
    for fold in pd.unique(folds):
        mask = folds == fold
        accuracy.append(accuracy_score(labels[mask], predictions[mask]))
        macro_f1.append(
            f1_score(labels[mask], predictions[mask], average="macro", zero_division=0)
        )
    return (
        float(np.mean(accuracy)),
        float(np.std(accuracy, ddof=0)),
        float(np.mean(macro_f1)),
    )


def _summary_row(
    name: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    english: np.ndarray,
    baseline_row: pd.Series | None,
) -> tuple[dict, np.ndarray]:
    predictions = _predictions(probabilities, labels)
    fold_accuracy, fold_accuracy_std, fold_f1 = _fold_metrics(
        labels, predictions, folds
    )
    if baseline_row is None:
        baseline_accuracy = baseline_f1 = np.nan
    else:
        baseline_accuracy = float(baseline_row["accuracy_mean"])
        baseline_f1 = float(baseline_row["macro_f1_mean"])
    summary = {
        "classifier": name,
        "accuracy_fold_mean": fold_accuracy,
        "accuracy_fold_std": fold_accuracy_std,
        "macro_f1_fold_mean": fold_f1,
        "accuracy_oof": accuracy_score(labels, predictions),
        "macro_f1_oof": f1_score(labels, predictions, average="macro", zero_division=0),
        "accuracy_english": accuracy_score(labels[english], predictions[english]) if english.any() else np.nan,
        "accuracy_non_english": accuracy_score(labels[~english], predictions[~english]) if (~english).any() else np.nan,
        "baseline_accuracy_mean": baseline_accuracy,
        "delta_accuracy_vs_baseline": fold_accuracy - baseline_accuracy,
        "baseline_macro_f1_mean": baseline_f1,
        "delta_macro_f1_vs_baseline": fold_f1 - baseline_f1,
    }
    return summary, predictions


def summarize(oof: pd.DataFrame, baseline: pd.DataFrame, output_dir: Path) -> None:
    """Write OOF metrics and pairwise disagreement/complementarity CSVs."""
    if not isinstance(oof, pd.DataFrame) or not isinstance(baseline, pd.DataFrame):
        raise TypeError("oof y baseline deben ser pandas DataFrames.")
    required = set(METADATA_COLUMNS)
    probability_columns = {
        f"{classifier}_p{class_id}"
        for classifier in CLASSIFIERS
        for class_id in range(3)
    }
    missing = (required | probability_columns) - set(oof.columns)
    if missing:
        raise ValueError(f"Faltan columnas OOF: {sorted(missing)}")
    if oof.empty or oof["id"].isna().any() or oof["id"].duplicated().any():
        raise ValueError("OOF debe contener filas y un id único y no nulo por ejemplo.")
    if oof[list(METADATA_COLUMNS)].isna().any().any():
        raise ValueError("Las columnas id, label, lang_abv y fold no pueden ser nulas.")
    if not set(oof["label"].unique()).issubset({0, 1, 2}):
        raise ValueError("Las etiquetas OOF deben ser 0, 1 o 2.")
    if not set(CLASSIFIERS).issubset(set(baseline.get("classifier", []))):
        missing_baseline = set(CLASSIFIERS) - set(baseline.get("classifier", []))
        raise ValueError(f"Faltan resultados de baseline para: {sorted(missing_baseline)}")

    labels = oof["label"].to_numpy()
    folds = oof["fold"].to_numpy()
    if len(np.unique(folds)) < 2:
        raise ValueError("Se necesitan al menos dos folds para resumir métricas por fold.")
    english = oof["lang_abv"].astype(str).str.lower().eq("en").to_numpy()
    summary_rows = []
    predictions_by_model: dict[str, np.ndarray] = {}

    for classifier in CLASSIFIERS:
        columns = [f"{classifier}_p{class_id}" for class_id in range(3)]
        probabilities = oof[columns].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise ValueError(f"Probabilidades no válidas para {classifier}.")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError(f"Las probabilidades de {classifier} no suman 1 (tol=1e-5).")
        baseline_row = baseline.loc[baseline["classifier"] == classifier].iloc[0]
        row, predictions = _summary_row(
            classifier, probabilities, labels, folds, english, baseline_row
        )
        predictions_by_model[classifier] = predictions
        summary_rows.append(row)

    ensemble_probabilities = np.mean(
        [oof[[f"{member}_p{i}" for i in range(3)]].to_numpy(dtype=float) for member in ENSEMBLE_MEMBERS],
        axis=0,
    )
    if not np.allclose(
        ensemble_probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError("Las probabilidades del voto suave no suman 1 (tol=1e-5).")
    ensemble_name = "soft_vote_rf_xgb_lgbm_catboost_mlp"
    ensemble_row, ensemble_predictions = _summary_row(
        ensemble_name, ensemble_probabilities, labels, folds, english, None
    )
    summary_rows.append(ensemble_row)
    predictions_by_model[ensemble_name] = ensemble_predictions

    pairwise_rows = []
    model_names = (*CLASSIFIERS, ensemble_name)
    for index, left in enumerate(model_names):
        left_predictions = predictions_by_model[left]
        left_correct = left_predictions == labels
        for right in model_names[index + 1 :]:
            right_predictions = predictions_by_model[right]
            right_correct = right_predictions == labels
            pairwise_rows.append(
                {
                    "classifier_a": left,
                    "classifier_b": right,
                    "disagreement_rate": float(np.mean(left_predictions != right_predictions)),
                    "a_correct_b_wrong": float(np.mean(left_correct & ~right_correct)),
                    "b_correct_a_wrong": float(np.mean(right_correct & ~left_correct)),
                    "both_correct": float(np.mean(left_correct & right_correct)),
                    "both_wrong": float(np.mean(~left_correct & ~right_correct)),
                    "error_overlap_jaccard": float(np.mean(~left_correct & ~right_correct) / np.mean(~left_correct | ~right_correct)) if np.any(~left_correct | ~right_correct) else 1.0,
                }
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["accuracy_fold_mean", "macro_f1_fold_mean"], ascending=False
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(output_dir / "pairwise.csv", index=False)
