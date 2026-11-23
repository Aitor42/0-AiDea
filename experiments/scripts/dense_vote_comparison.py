"""Compara votos sobre predicciones OOF guardadas, sin volver a entrenar."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/results/minilm/confirmation"
OUTPUT = ROOT / "experiments/.local/dense/minilm_votes"
MLP = "mlp_256_alpha0.0003"
XGB = "xgb_d3_500"
VOTE4 = ("mlp_256_alpha0.0003", "xgb_d3_500", "cat_d6_500", "lightgbm")
VOTE2 = (MLP, XGB)


def load_seed(seed: int) -> pd.DataFrame:
    frame = pd.read_csv(SOURCE / f"oof_{seed}.csv.gz")
    required = {"id", "label", "lang_abv", "fold"}
    required.update(f"{model}_p{i}" for model in (
        MLP, XGB, "cat_d6_500", "lightgbm", "mlp_original"
    ) for i in range(3))
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF {seed} sin columnas: {sorted(missing)}")
    if frame.empty or frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError(f"IDs OOF inválidos en semilla {seed}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"Hay nulos en las columnas necesarias, semilla {seed}")
    return frame


def metrics(labels, predictions, language):
    en = language == "en"
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
        "accuracy_en": accuracy_score(labels[en], predictions[en]),
        "accuracy_non_en": accuracy_score(labels[~en], predictions[~en]),
    }


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    keys = ["id", "label", "lang_abv"]
    train = pd.read_csv(ROOT / "experiments/results/validation/historical_split/train.csv", usecols=keys)
    summary_rows, paired_rows, error_rows = [], [], []

    for seed in (42, 7, 2024):
        frame = load_seed(seed)
        pd.testing.assert_frame_equal(frame[keys], train[keys])
        labels = frame["label"].to_numpy()
        language = frame["lang_abv"].astype(str).str.lower().to_numpy()
        folds = frame["fold"].to_numpy()
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for number, (_, valid) in enumerate(splitter.split(frame, labels), 1):
            np.testing.assert_array_equal(np.flatnonzero(folds == number), valid)
        probabilities = {}
        for model in ("mlp_original", MLP, XGB, "cat_d6_500", "lightgbm"):
            values = frame[[f"{model}_p{i}" for i in range(3)]].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Probabilidades inválidas en {model}, semilla {seed}")
            if not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=0):
                raise ValueError(f"Probabilidades no normalizadas en {model}, semilla {seed}")
            probabilities[model] = values
        probabilities["voto_2_mlp_xgb"] = np.mean(
            [probabilities[name] for name in VOTE2], axis=0
        )
        probabilities["voto_4"] = np.mean(
            [probabilities[name] for name in VOTE4], axis=0
        )
        predictions = {name: value.argmax(axis=1) for name, value in probabilities.items()}

        for name, pred in predictions.items():
            by_fold = [
                metrics(labels[folds == fold], pred[folds == fold], language[folds == fold])
                for fold in sorted(np.unique(folds))
            ]
            pooled = metrics(labels, pred, language)
            summary_rows.append({
                "seed": seed,
                "model": name,
                "accuracy_fold_mean": np.mean([row["accuracy"] for row in by_fold]),
                "accuracy_fold_std": np.std([row["accuracy"] for row in by_fold]),
                "macro_f1_fold_mean": np.mean([row["macro_f1"] for row in by_fold]),
                **{f"{key}_pooled": value for key, value in pooled.items()},
                "n": len(frame),
                "folds": len(np.unique(folds)),
            })

        for comparator in ("voto_2_mlp_xgb", "voto_4", "mlp_original"):
            candidate_correct = predictions[comparator] == labels
            baseline_correct = predictions[MLP] == labels
            gains = (~baseline_correct) & candidate_correct
            losses = baseline_correct & (~candidate_correct)
            paired_rows.append({
                "seed": seed, "comparator": comparator, "baseline": MLP,
                "candidate_correct_only": int(gains.sum()),
                "baseline_correct_only": int(losses.sum()),
                "both_correct": int((baseline_correct & candidate_correct).sum()),
                "both_wrong": int((~baseline_correct & ~candidate_correct).sum()),
                "net_correct": int(gains.sum() - losses.sum()),
            })
            for idx in np.flatnonzero(gains | losses):
                error_rows.append({
                    "seed": seed, "id": frame.iloc[idx]["id"], "fold": int(folds[idx]),
                    "language": language[idx], "label": int(labels[idx]),
                    "comparator": comparator,
                    "mlp_correct": bool(baseline_correct[idx]),
                    "comparator_correct": bool(candidate_correct[idx]),
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "metrics_by_seed.csv", index=False)
    overall = summary.groupby("model").agg(
        accuracy_mean=("accuracy_fold_mean", "mean"),
        accuracy_between_seed_std=("accuracy_fold_mean", "std"),
        macro_f1_mean=("macro_f1_fold_mean", "mean"),
        accuracy_en_mean=("accuracy_en_pooled", "mean"),
        accuracy_non_en_mean=("accuracy_non_en_pooled", "mean"),
    ).sort_values("accuracy_mean", ascending=False)
    overall.to_csv(OUTPUT / "metrics_summary.csv")
    pd.DataFrame(paired_rows).to_csv(OUTPUT / "paired_vs_mlp_by_seed.csv", index=False)
    pd.DataFrame(error_rows).to_csv(OUTPUT / "changed_errors.csv", index=False)
    metadata = json.loads((SOURCE / "run.json").read_text(encoding="utf-8"))
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "source": SOURCE.name,
        "seeds": [42, 7, 2024],
        "ensemble_2": list(VOTE2),
        "ensemble_4": list(VOTE4),
        "rows_per_seed": len(train),
        "identity_validation": "IDs, labels and languages match the training split; stratified fold membership verified for each seed.",
        "training_performed": False,
        "encoder": metadata["encoder"],
    }, indent=2), encoding="utf-8")
    print(overall.to_string(), flush=True)
    print(pd.DataFrame(paired_rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
