"""Confirmación final: CV repetida (3 semillas) + holdout reservado.

Uso (desde la raíz del proyecto):
    python -m experiments.scripts.rf_confirm base sel300_uni_spacy_rel_mindf5

Escribe `experiments/results/random_forest/rf_confirm.json` junto a este fichero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
from sklearn.base import clone

import experiments.scripts.rf_experiments as rf_experiments
from experiments.scripts.rf_experiments import CACHE, EXPERIMENTS, _build, _prep, _rf, _vec, run
from nli_project.evaluation.metrics import evaluate_holdout
from nli_project.data.dataset import (
    load_train_data,
    make_split,
)
from nli_project.preprocessing.preprocessing import preprocess_partition

SEEDS = (42, 7, 2024)


def repeated_cv(name, spec, seeds=SEEDS, n_splits=5):
    rows = [run(f"{name}@seed{seed}", n_splits=n_splits, seed=seed, **spec) for seed in seeds]
    means = [row["accuracy_mean"] for row in rows]
    return {
        "name": name,
        "repeat_means": means,
        "accuracy_mean": float(np.mean(means)),
        "std_across_seeds": float(np.std(means, ddof=0)),
        "macro_f1_mean": float(np.mean([row["macro_f1_mean"] for row in rows])),
    }


def holdout_once(name, spec, train, val):
    prep_params = _prep(**(spec.get("prep") or {}))
    processed_train = preprocess_partition(train, prep_params, cache_dir=CACHE)
    processed_val = preprocess_partition(val, prep_params, cache_dir=CACHE)
    pipeline = _build(
        _vec(**(spec.get("vec") or {})),
        _rf(**(spec.get("rf") or {})),
        use_pos_features=spec.get("use_pos_features", True),
        extra=spec.get("extra"),
        select_k=spec.get("select_k"),
        select_score=spec.get("select_score", "chi2"),
    )
    result = evaluate_holdout(clone(pipeline), processed_train, processed_val)
    print(f"holdout {name:28s} acc={result['accuracy']:.6f} f1={result['macro_f1']:.6f}", flush=True)
    return {
        "name": name,
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "confusion_matrix": np.asarray(result["confusion_matrix"]).tolist(),
        "per_language": result["per_language"].to_dict(orient="records"),
    }


def main(argv):
    names = argv or ["base", "sel300_uni_spacy_rel_mindf5"]
    df = load_train_data(ROOT / "experiments/data/train.csv")
    train, val = make_split(df, val_size=0.2, seed=42)
    del df
    rf_experiments.train = train

    report = {"repeated_cv": [], "holdout": []}
    for name in names:
        report["repeated_cv"].append(repeated_cv(name, EXPERIMENTS[name]))
    for name in names:
        report["holdout"].append(holdout_once(name, EXPERIMENTS[name], train, val))

    (ROOT / "experiments/.local/random_forest").mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments/.local/random_forest/rf_confirm.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print()
    print(
        pd.DataFrame(
            [
                {
                    "config": row["name"],
                    "cv_3seeds": round(row["accuracy_mean"], 6),
                    "std_seeds": round(row["std_across_seeds"], 5),
                    "per_seed": [round(v, 4) for v in row["repeat_means"]],
                }
                for row in report["repeated_cv"]
            ]
        ).to_string(index=False)
    )
    print()
    print(
        pd.DataFrame(
            [
                {"config": r["name"], "holdout_acc": round(r["accuracy"], 6),
                 "holdout_f1": round(r["macro_f1"], 6)}
                for r in report["holdout"]
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main(sys.argv[1:])
