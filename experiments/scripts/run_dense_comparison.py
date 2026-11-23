"""Compara los siete clasificadores sobre las cachés MiniLM del experimento RF.

Ejecutar con python -m experiments.scripts.run_dense_comparison. Solo necesita las dependencias
numéricas y los tres paquetes de boosting. Se comprueba el orden de filas
contra el split del experimento RF y se reconstruyen los folds de la comparación léxica.
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits

from nli_project.features.minilm import full_features
from nli_project.models.classifiers import build_classifier
from nli_project.evaluation.dense_oof_summary import CLASSIFIERS, ENSEMBLE_MEMBERS, summarize


ROOT = Path(__file__).resolve().parents[2]
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BASELINE = ROOT / "experiments/results/classical_models/algorithm_comparison" / "algorithm_comparison.csv"


def load_inputs():
    train = pd.read_csv(ROOT / "experiments/results/validation/historical_split/train.csv")
    source = pd.read_csv(ROOT / "experiments/data/train.csv")
    expected, _ = train_test_split(
        source, test_size=0.2, stratify=source["label"], random_state=42
    )
    pd.testing.assert_frame_equal(train, expected.reset_index(drop=True))

    vectors = []
    for field, suffix in (("premise", "p"), ("hypothesis", "h")):
        path = ROOT / "experiments/.local/embeddings" / f"{ENCODER.replace('/', '__')}_train_{suffix}.npy"
        data = np.load(path, allow_pickle=False)
        if data.shape != (len(train), 384) or not np.isfinite(data).all():
            raise ValueError(f"Caché MiniLM incompatible: {path}")
        # Los .npy históricos carecen de IDs. Los textos repetidos permiten
        # detectar permutaciones o un intercambio de premisa e hipótesis.
        codes, unique = pd.factorize(train[field])
        first = np.full(len(unique), len(train))
        np.minimum.at(first, codes, np.arange(len(train)))
        if not np.allclose(data, data[first[codes]], atol=1e-4):
            raise ValueError(f"Los vectores no coinciden para textos repetidos: {field}")
        vectors.append(data)
    premise, hypothesis = vectors
    full = full_features(premise, hypothesis)
    return train, full


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8, help="Hilos por ajuste en CPU")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/.local/dense/comparison")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads debe ser positivo")

    baseline = pd.read_csv(BASELINE)
    if not baseline["classifier"].is_unique:
        raise ValueError("La tabla léxica debe contener una fila por clasificador")
    specs = baseline.set_index("classifier").loc[list(CLASSIFIERS)]
    seed, = specs["seed"].unique()
    n_splits, = specs["n_splits"].unique()
    train, features = load_inputs()
    labels = train["label"].to_numpy()
    oof = train[["id", "label", "lang_abv"]].copy()
    splitter = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    folds = list(splitter.split(features, labels))
    for number, (fit, val) in enumerate(folds, start=1):
        oof.loc[val, "fold"] = number
        oof.loc[val, "premise_seen_in_fit"] = train.loc[val, "premise"].isin(train.loc[fit, "premise"])
    oof["fold"] = oof["fold"].astype(int)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_results = []
    with threadpool_limits(limits=args.threads):
        for name, spec in specs.iterrows():
            params = json.loads(spec["classifier_params"])
            if name == "catboost":
                params["thread_count"] = args.threads
            elif name in ("random_forest", "xgboost", "lightgbm", "knn"):
                params["n_jobs"] = args.threads
            probas = np.full((len(train), 3), np.nan)
            for number, (fit, val) in enumerate(folds, start=1):
                start = perf_counter()
                model = build_classifier(name, random_state=int(seed), **params)
                model.fit(features[fit], labels[fit])
                if not np.array_equal(model.classes_, [0, 1, 2]):
                    raise ValueError(f"Orden de clases inesperado: {model.classes_}")
                probas[val] = model.predict_proba(features[val])
                pred = probas[val].argmax(axis=1)
                row = {
                    "classifier": name,
                    "fold": number,
                    "n_fit": len(fit),
                    "n_val": len(val),
                    "accuracy": accuracy_score(labels[val], pred),
                    "macro_f1": f1_score(labels[val], pred, average="macro"),
                    "fit_predict_seconds": round(perf_counter() - start, 2),
                }
                fold_results.append(row)
                print(f"{name} fold {number}/{n_splits}: {row['accuracy']:.4f} "
                      f"({row['fit_predict_seconds']} s)", flush=True)
            oof[[f"{name}_p{i}" for i in range(3)]] = probas
            # Conserva el progreso en caso de que falle un clasificador posterior.
            oof.to_csv(args.output_dir / "oof.partial.csv.gz", index=False)
            pd.DataFrame(fold_results).to_csv(args.output_dir / "folds.csv", index=False)

    summarize(oof, baseline, args.output_dir)
    manifest = {
        "encoder": ENCODER, "features": ["p", "h", "abs(p-h)", "p*h"],
        "shape": list(features.shape), "folds": int(n_splits), "seed": int(seed),
        "split": "StratifiedKFold(shuffle=True), por fila",
        "ensemble_uniform": list(ENSEMBLE_MEMBERS), "threads": args.threads,
        "device": "cpu", "python": platform.python_version(),
        "versions": {key: version(key) for key in (
            "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm", "catboost"
        )},
        "params": {name: json.loads(raw) for name, raw in specs["classifier_params"].items()},
    }
    (args.output_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output_dir / "oof.partial.csv.gz").replace(args.output_dir / "oof.csv.gz")


if __name__ == "__main__":
    main()
