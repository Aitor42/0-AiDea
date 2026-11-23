"""CV repetida de configuraciones fijadas y evaluación en el holdout histórico."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from nli_project.models.classifiers import build_classifier
from nli_project.features.minilm import full_features
from experiments.scripts.replay_dense_rf import _load_split, _load_val_embeddings
from experiments.scripts.run_dense_comparison import ROOT, load_inputs


TUNED = ROOT / "experiments/results/minilm/tuning"
PREVIOUS = ROOT / "experiments/results/minilm/comparison"
SEEDS = (42, 7, 2024)
TAGS = {
    "mlp_256_alpha0.0003": "mlp", "xgb_d3_500": "xgboost",
    "cat_d6_500": "catboost", "lightgbm": "lightgbm", "mlp_original": "mlp",
}
VOTERS = ("mlp_256_alpha0.0003", "xgb_d3_500", "cat_d6_500", "lightgbm")


def predict(tag, params, seed, x, y, test):
    model = build_classifier(TAGS[tag], random_state=seed, **params)
    model.fit(x, y)
    np.testing.assert_array_equal(model.classes_, [0, 1, 2])
    if TAGS[tag] == "xgboost":
        device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not device.startswith("cuda"):
            raise RuntimeError(f"XGBoost entrenado en {device}")
    return model.predict_proba(test)


def save_results(output, filename, metadata, predictions):
    predictions["voto_4"] = np.mean([predictions[tag] for tag in VOTERS], axis=0)
    frame = metadata.copy()
    rows = []
    for tag, probabilities in predictions.items():
        if not np.isfinite(probabilities).all() or probabilities.min() < 0:
            raise ValueError(f"Probabilidades inválidas para {tag}")
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
        frame[[f"{tag}_p{i}" for i in range(3)]] = probabilities
        pred = probabilities.argmax(axis=1)
        for fold, group in metadata.groupby("fold"):
            y = group["label"].to_numpy()
            guess = pred[group.index]
            en = group["lang_abv"].eq("en").to_numpy()
            correct = guess == y
            rows.append({
                "model": tag, "fold": fold, "n": len(group),
                "accuracy": correct.mean(), "macro_f1": f1_score(y, guess, average="macro"),
                "accuracy_en": correct[en].mean(), "accuracy_non_en": correct[~en].mean(),
            })
    frame.to_csv(output / f"{filename}.csv.gz", index=False)
    return pd.DataFrame(rows)


def confirm(output):
    manifest = json.loads((TUNED / "run.json").read_text(encoding="utf-8"))
    configs = {
        tag: manifest["candidates"][tag] if tag in manifest["candidates"] else manifest["params"][model]
        for tag, model in TAGS.items()
    }
    configs["lightgbm"] = configs["lightgbm"] | {"n_jobs": 8}
    (output / "run.json").write_text(json.dumps({
        "encoder": manifest["encoder"], "features": manifest["features"], "seeds": SEEDS,
        "seed_reused": 42, "folds": 5, "configs": configs, "uniform_vote": VOTERS,
        "finalists": ["mlp_256_alpha0.0003", "xgb_d3_500"], "holdout_seed": 42,
        "versions": manifest["versions"],
    }, indent=2), encoding="utf-8")
    train, x = load_inputs()
    y = train.label.to_numpy()
    meta = train[["id", "label", "lang_abv"]].copy()
    seed42 = pd.read_csv(PREVIOUS / "oof.csv.gz")
    cv_rows = []
    for seed in SEEDS:
        folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(x, y))
        indices = np.empty(len(train), dtype=int)
        for number, (_, test) in enumerate(folds, 1):
            indices[test] = number
        data = meta.assign(fold=indices)
        predictions = {}
        for tag, params in configs.items():
            if seed == 42:
                if tag in manifest["candidates"]:
                    saved = pd.read_csv(TUNED / f"{tag}.csv.gz")
                    columns = ["p0", "p1", "p2"]
                else:
                    saved = seed42
                    columns = [f"{TAGS[tag]}_p{i}" for i in range(3)]
                keys = ["id", "label", "fold"]
                pd.testing.assert_frame_equal(data[keys], saved[keys])
                predictions[tag] = saved[columns].to_numpy()
                continue
            probs = np.full((len(train), 3), np.nan)
            for number, (fit, test) in enumerate(folds, 1):
                print(f"EN CURSO | semilla {seed} | {tag} | fold {number}/5", flush=True)
                probs[test] = predict(tag, params, seed, x[fit], y[fit], x[test])
            predictions[tag] = probs
        scores = save_results(output, f"oof_{seed}", data, predictions)
        cv_rows.append(scores.assign(seed=seed))
        pd.concat(cv_rows).to_csv(output / "folds.csv", index=False)

    scores = pd.concat(cv_rows)
    metrics = ["accuracy", "macro_f1", "accuracy_en", "accuracy_non_en"]
    scores.groupby(["model", "seed"])[metrics].mean().to_csv(output / "seeds.csv")
    scores.groupby("model")[metrics].mean().sort_values("accuracy", ascending=False).to_csv(output / "cv.csv")
    _, val = _load_split(train)
    x_val = full_features(*_load_val_embeddings(len(val)))
    predictions = {}
    for tag, params in configs.items():
        print(f"EN CURSO | holdout historico | {tag}", flush=True)
        predictions[tag] = predict(tag, params, 42, x, y, x_val)
    data = val[["id", "label", "lang_abv"]].assign(fold=1)
    save_results(output, "holdout_predictions", data, predictions).to_csv(output / "holdout.csv", index=False)
    print("TERMINADO | CV de tres semillas y holdout historico", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir
    output.mkdir(exist_ok=True, parents=True)
    with (output / "pid.txt").open("x") as pid:
        pid.write(str(os.getpid()))
    try:
        with threadpool_limits(limits=8):
            confirm(output)
    except Exception as exc:
        print(f"ERROR | {type(exc).__name__}: {exc}", flush=True)
        raise
