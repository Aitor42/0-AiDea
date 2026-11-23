"""Segunda ronda MiniLM. Ejecutar con --output-dir CARPETA_NUEVA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from nli_project.models.classifiers import build_classifier
from experiments.scripts.replay_dense_rf import replay_rf
from experiments.scripts.run_dense_comparison import ROOT, load_inputs


def candidates(base):
    for model in ("xgboost", "catboost"):
        gpu = {"device": "cuda:0", "n_jobs": 8} if model == "xgboost" else {
            "task_type": "GPU", "devices": "0", "thread_count": 8, "gpu_ram_part": 0.65,
        }
        spec = base[model] | gpu
        yield model, f"{model}_gpu_control", spec
        if model == "xgboost":
            for depth in (3, 4):
                yield model, f"xgb_d{depth}_500", spec | {
                    "max_depth": depth, "learning_rate": 0.05, "n_estimators": 500,
                }
        else:
            for trees in (500, 1000):
                yield model, f"cat_d6_{trees}", spec | {
                    "iterations": trees, "learning_rate": 0.05,
                }
    for width in (256, 512):
        for alpha in (1e-5, 3e-4):
            yield "mlp", f"mlp_{width}_alpha{alpha:g}", base["mlp"] | {
                "hidden_layer_sizes": [width], "alpha": alpha,
                "max_iter": 200, "n_iter_no_change": 10,
            }
    for leaf in (2, 10):
        yield "random_forest", f"rf_leaf{leaf}", base["random_forest"] | {
            "min_samples_leaf": leaf, "n_jobs": 8,
        }


def run(output):
    previous = ROOT / "experiments/results/minilm/comparison"
    manifest = json.loads((previous / "run.json").read_text(encoding="utf-8"))
    configs = list(candidates(manifest["params"]))
    manifest["candidates"] = {tag: params for _, tag, params in configs}
    manifest["devices"] = {"mlp": "cpu", "random_forest": "cpu", "xgboost": "cuda:0", "catboost": "GPU:0"}
    manifest.pop("device")
    manifest.pop("ensemble_uniform")
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    train, x = load_inputs()
    y = train["label"].to_numpy()
    en = train["lang_abv"].eq("en").to_numpy()
    ids = pd.read_csv(previous / "oof.csv.gz", usecols=["id", "label", "fold"])
    pd.testing.assert_frame_equal(train[["id", "label"]], ids[["id", "label"]])
    seed = manifest["seed"]
    folds = list(StratifiedKFold(manifest["folds"], shuffle=True, random_state=seed).split(x, y))
    for number, (_, test) in enumerate(folds, 1):
        np.testing.assert_array_equal(test, np.flatnonzero(ids["fold"] == number))

    total = len(configs) * len(folds)
    results, fold_rows = [], []
    with threadpool_limits(limits=8):
        for name, tag, params in configs:
            probs = np.full((len(train), 3), np.nan)
            acc, f1 = [], []
            for number, (fit, test) in enumerate(folds, 1):
                print(f"EN CURSO {len(fold_rows)}/{total} | {tag} | fold {number}", flush=True)
                start = perf_counter()
                model = build_classifier(name, random_state=seed, **params)
                model.fit(x[fit], y[fit])
                if name == "xgboost":
                    device = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
                    if not device.startswith("cuda"):
                        raise RuntimeError(f"XGBoost se ha entrenado en {device}")
                if name == "catboost" and model.get_all_params()["task_type"] != "GPU":
                    raise RuntimeError("CatBoost requiere GPU en esta ronda")
                np.testing.assert_array_equal(model.classes_, [0, 1, 2])
                probs[test] = model.predict_proba(x[test])
                pred = probs[test].argmax(axis=1)
                acc.append(float(np.mean(pred == y[test])))
                f1.append(f1_score(y[test], pred, average="macro"))
                fold_rows.append({
                    "candidate": tag, "fold": number, "accuracy": acc[-1], "macro_f1": f1[-1],
                    "seconds": round(perf_counter() - start, 2),
                })
                pd.DataFrame(fold_rows).to_csv(output / "folds.csv", index=False)
            if not np.isfinite(probs).all() or np.min(probs) < 0:
                raise ValueError(f"Probabilidades no válidas: {tag}")
            np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-5)
            oof = ids.assign(p0=probs[:, 0], p1=probs[:, 1], p2=probs[:, 2])
            oof.to_csv(output / f"{tag}.csv.gz", index=False)
            correct = probs.argmax(axis=1) == y
            results.append({
                "model": name, "candidate": tag, "accuracy_mean": np.mean(acc),
                "accuracy_std": np.std(acc), "macro_f1_mean": np.mean(f1),
                "accuracy_en": correct[en].mean(), "accuracy_non_en": correct[~en].mean(),
            })
            pd.DataFrame(results).sort_values(
                ["accuracy_mean", "macro_f1_mean"], ascending=False
            ).to_csv(output / "summary.csv", index=False)
        print(f"EN CURSO {total}/{total} | reproduciendo RF historico en su holdout", flush=True)
        replay_rf(train, x, output)
    print(f"TERMINADO | {total} ajustes CV y reproduccion del Random Forest", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    output = parser.parse_args().output_dir
    output.mkdir(exist_ok=True, parents=True)
    # La creación exclusiva evita mezclar resultados de dos ejecuciones.
    with (output / "pid.txt").open("x") as pid:
        pid.write(str(os.getpid()))
    try:
        run(output)
    except Exception as exc:
        print(f"ERROR | {type(exc).__name__}: {exc}", flush=True)
        raise
