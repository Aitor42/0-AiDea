"""Compara los modelos fijados en folds agrupados por premisa y ejecuta la ablación inglesa."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from run_dense_comparison import ROOT, load_inputs
from run_dense_confirmation import VOTERS, predict, save_results


def grouped_comparison(train, x, output):
    previous = ROOT / "results_dense_confirmation"
    manifest = json.loads((previous / "run.json").read_text(encoding="utf-8"))
    configs = manifest["configs"]
    seeds = manifest["seeds"]
    n_folds = manifest["folds"]
    two_voters = ["mlp_256_alpha0.0003", "xgb_d3_500"]
    if tuple(manifest["uniform_vote"]) != VOTERS:
        raise ValueError("La composición del voto no coincide con la confirmación previa")
    (output / "protocol.json").write_text(json.dumps({
        "configs": configs, "seeds": seeds, "folds": n_folds,
        "encoder": manifest["encoder"], "features": manifest["features"],
        "groups": "premise (texto exacto)", "ensemble_4": VOTERS,
        "ensemble_2": two_voters, "versions": manifest["versions"],
    }, indent=2), encoding="utf-8")
    y = train.label.to_numpy()
    groups = train["premise"]
    if groups.isna().any() or not train.id.is_unique:
        raise ValueError("Se requieren premisas definidas e identificadores únicos")
    meta = train[["id", "label", "lang_abv"]]
    audits, scores = [], []
    completed = 0
    total = len(seeds) * n_folds * len(configs)
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        folds = list(splitter.split(x, y, groups))
        indices = np.zeros(len(y), dtype=int)
        for k, (fit, val) in enumerate(folds, 1):
            seen, held_out = set(groups.iloc[fit]), set(groups.iloc[val])
            overlap = len(seen & held_out)
            if overlap or indices[val].any():
                raise ValueError(f"Solapamiento en semilla {seed}, fold {k}")
            indices[val] = k
            audits.append({
                "seed": seed, "fold": k, "fit_rows": len(fit), "val_rows": len(val),
                "fit_premises": len(seen), "val_premises": len(held_out),
                "shared_premises": overlap,
            })
        if np.any(indices == 0):
            raise ValueError("Hay ejemplos sin fold asignado")
        pd.DataFrame(audits).to_csv(output / "split_audit.csv", index=False)
        predictions = {}
        for tag, params in configs.items():
            probs = np.full((len(y), 3), np.nan)
            for k, (fit, val) in enumerate(folds, 1):
                print(f"EN CURSO | agrupado {completed}/{total} | semilla {seed} | {tag} | fold {k}", flush=True)
                probs[val] = predict(tag, params, seed, x[fit], y[fit], x[val])
                completed += 1
            predictions[tag] = probs
        predictions["voto_2"] = np.mean([predictions[tag] for tag in two_voters], axis=0)
        result = save_results(output, f"grouped_oof_{seed}", meta.assign(fold=indices), predictions)
        scores.append(result.assign(seed=seed))
        pd.concat(scores).to_csv(output / "grouped_folds.csv", index=False)

    results = pd.concat(scores)
    metrics = ["accuracy", "macro_f1", "accuracy_en", "accuracy_non_en"]
    results.groupby(["model", "seed"])[metrics].mean().to_csv(output / "grouped_seeds.csv")
    summary = results.groupby("model")[metrics].mean()
    baseline = pd.read_csv(ROOT / "results_dense_votes" / "metrics_summary.csv").set_index("model")
    baseline = baseline.rename(index={"voto_2_mlp_xgb": "voto_2"})
    summary["row_cv_accuracy"] = baseline.loc[summary.index, "accuracy_mean"]
    summary["delta_accuracy_pp"] = 100 * (summary["accuracy"] - summary["row_cv_accuracy"])
    summary.sort_values("accuracy", ascending=False).to_csv(output / "grouped_summary.csv")


def run(output):
    train, x = load_inputs()
    with threadpool_limits(limits=8):
        grouped_comparison(train, x, output)
        from dense_spacy_ablation import run_ablation

        print("EN CURSO | ablacion inglesa con spaCy", flush=True)
        run_ablation(train, x, output)
    print("TERMINADO | comparacion agrupada y ablacion inglesa", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pid.txt").open("x") as pid:
        pid.write(str(os.getpid()))
    try:
        run(output)
    except Exception as exc:
        print(f"ERROR | {type(exc).__name__}: {exc}", flush=True)
        raise
