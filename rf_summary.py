"""Resume `rf_experiments_results.jsonl` en una tabla y compara con la base.

Uso (desde la raíz del proyecto):
    python rf_summary.py

Escribe `rf_table.csv` junto a este fichero.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
rows = [
    json.loads(line)
    for line in (ROOT / "rf_experiments_results.jsonl").read_text().splitlines()
    if line.strip()
]
frame = pd.DataFrame(rows)

# Última ejecución de cada configuración con semilla 42.
single = frame[~frame["name"].str.contains("@")]
single = single.drop_duplicates(subset="name", keep="last").set_index("name")
base = (
    single.loc["base"]
    if "base" in single.index
    else frame[frame["name"] == "base@seed42"].iloc[-1]
)

out = []
for name, row in single.iterrows():
    diff = np.array(row["fold_accuracy"]) - np.array(base["fold_accuracy"])
    _, p_value = stats.ttest_rel(row["fold_accuracy"], base["fold_accuracy"])
    out.append(
        {
            "config": name,
            "cv_acc": row["accuracy_mean"],
            "cv_f1": row["macro_f1_mean"],
            "delta_pts": 100 * diff.mean(),
            "fold_wins": int((diff > 0).sum()),
            "p_vs_base": p_value,
            "fit_s": row["fit_seconds_mean"],
            "features": row["select_k"] if row["select_k"] else "todas",
            "tokenizer": row["preprocessing"].get("tokenizer"),
            "ngram": str(row["vectorizer"].get("word_ngram_range")),
            "min_df": row["vectorizer"].get("min_df"),
        }
    )

table = pd.DataFrame(out).sort_values("cv_acc", ascending=False)
table.to_csv(ROOT / "rf_table.csv", index=False)
pd.set_option("display.width", 200)
print(table.round(4).to_string(index=False))

confirm = frame[frame["name"].str.contains("@")].copy()
confirm["config"] = confirm["name"].str.split("@").str[0]
print("\nCV repetida (medias por semilla):")
for config, group in confirm.groupby("config"):
    means = group["accuracy_mean"].tolist()
    print(
        f"  {config:32s} {np.mean(means):.6f} ± {np.std(means, ddof=0):.6f}  "
        f"{[round(m, 4) for m in means]}"
    )

base_folds = np.concatenate(confirm[confirm["config"] == "base"]["fold_accuracy"].tolist())
print("\nComparación pareada de los 15 folds frente a la base:")
for config, group in confirm.groupby("config"):
    if config == "base":
        continue
    folds = np.concatenate(group["fold_accuracy"].tolist())
    diff = folds - base_folds
    _, p = stats.ttest_rel(folds, base_folds)
    print(
        f"  {config:32s} Δ={100 * diff.mean():+.2f} pts | "
        f"folds ganados={int((diff > 0).sum())}/15 | p={p:.2e}"
    )
