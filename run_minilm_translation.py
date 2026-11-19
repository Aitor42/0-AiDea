"""Compara MiniLM con texto original y traducido en los mismos folds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from FINAL.translator import load_translations
from replay_dense_rf import _full_features
from run_dense_comparison import ENCODER, ROOT, load_inputs
from run_dense_confirmation import predict


def run(translations: Path, output: Path, seed=42, download_model=False):
    if output.exists():
        raise FileExistsError(output)
    options = {"dtype": {"id": str}, "keep_default_na": False}
    official = pd.read_csv(ROOT / "data/train.csv", **options)
    translated = load_translations(official, translations).set_index("id")
    train, original = load_inputs()
    rows = translated.loc[train.id]
    manifest = json.loads((ROOT / "results_dense_confirmation/run.json").read_text(encoding="utf-8"))
    tags = manifest["finalists"]
    configs = {tag: manifest["configs"][tag] for tag in tags}
    folds = list(StratifiedGroupKFold(manifest["folds"], shuffle=True, random_state=seed)
                 .split(train, train.label, train.premise))
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(8)
    encoder = SentenceTransformer(ENCODER, local_files_only=not download_model, device=device)
    output.mkdir(parents=True, exist_ok=False)
    non_en = rows.lang_abv.ne("en").to_numpy()
    premise, hypothesis = original[:, :384].copy(), original[:, 384:768].copy()
    texts = pd.concat([rows.loc[non_en, "premise_translated"],
                       rows.loc[non_en, "hypothesis_translated"]], ignore_index=True)
    indices, unique = pd.factorize(texts)
    print(f"Codificando {len(unique)} textos traducidos en {device}", flush=True)
    vectors = encoder.encode(unique.tolist(), batch_size=64, normalize_embeddings=True,
                             show_progress_bar=False)
    premise[non_en], hypothesis[non_en] = np.split(vectors[indices], 2)
    del encoder
    torch.cuda.empty_cache()
    y = train.label.to_numpy()
    predictions = train[["id", "label", "lang_abv"]].copy()
    scores = {}
    with threadpool_limits(limits=8):
        for arm, features in (("original", original), ("translated", _full_features(premise, hypothesis))):
            guesses = np.empty(len(y), dtype=int)
            for k, (fit, val) in enumerate(folds, 1):
                print(f"{arm} | fold {k}/{len(folds)}", flush=True)
                predictions.loc[val, "fold"] = k
                probabilities = [predict(tag, configs[tag], seed, features[fit], y[fit], features[val])
                                 for tag in tags]
                guesses[val] = np.mean(probabilities, axis=0).argmax(axis=1)
            predictions[arm] = guesses
            scores[arm] = {
                "accuracy": accuracy_score(y, guesses),
                "macro_f1": f1_score(y, guesses, average="macro"),
                "accuracy_en": accuracy_score(y[~non_en], guesses[~non_en]),
                "accuracy_non_en": accuracy_score(y[non_en], guesses[non_en]),
            }
    predictions.to_csv(output / "predictions.csv", index=False)
    results = {"encoder": ENCODER, "seed": seed, "folds": len(folds), "models": configs,
               "groups": "original premise", "embedding_device": device, "scores": scores,
               "translations": str(translations.resolve())}
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2), flush=True)
