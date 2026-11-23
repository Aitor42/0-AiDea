"""Replay the previously reported MiniLM Random Forest holdout measurement."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from nli_project.features.minilm import full_features
from nli_project.models.classifiers import build_classifier


ROOT = Path(__file__).resolve().parents[2]
ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_RESULTS = ROOT / "experiments/results/random_forest/embeddings" / (
    "rf_embeddings_results_sentence-transformers__"
    "paraphrase-multilingual-MiniLM-L12-v2.json"
)


def _historical_rf_params() -> dict:
    """Load the saved configuration used by the historical RF experiment."""
    config = ROOT / "experiments/results/random_forest/tfidf" / "final_config.json"
    return json.loads(config.read_text(encoding="utf-8"))["classifier_params"]


def _load_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(ROOT / "experiments/data/train.csv")
    expected_train, expected_val = train_test_split(
        source, test_size=0.2, random_state=42, stratify=source["label"]
    )
    expected_train = expected_train.reset_index(drop=True)
    expected_val = expected_val.reset_index(drop=True)
    pd.testing.assert_frame_equal(train.reset_index(drop=True), expected_train)

    cached_train = pd.read_csv(ROOT / "experiments/results/validation/historical_split/train.csv")
    cached_val = pd.read_csv(ROOT / "experiments/results/validation/historical_split/val.csv")
    pd.testing.assert_frame_equal(cached_train, expected_train)
    pd.testing.assert_frame_equal(cached_val, expected_val)
    return expected_train, expected_val


def _load_val_embeddings(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    embedding_dir = ROOT / "experiments/.local/embeddings"
    slug = ENCODER.replace("/", "__")
    embeddings = []
    for side in ("p", "h"):
        path = embedding_dir / f"{slug}_val_{side}.npy"
        array = np.load(path, allow_pickle=False)
        if array.shape != (n_rows, 384) or not np.isfinite(array).all():
            raise ValueError(f"Caché de validación incompatible: {path}")
        embeddings.append(array)
    return embeddings[0], embeddings[1]





def replay_rf(
    train: pd.DataFrame,
    features: np.ndarray,
    output_dir: str | Path,
) -> dict:
    """Refit the historical RF on the 80% train split and replay its 20% holdout.

    This is an audit result only. Callers must not use it for CV tuning or model
    selection because this holdout was used in the original representation study.
    """
    if features.shape != (len(train), 1536) or not np.isfinite(features).all():
        raise ValueError("features debe contener los 1536 rasgos MiniLM por fila train.")

    train, val = _load_split(train)
    val_premise, val_hypothesis = _load_val_embeddings(len(val))
    val_features = full_features(val_premise, val_hypothesis)
    if val_features.shape[1] != features.shape[1]:
        raise ValueError("Los rasgos de train y holdout no tienen el mismo tamaño.")

    result_file = ROOT / "experiments/results/minilm/comparison" / "run.json"
    if not result_file.exists():
        raise FileNotFoundError(f"Falta el manifiesto de comparación: {result_file}")
    manifest = json.loads(result_file.read_text(encoding="utf-8"))
    historical_params = _historical_rf_params()
    dense_params = manifest["params"]["random_forest"]
    if dense_params != historical_params:
        raise ValueError(
            "La configuración RF del barrido MiniLM difiere de la del experimento original."
        )

    historical = json.loads(EMBEDDING_RESULTS.read_text(encoding="utf-8"))
    reference = next(
        row for row in historical["rows"] if row["name"] == "emb full [p,h,|d|,p*h]"
    )
    if len(train) != historical["n_train"] or len(val) != historical["n_val"]:
        raise ValueError("El tamaño del split no coincide con el experimento RF guardado.")

    labels = train["label"].to_numpy()
    val_labels = val["label"].to_numpy()
    if not np.array_equal(np.unique(labels), np.array([0, 1, 2])):
        raise ValueError("El entrenamiento no contiene las tres clases en orden esperado.")

    started = perf_counter()
    with threadpool_limits(limits=8):
        model = build_classifier("random_forest", random_state=42, n_jobs=8, **historical_params)
        model.fit(features, labels)
        predicted = model.predict(val_features)
    elapsed = perf_counter() - started

    english = val["lang_abv"].to_numpy() == "en"
    measured = {
        "accuracy": float(accuracy_score(val_labels, predicted)),
        "f1_macro": float(f1_score(val_labels, predicted, average="macro")),
        "accuracy_english": float(accuracy_score(val_labels[english], predicted[english])),
        "accuracy_non_english": float(accuracy_score(val_labels[~english], predicted[~english])),
        "n_english": int(english.sum()),
        "n_non_english": int((~english).sum()),
        "n_train": len(train),
        "n_val": len(val),
        "fit_predict_seconds": round(elapsed, 2),
    }
    keys = (
        "accuracy",
        "f1_macro",
        "accuracy_english",
        "accuracy_non_english",
    )
    result = {
        "encoder": ENCODER,
        "features": ["p", "h", "abs(p-h)", "p*h"],
        "classifier": "random_forest",
        "params": historical_params,
        "device": "cpu",
        "seed": 42,
        "comparison_scope": "Replay of the holdout previously used by the embedding experiment.",
        "historical": {key: reference[key] for key in keys},
        "measured": measured,
        "delta": {key: measured[key] - reference[key] for key in keys},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rf_holdout_replay.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
