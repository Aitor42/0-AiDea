"""Ablate relation and spaCy POS features on English MiniLM examples."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

from classifiers import build_classifier
from pos_features import POSRatioFeatures
from relation_features import RelationFeatures


ROOT = Path(__file__).resolve().parent
PREPROCESSING_CONFIG = {
    "tokenizer": "spacy",
    "reduction": "spacy_lemma",
    "use_pos": True,
    "stopword_method": "none",
}
MODEL_PARAMS = {
    "hidden_layer_sizes": [256],
    "alpha": 3e-4,
    "batch_size": 256,
    "max_iter": 200,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
}
ARMS = ("minilm", "minilm_relation", "minilm_relation_pos")


def _load_spacy_cache(train: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cache_dir = ROOT / "artifacts" / "preprocessed"
    required_ids = train["id"].tolist()
    for path in sorted(cache_dir.glob("*.pkl")):
        try:
            frame = pd.read_pickle(path)
        except Exception:
            continue
        if not isinstance(frame, pd.DataFrame) or "id" not in frame:
            continue
        if frame["id"].tolist() != required_ids:
            continue
        config = frame["preprocessing_config"].iloc[0]
        if isinstance(config, str):
            config = json.loads(config)
        if not all(config.get(key) == value for key, value in PREPROCESSING_CONFIG.items()):
            continue
        for column in ("premise_processed", "hypothesis_processed"):
            if column not in frame:
                raise ValueError(f"Falta la columna spaCy {column} en {path.name}")
        return frame, path.name
    raise FileNotFoundError(
        "No se encontró la caché spaCy etiquetada para estas filas y esta configuración."
    )


def _auxiliary_features(
    frame: pd.DataFrame,
    english: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    columns = ("premise_processed", "hypothesis_processed")
    relation = RelationFeatures(columns=columns).fit_transform(frame)
    pos_transformer = POSRatioFeatures(columns=columns).fit(frame)
    pos = pos_transformer.transform(frame)
    english_tagged_tokens = sum(
        isinstance(token, (tuple, list)) and len(token) >= 2
        for column in columns
        for cell in frame.loc[english, column]
        for token in cell
    )
    total_tagged_tokens = sum(
        isinstance(token, (tuple, list)) and len(token) >= 2
        for column in columns
        for cell in frame[column]
        for token in cell
    )
    if (
        english_tagged_tokens == 0
        or not np.isfinite(pos).all()
        or not np.isfinite(relation).all()
    ):
        raise ValueError("La caché no contiene rasgos spaCy POS válidos.")
    return (
        relation.astype(np.float32),
        pos.astype(np.float32),
        int(total_tagged_tokens),
        int(english_tagged_tokens),
    )


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities.argmax(axis=1)
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro")),
    }


def run_ablation(
    train: pd.DataFrame,
    x: np.ndarray,
    output: str | Path,
) -> Path:
    """Run three comparable English-only grouped folds and save OOF evidence."""
    if len(train) != len(x) or x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("train y x deben estar alineados y contener features finitas.")
    english = train["lang_abv"].astype(str).str.lower().eq("en").to_numpy()
    cache, cache_filename = _load_spacy_cache(train)
    if not cache["id"].reset_index(drop=True).equals(train["id"].reset_index(drop=True)):
        raise ValueError("La caché spaCy está desalineada con train.")
    relation, pos, tagged_tokens, english_tagged_tokens = _auxiliary_features(
        cache, english
    )

    english_idx = np.flatnonzero(english)
    subset = train.iloc[english_idx].reset_index(drop=True)
    labels = subset["label"].to_numpy()
    groups = subset["premise"].to_numpy()
    auxiliary_arrays = {
        "minilm_relation": relation[english_idx],
        "minilm_relation_pos": np.hstack(
            [relation[english_idx], pos[english_idx]]
        ),
    }
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(x[english_idx], labels, groups))
    for fit, valid in folds:
        if set(groups[fit]) & set(groups[valid]):
            raise AssertionError("Una premisa aparece a ambos lados de un fold.")

    output_dir = Path(output) / "dense_spacy_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "encoder": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "representation": ["p", "h", "abs(p-h)", "p*h"],
        "preprocessing_cache": cache_filename,
        "preprocessing_config": PREPROCESSING_CONFIG,
        "tagged_tokens": tagged_tokens,
        "english_tagged_tokens": english_tagged_tokens,
        "auxiliary_scaling": {
            "method": "MinMaxScaler",
            "fit": "training portion of each fold",
            "clip": True,
            "columns": "auxiliary relation and POS only",
        },
        "english_rows": int(len(subset)),
        "unique_premises": int(pd.Series(groups).nunique()),
        "folds": 5,
        "seed": 42,
        "split": "StratifiedGroupKFold grouped by premise, English only",
        "model": "MLPClassifier",
        "model_params": MODEL_PARAMS,
        "feature_dimensions": {
            "minilm": int(x.shape[1]),
            "minilm_relation": int(x.shape[1] + relation.shape[1]),
            "minilm_relation_pos": int(x.shape[1] + relation.shape[1] + pos.shape[1]),
        },
    }
    (output_dir / "run.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    labels_oof = np.full((len(subset), len(ARMS), 3), np.nan, dtype=np.float32)
    fold_by_row = np.zeros(len(subset), dtype=np.int8)
    for fold_number, (_, valid) in enumerate(folds, start=1):
        fold_by_row[valid] = fold_number
    rows = []
    for arm_index, arm in enumerate(ARMS):
        auxiliary = auxiliary_arrays.get(arm)
        for fold_number, (fit, valid) in enumerate(folds, start=1):
            started = perf_counter()
            if auxiliary is None:
                fit_features = x[english_idx[fit]]
                valid_features = x[english_idx[valid]]
            else:
                scaler = MinMaxScaler(clip=True)
                scaled_fit = scaler.fit_transform(auxiliary[fit])
                scaled_valid = scaler.transform(auxiliary[valid])
                fit_features = np.hstack([x[english_idx[fit]], scaled_fit])
                valid_features = np.hstack([x[english_idx[valid]], scaled_valid])
            with threadpool_limits(limits=8):
                model = build_classifier(
                    "mlp",
                    random_state=42,
                    **MODEL_PARAMS,
                )
                model.fit(fit_features, labels[fit])
                if not np.array_equal(model.classes_, [0, 1, 2]):
                    raise ValueError(f"Etiquetas inesperadas en {arm}: {model.classes_}")
                probabilities = model.predict_proba(valid_features)
            if (
                not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not np.allclose(
                    probabilities.sum(axis=1), 1.0, atol=1e-5, rtol=0.0
                )
            ):
                raise ValueError(f"Probabilidades no válidas para {arm}, fold {fold_number}.")
            labels_oof[valid, arm_index] = probabilities
            row = {"arm": arm, "fold": fold_number}
            row.update(_metrics(labels[valid], probabilities))
            row["fit_predict_seconds"] = round(perf_counter() - started, 2)
            rows.append(row)
            pd.DataFrame(rows).to_csv(output_dir / "folds.csv", index=False)
            print(
                f"EN CURSO | {arm} | fold {fold_number}/5 | "
                f"accuracy={row['accuracy']:.4f} | {row['fit_predict_seconds']} s",
                flush=True,
            )

        if not np.isfinite(labels_oof[:, arm_index]).all():
            raise ValueError(f"Predicciones OOF incompletas para {arm}.")

    oof = subset[["id", "label", "premise"]].copy()
    oof["fold"] = fold_by_row
    for arm_index, arm in enumerate(ARMS):
        for class_id in range(3):
            oof[f"{arm}_p{class_id}"] = labels_oof[:, arm_index, class_id]
    oof.to_csv(output_dir / "oof.csv.gz", index=False)

    summaries = []
    for arm_index, arm in enumerate(ARMS):
        probabilities = labels_oof[:, arm_index]
        predictions = probabilities.argmax(axis=1)
        per_fold = [_metrics(labels[valid], probabilities[valid]) for _, valid in folds]
        summaries.append({
            "arm": arm,
            "n": len(subset),
            "accuracy_oof": float(accuracy_score(labels, predictions)),
            "macro_f1_oof": float(f1_score(labels, predictions, average="macro")),
            "accuracy_fold_mean": float(np.mean([row["accuracy"] for row in per_fold])),
            "accuracy_fold_std": float(np.std([row["accuracy"] for row in per_fold])),
            "macro_f1_fold_mean": float(np.mean([row["macro_f1"] for row in per_fold])),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print("TERMINADO | ablación spaCy en inglés", flush=True)
    return output_dir
