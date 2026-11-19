"""Busca configuraciones de profundidad y chi-cuadrado con MiniLM."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler
from threadpoolctl import threadpool_limits

from classifiers import build_classifier
from preprocessing import MULTILINGUAL_PREPROCESSING_PARAMS, preprocess_partition
from replay_dense_rf import _full_features
from run_dense_comparison import ENCODER, load_inputs


def preprocessed_embeddings(train, tokenizer):
    import torch
    from sentence_transformers import SentenceTransformer

    params = {**MULTILINGUAL_PREPROCESSING_PARAMS, "tokenizer": tokenizer}
    processed = preprocess_partition(train, params)
    texts = pd.concat([processed[column].str.join(" ") for column in
                       ("premise_processed", "hypothesis_processed")], ignore_index=True)
    indices, unique = pd.factorize(texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"MiniLM: {len(unique)} textos tokenizados con {tokenizer} ({device})", flush=True)
    with threadpool_limits(limits=8):
        encoder = SentenceTransformer(ENCODER, local_files_only=True, device=device)
        vectors = encoder.encode(unique.tolist(), batch_size=64, normalize_embeddings=True,
                                 show_progress_bar=False)
    return _full_features(*np.split(vectors[indices], 2))


def tune(output: Path, seed: int = 42, inputs=("original", "spacy", "whitespace")) -> pd.DataFrame:
    """Evalúa las entradas con la misma rejilla y tres folds por premisa."""
    if output.exists():
        raise FileExistsError(output)
    train, original = load_inputs()
    y = train.label.to_numpy()
    folds = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed)
                 .split(train, y, train.premise))
    models = {
        "catboost": ((4, 6), "depth", {
            "iterations": 300, "learning_rate": 0.1, "task_type": "GPU",
            "devices": "0", "thread_count": 8,
        }),
        "xgboost": ((3, 5), "max_depth", {
            "n_estimators": 300, "learning_rate": 0.05, "device": "cuda:0", "n_jobs": 8,
        }),
    }
    rows = []
    with threadpool_limits(limits=8):
        for input_name in inputs:
            vectors = original if input_name == "original" else preprocessed_embeddings(train, input_name)
            for i, (fit, val) in enumerate(folds, 1):
                print(f"{input_name} | fold {i}/{len(folds)}", flush=True)
                x_train, x_val = vectors[fit], vectors[val]
                scaled_fit = MinMaxScaler().fit_transform(x_train)
                for k in (None, 300, 768):
                    selected_train, selected_val = x_train, x_val
                    if k is not None:
                        selector = SelectKBest(chi2, k=k).fit(scaled_fit, y[fit])
                        selected_train, selected_val = (selector.transform(x_train),
                                                        selector.transform(x_val))
                    for name, (depths, param, settings) in models.items():
                        for depth in depths:
                            model = build_classifier(name, random_state=seed,
                                                     **settings, **{param: depth})
                            model.fit(selected_train, y[fit])
                            predictions = model.predict(selected_val)
                            rows.append({
                                "input": input_name, "model": name, "depth": depth,
                                "chi2_k": "all" if k is None else str(k), "fold": i,
                                "accuracy": accuracy_score(y[val], predictions),
                                "macro_f1": f1_score(y[val], predictions, average="macro"),
                            })

    scores = pd.DataFrame(rows)
    summary = (scores.groupby(["input", "model", "depth", "chi2_k"], as_index=False)
               [["accuracy", "macro_f1"]].mean()
               .sort_values(["accuracy", "macro_f1"], ascending=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        scores.to_csv(stage / "folds.csv", index=False)
        summary.to_csv(stage / "summary.csv", index=False)
        stage.rename(output)
    return summary
