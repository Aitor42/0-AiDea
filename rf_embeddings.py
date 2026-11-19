"""Experimento controlado de embeddings para el Random Forest de NLI.

Compara, sobre el mismo split holdout que el modelo final, la representación
TF-IDF de ``pair_diff`` con varias representaciones densas construidas a partir
de un sentence transformer multilingüe. Todo se evalúa con el mismo Random
Forest (sin tocar hiperparámetros salvo un diagnóstico de ``max_features``).

El objetivo es responder a una pregunta concreta: ¿los embeddings recuperan
parte de la brecha entre inglés y el resto de lenguas? Por eso cada variante
se mide en accuracy/F1 global y separada por lengua.

El script es autocontenido: replica el split de ``training.make_split`` y lee
las particiones preprocesadas de la caché de ``artifacts/preprocessed`` (no
necesita spaCy ni ``training.py``). Los embeddings se cachean en
``artifacts/embeddings`` y los resultados se guardan en
``rf_embeddings_results.json``.

Uso:
    .venv/bin/python rf_embeddings.py [modelo_sentence_transformers]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

from pos_features import POSRatioFeatures
from relation_features import RelationFeatures
from vectorizer import NLITextVectorizer

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "artifacts" / "embeddings"
PREPROCESSED_DIR = ROOT / "artifacts" / "preprocessed"
RESULTS_PATH_PREFIX = "rf_embeddings_results"
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Prefijo opcional para modelos que lo exigen (E5 recomienda "query: " tanto
# para la premisa como para la hipótesis en similitud simétrica).
EMB_PREFIX = os.environ.get("EMB_PREFIX", "")

# Configuración final del modelo (copiada de training.py para no importarlo:
# training importa spaCy y aquí solo se usan las particiones ya preprocesadas).
SELECT_K = 300
PREPROCESSING_KEY = {
    "tokenizer": "spacy",
    "reduction": "spacy_lemma",
    "use_pos": True,
    "stopword_method": "none",
}
VECTORIZER_PARAMS = {
    "method": "pair_diff",
    "columns": ("premise_processed", "hypothesis_processed"),
    "min_df": 5,
    "word_ngram_range": (1, 1),
    "use_pos": False,
}
RANDOM_FOREST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 50,
    "min_samples_leaf": 5,
    "min_samples_split": 5,
    "max_features": "sqrt",
    "criterion": "gini",
}


def make_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replica ``training.make_split`` (holdout 20 %, estratificado, seed 42)."""
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    return (
        train_df.copy().reset_index(drop=True),
        val_df.copy().reset_index(drop=True),
    )


def find_processed(ids: list[int]) -> pd.DataFrame:
    """Busca en la caché la partición preprocesada que corresponde a ``ids``."""
    for path in sorted(PREPROCESSED_DIR.glob("*.pkl")):
        try:
            frame = pd.read_pickle(path)
        except Exception:
            continue
        if not isinstance(frame, pd.DataFrame) or "premise_processed" not in frame:
            continue
        if frame["id"].tolist() != ids:
            continue
        config = frame["preprocessing_config"].iloc[0]
        if isinstance(config, str):
            config = json.loads(config)
        if all(config.get(key) == value for key, value in PREPROCESSING_KEY.items()):
            return frame
    raise FileNotFoundError(
        "No hay partición preprocesada en artifacts/preprocessed para este "
        "split; ejecuta antes el notebook o training.py."
    )


def _rf(**overrides):
    params = dict(RANDOM_FOREST_PARAMS, n_jobs=-1, random_state=42)
    params.update(overrides)
    return RandomForestClassifier(**params)


def _metrics(name, model, Xtr, Xva, ytr, yva, english, seconds):
    start = time.perf_counter()
    model.fit(Xtr, ytr)
    fit_seconds = time.perf_counter() - start
    predictions = model.predict(Xva)
    row = {
        "name": name,
        "n_features": int(Xtr.shape[1]),
        "accuracy": float(accuracy_score(yva, predictions)),
        "f1_macro": float(f1_score(yva, predictions, average="macro")),
        "accuracy_english": float(accuracy_score(yva[english], predictions[english])),
        "accuracy_non_english": float(
            accuracy_score(yva[~english], predictions[~english])
        ),
        "n_english": int(english.sum()),
        "n_non_english": int((~english).sum()),
        "fit_seconds": float(round(fit_seconds, 1)),
        "feature_seconds": float(round(seconds, 1)),
    }
    print(
        f"{name:36s} dims={row['n_features']:5d} "
        f"acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f} "
        f"en={row['accuracy_english']:.4f} "
        f"no-en={row['accuracy_non_english']:.4f} ({fit_seconds:.0f}s)",
        flush=True,
    )
    return row


def _load_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    start = time.perf_counter()
    model = SentenceTransformer(model_name)
    print(f"modelo cargado en {time.perf_counter() - start:.1f}s: {model_name}")
    return model


def encode(model, texts, cache_path: Path) -> np.ndarray:
    """Codifica textos (normalizados en L2) con caché en disco."""
    if cache_path.exists():
        return np.load(cache_path)
    start = time.perf_counter()
    embeddings = model.encode(
        [EMB_PREFIX + str(text) for text in texts],
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)
    np.save(cache_path, embeddings)
    print(
        f"  {cache_path.name}: {embeddings.shape} en "
        f"{time.perf_counter() - start:.1f}s",
        flush=True,
    )
    return embeddings


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_name.replace("/", "__")

    df = pd.read_csv(ROOT / "data" / "train.csv")
    df = df[["id", "premise", "hypothesis", "lang_abv", "language", "label"]]
    train_df, val_df = make_split(df)
    ytr, yva = train_df["label"].to_numpy(), val_df["label"].to_numpy()
    english = val_df["lang_abv"].to_numpy() == "en"

    results = {
        "model": model_name,
        "prefix": EMB_PREFIX,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "rows": [],
    }

    # 1. Referencia: TF-IDF pair_diff + POS + relación + chi2 + RF.
    start = time.perf_counter()
    train_proc = find_processed(train_df["id"].tolist())
    val_proc = find_processed(val_df["id"].tolist())
    features = FeatureUnion(
        [
            ("text", NLITextVectorizer(**VECTORIZER_PARAMS, random_state=42)),
            ("pos_ratios", POSRatioFeatures(columns=VECTORIZER_PARAMS["columns"])),
            ("relation", RelationFeatures(columns=VECTORIZER_PARAMS["columns"])),
        ]
    )
    features.fit(train_proc)
    Xtr_base = features.transform(train_proc)
    Xva_base = features.transform(val_proc)
    baseline_selector = SelectKBest(chi2, k=SELECT_K).fit(Xtr_base, ytr)
    baseline_row = _metrics(
        "baseline tfidf_pair_diff",
        _rf(),
        baseline_selector.transform(Xtr_base),
        baseline_selector.transform(Xva_base),
        ytr,
        yva,
        english,
        time.perf_counter() - start,
    )
    baseline_row["selection"] = "chi2"
    results["rows"].append(baseline_row)

    # 2. Embeddings de texto crudo (sin lematizar).
    model_start = time.perf_counter()
    model = _load_model(model_name)
    ep_tr = encode(model, train_df["premise"], CACHE_DIR / f"{slug}_train_p.npy")
    eh_tr = encode(model, train_df["hypothesis"], CACHE_DIR / f"{slug}_train_h.npy")
    ep_va = encode(model, val_df["premise"], CACHE_DIR / f"{slug}_val_p.npy")
    eh_va = encode(model, val_df["hypothesis"], CACHE_DIR / f"{slug}_val_h.npy")
    encode_seconds = time.perf_counter() - model_start
    results["encode_seconds"] = float(round(encode_seconds, 1))
    results["embedding_dim"] = int(ep_tr.shape[1])

    def pair(ep, eh):
        return np.hstack([ep, eh])

    def full(ep, eh):
        return np.hstack([ep, eh, np.abs(ep - eh), ep * eh])

    def diff_product(ep, eh):
        return np.hstack([np.abs(ep - eh), ep * eh])

    def pair_cosine(ep, eh):
        return np.hstack([ep, eh, (ep * eh).sum(axis=1, keepdims=True)])

    variants = {
        "emb concat [p,h]": (pair(ep_tr, eh_tr), pair(ep_va, eh_va)),
        "emb full [p,h,|d|,p*h]": (full(ep_tr, eh_tr), full(ep_va, eh_va)),
        "emb diff+product": (
            diff_product(ep_tr, eh_tr),
            diff_product(ep_va, eh_va),
        ),
        "emb concat + cosine": (
            pair_cosine(ep_tr, eh_tr),
            pair_cosine(ep_va, eh_va),
        ),
    }

    for name, (xtr, xva) in variants.items():
        results["rows"].append(
            _metrics(name, _rf(), xtr, xva, ytr, yva, english, encode_seconds)
        )

    # 3. Reducción de dimensión antes del bosque.
    pca = PCA(n_components=200, random_state=42).fit(pair(ep_tr, eh_tr))
    results["rows"].append(
        _metrics(
            "emb concat + PCA200",
            _rf(),
            pca.transform(pair(ep_tr, eh_tr)),
            pca.transform(pair(ep_va, eh_va)),
            ytr,
            yva,
            english,
            encode_seconds,
        )
    )

    # 4. Diagnóstico: ¿el bosque se queda corto con max_features="sqrt"?
    results["rows"].append(
        _metrics(
            "emb full + RF max_features=0.1",
            _rf(max_features=0.1),
            full(ep_tr, eh_tr),
            full(ep_va, eh_va),
            ytr,
            yva,
            english,
            encode_seconds,
        )
    )

    # 5. Híbrido: TF-IDF + POS + relación + embeddings, selección f_classif
    #    (chi2 no admite valores negativos). 400 features para no dar al bloque
    #    denso más peso del que tendría en el modelo final.
    Xtr_hybrid = sparse.hstack(
        [Xtr_base, sparse.csr_matrix(pair(ep_tr, eh_tr))], format="csr"
    )
    Xva_hybrid = sparse.hstack(
        [Xva_base, sparse.csr_matrix(pair(ep_va, eh_va))], format="csr"
    )
    hybrid = SelectKBest(f_classif, k=400).fit(Xtr_hybrid, ytr)
    results["rows"].append(
        _metrics(
            "hibrido tfidf+emb f_classif400",
            _rf(),
            hybrid.transform(Xtr_hybrid),
            hybrid.transform(Xva_hybrid),
            ytr,
            yva,
            english,
            encode_seconds,
        )
    )

    results_path = ROOT / f"{RESULTS_PATH_PREFIX}_{slug}.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nresultados: {results_path}")


if __name__ == "__main__":
    main()
