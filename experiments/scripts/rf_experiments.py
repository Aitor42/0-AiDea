"""Banco de experimentos para mejorar el Random Forest (solo investigación).

Protocolo:
- Los datos de selección son SOLO el split de entrenamiento (9.696 pares).
- El holdout reservado no se toca aquí; se usa al final para confirmar.
- Cada configuración se evalúa con la misma CV estratificada de 5 folds.
- Se guardan las accuracy por fold para comparaciones pareadas.

Uso:
    python -m experiments.scripts.rf_experiments <experiment> [<experiment> ...]
    python -m experiments.scripts.rf_experiments --list
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.pipeline import FeatureUnion, Pipeline

from nli_project.preprocessing.preprocessing import (
    PREPROCESSING_PARAMS,
    preprocess_partition,
)
from nli_project.training.training import (
    RANDOM_FOREST_PARAMS,
    VECTORIZER_PARAMS,
    build_pipeline,
    cross_validate,
)
from nli_project.data.dataset import (
    load_train_data,
    make_split,
)

CACHE = ROOT / "experiments/.local/preprocessed"
HISTORY = ROOT / "experiments/results/random_forest/experiments.jsonl"
RESULTS = ROOT / "experiments/.local/random_forest/experiments.jsonl"

# ---------------------------------------------------------------- utilidades


def _prep(**overrides):
    params = dict(PREPROCESSING_PARAMS)
    params.update(overrides)
    return params


def _vec(**overrides):
    params = dict(VECTORIZER_PARAMS)
    params.update(overrides)
    return params


def _rf(**overrides):
    params = dict(RANDOM_FOREST_PARAMS)
    params.update(overrides)
    return params


class NegationFeatures(BaseEstimator, TransformerMixin):
    """Cuenta negaciones por columna y su discrepancia entre premisa e hipótesis.

    Concepto de teoría: la negación invierte la relación lógica, así que el
    número de negaciones en cada frase y si coinciden es una señal directa
    para entailment/contradiction que el TF-IDF no representa explícitamente.
    """

    NEGATIONS = frozenset(
        {
            "not", "no", "never", "none", "nobody", "nothing", "neither",
            "nor", "nowhere", "cannot", "n't", "without", "hardly", "barely",
        }
    )

    def __init__(self, columns=("premise_processed", "hypothesis_processed")):
        self.columns = tuple(columns)

    def fit(self, X, y=None):
        self.n_features_in_ = len(self.columns)
        return self

    @staticmethod
    def _tokens(value):
        if isinstance(value, str):
            return value.split()
        return [token[0] if isinstance(token, tuple) else token for token in value]

    def transform(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        cols = []
        for column in self.columns:
            counts = frame[column].map(
                lambda value: sum(
                    token.lower() in self.NEGATIONS
                    or token.lower().endswith("n't")
                    for token in self._tokens(value)
                )
            )
            cols.append(counts.to_numpy(dtype=np.float32))
        p, h = cols
        return np.column_stack(
            [p, h, np.abs(p - h), (p > 0).astype(np.float32), (h > 0).astype(np.float32)]
        )


class RelationFeatures(BaseEstimator, TransformerMixin):
    """Señales léxicas de la relación premisa-hipótesis.

    Conceptos: en NLI la contradicción suele venir de negaciones o de
    entidades/números incompatibles, y la neutralidad de una hipótesis que
    añade información no cubierta por la premisa. El TF-IDF normalizado pierde
    la longitud y el número de negaciones; aquí se recuperan de forma explícita
    y barata (todo se calcula fila a fila, sin aprender del corpus).
    """

    NEGATIONS = NegationFeatures.NEGATIONS

    def __init__(self, columns=("premise_processed", "hypothesis_processed")):
        self.columns = tuple(columns)

    def fit(self, X, y=None):
        self.n_features_in_ = len(self.columns)
        return self

    @staticmethod
    def _tokens(value):
        if isinstance(value, str):
            return value.split()
        return [token[0] if isinstance(token, tuple) else token for token in value]

    @staticmethod
    def _numbers(tokens):
        found = set()
        for token in tokens:
            if any(char.isdigit() for char in token):
                found.add(token)
        return found

    def transform(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        p_tokens = [self._tokens(value) for value in frame[self.columns[0]]]
        h_tokens = [self._tokens(value) for value in frame[self.columns[1]]]

        rows = []
        for p_tok, h_tok in zip(p_tokens, h_tokens):
            p_neg = sum(t.lower() in self.NEGATIONS or t.lower().endswith("n't") for t in p_tok)
            h_neg = sum(t.lower() in self.NEGATIONS or t.lower().endswith("n't") for t in h_tok)
            p_num, h_num = self._numbers(p_tok), self._numbers(h_tok)
            shared_num = p_num & h_num
            union_num = p_num | h_num
            rows.append(
                [
                    p_neg,
                    h_neg,
                    abs(p_neg - h_neg),
                    float((p_neg > 0) != (h_neg > 0)),
                    len(p_tok),
                    len(h_tok),
                    abs(len(h_tok) - len(p_tok)),
                    len(h_tok) / (len(p_tok) + 1.0),
                    float(len(h_tok) > len(p_tok)),
                    float(len(p_tok) > len(h_tok)),
                    len(p_num),
                    len(h_num),
                    abs(len(p_num) - len(h_num)),
                    len(shared_num) / len(union_num) if union_num else 0.0,
                ]
            )
        return np.asarray(rows, dtype=np.float32)


class LengthFeatures(BaseEstimator, TransformerMixin):
    """Longitudes de premisa e hipótesis y su relación (fila a fila)."""

    def __init__(self, columns=("premise_processed", "hypothesis_processed")):
        self.columns = tuple(columns)

    def fit(self, X, y=None):
        self.n_features_in_ = len(self.columns)
        return self

    @staticmethod
    def _tokens(value):
        if isinstance(value, str):
            return value.split()
        return [token[0] if isinstance(token, tuple) else token for token in value]

    def transform(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        p = [len(self._tokens(v)) for v in frame[self.columns[0]]]
        h = [len(self._tokens(v)) for v in frame[self.columns[1]]]
        return np.asarray(
            [
                [len_p, len_h, abs(len_h - len_p), len_h / (len_p + 1.0),
                 float(len_h > len_p), float(len_p > len_h)]
                for len_p, len_h in zip(p, h)
            ],
            dtype=np.float32,
        )


class NumberFeatures(BaseEstimator, TransformerMixin):
    """Números/cifras en cada frase y su solape (pista de contradicción)."""

    def __init__(self, columns=("premise_processed", "hypothesis_processed")):
        self.columns = tuple(columns)

    def fit(self, X, y=None):
        self.n_features_in_ = len(self.columns)
        return self

    @staticmethod
    def _numbers(value):
        tokens = value.split() if isinstance(value, str) else [
            token[0] if isinstance(token, tuple) else token for token in value
        ]
        return {token for token in tokens if any(char.isdigit() for char in token)}

    def transform(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        p = [self._numbers(v) for v in frame[self.columns[0]]]
        h = [self._numbers(v) for v in frame[self.columns[1]]]
        rows = []
        for p_num, h_num in zip(p, h):
            union = p_num | h_num
            rows.append(
                [
                    len(p_num),
                    len(h_num),
                    abs(len(p_num) - len(h_num)),
                    len(p_num & h_num) / len(union) if union else 0.0,
                ]
            )
        return np.asarray(rows, dtype=np.float32)


class LanguageFeatures(BaseEstimator, TransformerMixin):
    """One-hot de la lengua declarada (15 idiomas) como feature del bosque."""

    def __init__(self, column="lang_abv"):
        self.column = column

    def fit(self, X, y=None):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # 15 lenguas oficiales del dataset + salvaguarda para categorías nuevas.
        self.categories_ = sorted(frame[self.column].astype(str).unique())
        return self

    def transform(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        values = frame[self.column].astype(str)
        return np.column_stack(
            [
                (values == category).to_numpy(dtype=np.float32)
                for category in self.categories_
            ]
        )


def _build(
    vec_params=None,
    rf_params=None,
    use_pos_features=True,
    extra=None,
    select_k=None,
    select_score="chi2",
):
    """Pipeline features -> [selección] -> RF, con bloques extra opcionales."""
    pipeline = build_pipeline(
        vec_params,
        classifier_params=rf_params,
        use_pos_features=use_pos_features,
        random_state=42,
    )
    features = pipeline.named_steps["features"]
    if extra:
        features = FeatureUnion(
            list(features.transformer_list) + extra if isinstance(features, FeatureUnion)
            else [("text", features)] + extra
        )
    steps = [("features", features)]
    if select_k is not None:
        score_func = chi2 if select_score == "chi2" else f_classif
        steps.append(("select", SelectKBest(score_func, k=select_k)))
    steps.append(("classifier", pipeline.named_steps["classifier"]))
    return Pipeline(steps)


def run(name, *, prep=None, vec=None, rf=None, use_pos_features=True, extra=None,
        select_k=None, select_score="chi2", n_splits=5, seed=42, note=""):
    """Ejecuta una CV y devuelve/guarda el resultado."""
    prep_params = _prep(**(prep or {}))
    vec_params = _vec(**(vec or {}))
    rf_params = _rf(**(rf or {}))

    started = time.perf_counter()
    processed = preprocess_partition(train, prep_params, cache_dir=CACHE)
    prep_seconds = time.perf_counter() - started

    pipeline = _build(
        vec_params,
        rf_params,
        use_pos_features=use_pos_features,
        extra=extra,
        select_k=select_k,
        select_score=select_score,
    )

    started = time.perf_counter()
    cv = cross_validate(processed, pipeline, n_splits=n_splits, seed=seed)
    cv_seconds = time.perf_counter() - started

    result = {
        "name": name,
        "note": note,
        "preprocessing": {k: prep_params[k] for k in (
            "reduction", "stopword_method", "keep_negations", "tokenizer",
            "strip_accents", "lowercase", "use_pos")},
        "vectorizer": vec_params,
        "random_forest": rf_params,
        "use_pos_features": use_pos_features,
        "select_k": select_k,
        "select_score": select_score,
        "extra_features": [n for n, _ in (extra or [])],
        "n_splits": n_splits,
        "accuracy_mean": cv["summary"]["accuracy_mean"],
        "accuracy_std": cv["summary"]["accuracy_std"],
        "macro_f1_mean": cv["summary"]["macro_f1_mean"],
        "macro_f1_std": cv["summary"]["macro_f1_std"],
        "fold_accuracy": [float(v) for v in cv["folds"]["accuracy"]],
        "fold_macro_f1": [float(v) for v in cv["folds"]["macro_f1"]],
        "fit_seconds_mean": cv["summary"]["fit_seconds_mean"],
        "preprocess_seconds": round(prep_seconds, 1),
        "cv_seconds": round(cv_seconds, 1),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    print(
        f"{name:28s} acc={result['accuracy_mean']:.6f} ± {result['accuracy_std']:.4f} "
        f"f1={result['macro_f1_mean']:.6f} fits={result['fit_seconds_mean']:.1f}s "
        f"cv={result['cv_seconds']:.0f}s",
        flush=True,
    )
    del processed
    gc.collect()
    return result


def load_results():
    if not HISTORY.exists() and not RESULTS.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for path in (HISTORY, RESULTS) if path.exists()
            for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ registro

EXPERIMENTS = {
    # --- A. representación (preprocesado y modelo fijos) ---
    "base": dict(note="configuración actual"),
    "mindf_5": dict(vec={"min_df": 5}, note="vocabulario más amplio"),
    "mindf_20": dict(vec={"min_df": 20}),
    "mindf_100": dict(vec={"min_df": 100}),
    "ngram_11": dict(vec={"word_ngram_range": (1, 1)}, note="sin bigramas"),
    "ngram_13": dict(vec={"word_ngram_range": (1, 3)}, note="hasta trigramas"),
    "pair_separate": dict(vec={"method": "pair_separate"}, note="sin diff ni solape"),
    "count": dict(vec={"method": "count"}),
    "tfidf": dict(vec={"method": "tfidf"}),
    "word_char": dict(vec={"method": "word_char"}, note="palabra + carácter"),
    "tfidf_svd": dict(vec={"method": "tfidf_svd", "svd_components": 300}),
    "no_pos_features": dict(use_pos_features=False, note="sin ratios POS"),
    "select_500": dict(select_k=500),
    "select_1000": dict(select_k=1000),
    "select_200": dict(select_k=200),
    "select_300": dict(select_k=300),
    "select_750": dict(select_k=750),
    "sel500_uni": dict(select_k=500, vec={"word_ngram_range": (1, 1)}),
    "sel300_uni": dict(select_k=300, vec={"word_ngram_range": (1, 1)}),
    "sel500_tri": dict(select_k=500, vec={"word_ngram_range": (1, 3)}),
    "sel500_posoff": dict(select_k=500, use_pos_features=False),
    "sel500_leaf1": dict(select_k=500, rf={"min_samples_leaf": 1}),
    "sel500_leaf10": dict(select_k=500, rf={"min_samples_leaf": 10}),
    "sel500_depthnone": dict(select_k=500, rf={"max_depth": None}),
    "sel500_entropy": dict(select_k=500, rf={"criterion": "entropy"}),
    "sel500_maxfeat033": dict(select_k=500, rf={"max_features": 0.33}),
    "sel500_balanced": dict(select_k=500, rf={"class_weight": "balanced"}),
    "sel500_uni_leaf1": dict(
        select_k=500, vec={"word_ngram_range": (1, 1)}, rf={"min_samples_leaf": 1}
    ),
    # --- C2. preprocesado con la mejor configuración de features ---
    "sel500_porter": dict(select_k=500, prep={"reduction": "porter"}),
    "sel500_snowball": dict(select_k=500, prep={"reduction": "snowball"}),
    "sel500_wordnet": dict(select_k=500, prep={"reduction": "wordnet"}),
    "sel500_no_reduction": dict(select_k=500, prep={"reduction": "none"}),
    "sel500_stopwords": dict(select_k=500, prep={"stopword_method": "sklearn"}),
    "sel500_strip_accents": dict(select_k=500, prep={"strip_accents": True}),
    "sel500_tokenizer_spacy": dict(select_k=500, prep={"tokenizer": "spacy"}),
    "sel500_negations": dict(
        select_k=500, extra=[("negations", NegationFeatures())]
    ),
    "sel500_language": dict(
        select_k=500, extra=[("language", LanguageFeatures())]
    ),
    "sel500_neg_lang": dict(
        select_k=500,
        extra=[("negations", NegationFeatures()), ("language", LanguageFeatures())],
    ),
    # --- D2. sobre el mejor combo (unigramas + k≈300) ---
    "sel400_uni": dict(select_k=400, vec={"word_ngram_range": (1, 1)}),
    "sel300_uni_f": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, select_score="f_classif"
    ),
    "sel500_f": dict(select_k=500, select_score="f_classif"),
    "sel300_uni_negations": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)},
        extra=[("negations", NegationFeatures())],
    ),
    "sel300_uni_language": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)},
        extra=[("language", LanguageFeatures())],
    ),
    "sel300_uni_neg_lang": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)},
        extra=[("negations", NegationFeatures()), ("language", LanguageFeatures())],
    ),
    "sel300_uni_porter": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"reduction": "porter"}
    ),
    "sel300_uni_snowball": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"reduction": "snowball"}
    ),
    "sel300_uni_wordnet": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"reduction": "wordnet"}
    ),
    "sel300_uni_noredux": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"reduction": "none"}
    ),
    "sel300_uni_stopwords": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)},
        prep={"stopword_method": "sklearn"},
    ),
    "sel300_uni_accents": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"strip_accents": True}
    ),
    "sel300_uni_spacy_tok": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"}
    ),
    # --- E. interacción entre tamaño de vocabulario y selección ---
    "sel300_uni_mindf5": dict(
        select_k=300, vec={"word_ngram_range": (1, 1), "min_df": 5}
    ),
    "sel300_uni_mindf20": dict(
        select_k=300, vec={"word_ngram_range": (1, 1), "min_df": 20}
    ),
    "sel300_uni_nosublinear": dict(
        select_k=300, vec={"word_ngram_range": (1, 1), "sublinear_tf": False}
    ),
    "sel250_uni": dict(select_k=250, vec={"word_ngram_range": (1, 1)}),
    # --- F. ablaciones de las features de relación y combinaciones ---
    "negations_noselect": dict(extra=[("negations", NegationFeatures())]),
    "relations_noselect": dict(extra=[("relations", RelationFeatures())]),
    "sel400_uni_negations": dict(
        select_k=400, vec={"word_ngram_range": (1, 1)},
        extra=[("negations", NegationFeatures())],
    ),
    "sel500_uni_negations": dict(
        select_k=500, vec={"word_ngram_range": (1, 1)},
        extra=[("negations", NegationFeatures())],
    ),
    "sel300_uni_relations": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)},
        extra=[("relations", RelationFeatures())],
    ),
    "sel400_uni_relations": dict(
        select_k=400, vec={"word_ngram_range": (1, 1)},
        extra=[("relations", RelationFeatures())],
    ),
    "sel300_uni_neg_relations": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)},
        extra=[("negations", NegationFeatures()), ("relations", RelationFeatures())],
    ),
    # --- G. tokenización spaCy y combinaciones ---
    "sel300_uni_spacy": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"}
    ),
    "sel400_uni_spacy": dict(
        select_k=400, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"}
    ),
    "sel500_uni_spacy": dict(
        select_k=500, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"}
    ),
    "spacy_noselect": dict(prep={"tokenizer": "spacy"}),
    "sel300_uni_spacy_neg": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"},
        extra=[("negations", NegationFeatures())],
    ),
    "sel300_uni_spacy_rel": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"},
        extra=[("relations", RelationFeatures())],
    ),
    "sel250_uni_spacy_rel": dict(
        select_k=250, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"},
        extra=[("relations", RelationFeatures())],
    ),
    "sel400_uni_spacy_rel": dict(
        select_k=400, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"},
        extra=[("relations", RelationFeatures())],
    ),
    "sel500_uni_spacy_rel": dict(
        select_k=500, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"},
        extra=[("relations", RelationFeatures())],
    ),
    # --- H. atribución dentro de las features de relación ---
    "sel300_uni_spacy_len": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"},
        extra=[("lengths", LengthFeatures())],
    ),
    "sel300_uni_spacy_num": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"},
        extra=[("numbers", NumberFeatures())],
    ),
    "sel300_uni_spacy_neg_len": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"},
        extra=[("negations", NegationFeatures()), ("lengths", LengthFeatures())],
    ),
    "sel300_uni_spacy_rel_mindf5": dict(
        select_k=300, vec={"word_ngram_range": (1, 1), "min_df": 5},
        prep={"tokenizer": "spacy"}, extra=[("relations", RelationFeatures())],
    ),
    "sel300_uni_spacy_rel_mindf20": dict(
        select_k=300, vec={"word_ngram_range": (1, 1), "min_df": 20},
        prep={"tokenizer": "spacy"}, extra=[("relations", RelationFeatures())],
    ),
    "sel200_uni_spacy_rel_mindf5": dict(
        select_k=200, vec={"word_ngram_range": (1, 1), "min_df": 5},
        prep={"tokenizer": "spacy"}, extra=[("relations", RelationFeatures())],
    ),
    "sel300_uni_spacy_rel_mindf5_posoff": dict(
        select_k=300, vec={"word_ngram_range": (1, 1), "min_df": 5},
        prep={"tokenizer": "spacy"}, extra=[("relations", RelationFeatures())],
        use_pos_features=False,
    ),
    "sel300_uni_spacy_rel_mindf5_noposvec": dict(
        select_k=300,
        vec={"word_ngram_range": (1, 1), "min_df": 5, "use_pos": False},
        prep={"tokenizer": "spacy"}, extra=[("relations", RelationFeatures())],
    ),
    "sel300_uni_spacy_neg_rel": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)}, prep={"tokenizer": "spacy"},
        extra=[("negations", NegationFeatures()), ("relations", RelationFeatures())],
    ),
    "sel300_uni_spacy_mindf20": dict(
        select_k=300, vec={"word_ngram_range": (1, 1), "min_df": 20},
        prep={"tokenizer": "spacy"},
    ),
    "sel300_spacy_ngram12": dict(
        select_k=300, prep={"tokenizer": "spacy"}
    ),
    "sel300_uni_spacy_noredux": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)},
        prep={"tokenizer": "spacy", "reduction": "none"},
    ),
    "sel300_uni_spacy_stopwords": dict(
        select_k=300, vec={"word_ngram_range": (1, 1)},
        prep={"tokenizer": "spacy", "stopword_method": "sklearn"},
    ),
    "sel300_uni_spacy_mindf5": dict(
        select_k=300, vec={"word_ngram_range": (1, 1), "min_df": 5},
        prep={"tokenizer": "spacy"},
    ),
    # --- B. hiperparámetros del bosque ---
    "leaf_1": dict(rf={"min_samples_leaf": 1}),
    "leaf_10": dict(rf={"min_samples_leaf": 10}),
    "depth_none": dict(rf={"max_depth": None}),
    "split_2": dict(rf={"min_samples_split": 2}),
    "entropy": dict(rf={"criterion": "entropy"}),
    "maxfeat_0.33": dict(rf={"max_features": 0.33}),
    "maxfeat_log2": dict(rf={"max_features": "log2"}),
    "balanced": dict(rf={"class_weight": "balanced"}),
    "trees_500": dict(rf={"n_estimators": 500}),
    # --- C. preprocesado ---
    "porter": dict(prep={"reduction": "porter"}, note="stemming en vez de lemma"),
    "snowball": dict(prep={"reduction": "snowball"}),
    "no_reduction": dict(prep={"reduction": "none"}),
    "stopwords_sklearn": dict(prep={"stopword_method": "sklearn"}, note="conservando negaciones"),
    "strip_accents": dict(prep={"strip_accents": True}),
    "tokenizer_spacy": dict(prep={"tokenizer": "spacy"}, note="tokenización real de spaCy"),
    # --- D. ingeniería de features ---
    "negation_features": dict(extra=[("negations", NegationFeatures())]),
    "language_onehot": dict(extra=[("language", LanguageFeatures())]),
}


def main(argv):
    if "--list" in argv:
        for name, spec in EXPERIMENTS.items():
            print(f"{name:28s} {spec.get('note', '')}")
        return
    names = [arg for arg in argv if not arg.startswith("--")]
    if not names:
        names = list(EXPERIMENTS)
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"experimento desconocido: {name}")
        print(f"== {name} ==", flush=True)
        run(name, **EXPERIMENTS[name])


if __name__ == "__main__":
    import sys

    train, val = None, None
    df = load_train_data(ROOT / "experiments/data/train.csv")
    train, val = make_split(df, val_size=0.2, seed=42)
    del df
    main(sys.argv[1:])
