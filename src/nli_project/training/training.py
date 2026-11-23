"""Construcción del pipeline NLI y validación cruzada."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from nli_project.data.dataset import LABEL_IDS
from nli_project.evaluation.metrics import _accuracy_and_macro_f1, _quantize_seconds
from nli_project.models.classifiers import CLASSIFIER_NAMES, build_classifier, is_naive_bayes
from nli_project.features.pos_features import POSRatioFeatures
from nli_project.features.relation_features import RelationFeatures
from nli_project.features.vectorizer import NLITextVectorizer

# Columnas por defecto del vectorizador cuando los datos ya están preprocesados.
DEFAULT_VECTORIZER_COLUMNS = ("premise_processed", "hypothesis_processed")

# Representación final: TF-IDF de unigramas de premisa, hipótesis y su
# diferencia absoluta, más las features de solape léxico de ``pair_diff``.
# ``use_pos=False`` evita fragmentar el vocabulario con el sufijo ``_POS``
# (los tags siguen disponibles para ``POSRatioFeatures``).
VECTORIZER_PARAMS = {
    "method": "pair_diff",
    "columns": DEFAULT_VECTORIZER_COLUMNS,
    "min_df": 5,
    "word_ngram_range": (1, 1),
    "use_pos": False,
}

# Número de características que conserva ``SelectKBest(chi2)``. Con
# ``min_df=5`` el vocabulario ronda las 20.000 features; k=300 elimina el
# ruido y es el óptimo medido en validación cruzada. ``None`` desactiva la
# selección (ablaciones y representaciones con signo, que chi² no admite).
SELECT_K_FEATURES = 300

# Hiperparámetros del Random Forest seleccionados en la rejilla reproducible.
RANDOM_FOREST_PARAMS = {
    "n_estimators": 1000,
    "max_depth": 50,
    "min_samples_leaf": 5,
    "min_samples_split": 5,
    "max_features": "sqrt",
    "criterion": "gini",
}

# Métodos del vectorizador que producen componentes con signo. ``chi2`` exige
# entradas no negativas y la familia Naive Bayes también, así que ambos se
# rechazan al construir el pipeline en vez de fallar en ``fit``.
NEGATIVE_FEATURE_METHODS = frozenset({"tfidf_svd", "embeddings"})

# Columnas de la tabla de folds de validación cruzada.
FOLD_COLUMNS = (
    "fold",
    "n_train",
    "n_test",
    "accuracy",
    "macro_f1",
    "fit_seconds",
    "predict_seconds",
)


def build_pipeline(
    vectorizer_params: dict | None = None,
    classifier_params: dict | None = None,
    use_pos_features: bool = True,
    use_relation_features: bool = True,
    select_k: int | None = SELECT_K_FEATURES,
    random_state: int = 42,
    classifier: str = "random_forest",
) -> Pipeline:
    """Construye el pipeline ``features -> selección -> clasificador``.

    El paso ``features`` es un ``NLITextVectorizer`` y, según los indicadores
    de features auxiliares, un ``FeatureUnion`` que concatena el texto
    vectorizado con los ratios POS de ``POSRatioFeatures`` (columnas
    ``*_processed`` de tuplas ``(token, POS)``, es decir, preprocesado con
    ``use_pos=True``) y/o las señales de relación de ``RelationFeatures``.
    Las features auxiliares derivan sus columnas de las del vectorizador: con
    una sola columna (solo premisas) emiten las señales de esa frase, mientras
    que ``pair_diff`` —definido como ``|Xp - Xh|``— requiere las dos.

    Cuando ``select_k`` no es ``None`` se añade un paso ``select`` con
    ``SelectKBest(chi2)``. Al vivir dentro del ``Pipeline``, la selección se
    ajusta en cada fold con solo el bloque de entrenamiento: no hay fuga de
    datos. chi² exige entradas no negativas, así que con representaciones con
    signo (``tfidf_svd``, ``embeddings``) hay que pasar ``select_k=None``.

    El clasificador se construye con ``classifiers.build_classifier``: el
    nombre corto ``classifier`` selecciona el algoritmo y ``classifier_params``
    aporta sus hiperparámetros. Por defecto se mantiene el Random Forest de la
    configuración final (``RANDOM_FOREST_PARAMS``), de modo que las llamadas
    existentes conservan su comportamiento.

    Args:
        vectorizer_params: Argumentos del constructor de
            ``NLITextVectorizer``; ``None`` usa ``VECTORIZER_PARAMS``. El
            diccionario de entrada no se muta; si no incluye ``random_state``,
            se fija con la semilla de este argumento.
        classifier_params: Hiperparámetros del clasificador; ``None`` usa los
            valores por defecto de la fábrica (para el Random Forest, los de
            ``RANDOM_FOREST_PARAMS``). Tienen precedencia sobre esos defaults;
            la semilla global se aplica salvo que el propio diccionario fije
            una (en Random Forest, la semilla de este argumento gana).
        use_pos_features: Si es True añade los ratios POS al paso
            ``features`` mediante un ``FeatureUnion``.
        use_relation_features: Si es True añade las features de relación
            (negaciones, longitudes y números) al paso ``features``.
        select_k: Número de features que conserva ``SelectKBest(chi2)``;
            ``None`` desactiva la selección.
        random_state: Semilla del vectorizador y de los clasificadores
            estocásticos.
        classifier: Nombre corto del clasificador; debe pertenecer a
            ``CLASSIFIER_NAMES``. Por defecto ``"random_forest"``.

    Returns:
        ``Pipeline`` con los pasos ``"features"``, ``"select"`` (si
        ``select_k`` no es ``None``) y ``"classifier"``.

    Raises:
        ValueError: Si el clasificador no pertenece a ``CLASSIFIER_NAMES``, si
            el método de vectorización no pertenece a
            ``NLITextVectorizer.VALID_METHODS``, si ``select_k`` se combina con
            un método con signo, si la familia Naive Bayes recibe una
            representación con signo o si ``select_k`` no es ``None`` ni un
            entero positivo.
    """
    if classifier not in CLASSIFIER_NAMES:
        raise ValueError(
            f"Clasificador no válido: {classifier!r}. "
            f"Valores válidos: {list(CLASSIFIER_NAMES)}"
        )

    params = dict(VECTORIZER_PARAMS if vectorizer_params is None else vectorizer_params)
    # ``final_config.json`` serializa las tuplas como listas; se normalizan
    # aquí para que la configuración guardada sea directamente reutilizable.
    for key in ("columns", "word_ngram_range", "char_ngram_range"):
        if isinstance(params.get(key), list):
            params[key] = tuple(params[key])
    params.setdefault("columns", DEFAULT_VECTORIZER_COLUMNS)
    params.setdefault("random_state", random_state)

    method = params.get("method", "tfidf")
    if method not in NLITextVectorizer.VALID_METHODS:
        raise ValueError(
            f"Método de vectorización no válido: {method!r}. "
            f"Valores válidos: {sorted(NLITextVectorizer.VALID_METHODS)}"
        )

    if select_k is not None:
        if method in NEGATIVE_FEATURE_METHODS:
            raise ValueError(
                "SelectKBest(chi2) exige características no negativas; el "
                f"método {method!r} produce componentes con signo. Usa "
                "conteos/TF-IDF o pasa select_k=None."
            )
        if isinstance(select_k, bool) or not isinstance(select_k, int):
            raise ValueError(
                "select_k debe ser un entero positivo o None; "
                f"recibido {select_k!r}"
            )
        if select_k < 1:
            raise ValueError(f"select_k debe ser >= 1 o None; recibido {select_k}")

    if is_naive_bayes(classifier) and method in NEGATIVE_FEATURE_METHODS:
        raise ValueError(
            "La familia Naive Bayes requiere características no negativas; "
            f"el método {method!r} produce componentes con signo. Usa "
            "conteos/TF-IDF."
        )

    auxiliary_features = []
    if use_pos_features:
        auxiliary_features.append(
            ("pos_ratios", POSRatioFeatures(columns=tuple(params["columns"])))
        )
    if use_relation_features:
        auxiliary_features.append(
            ("relation", RelationFeatures(columns=tuple(params["columns"])))
        )

    if auxiliary_features:
        features = FeatureUnion(
            [
                ("text", NLITextVectorizer(**params)),
                *auxiliary_features,
            ]
        )
    else:
        features = NLITextVectorizer(**params)

    # El Random Forest conserva la configuración final medida (1000 árboles,
    # profundidad 50...) y su ``n_jobs=-1``; el resto de algoritmos usa los
    # defaults de la fábrica salvo que ``classifier_params`` diga otra cosa.
    if classifier == "random_forest":
        estimator_params = {
            "n_jobs": -1,
            **(RANDOM_FOREST_PARAMS if classifier_params is None else classifier_params),
        }
        # La semilla de este argumento gana sobre ``classifier_params``.
        estimator_params.pop("random_state", None)
    else:
        estimator_params = dict(classifier_params or {})

    estimator = build_classifier(
        classifier, random_state=random_state, **estimator_params
    )

    steps = [("features", features)]
    if select_k is not None:
        steps.append(("select", SelectKBest(score_func=chi2, k=select_k)))
    steps.append(("classifier", estimator))
    return Pipeline(steps)



def cross_validate(
    df: pd.DataFrame,
    pipeline: Pipeline,
    n_splits: int = 5,
    seed: int = 42,
    group_by_premise: bool = False,
    return_fold_frames: bool = False,
) -> dict:
    """Validación cruzada estratificada (y opcionalmente agrupada por premisa).

    En cada fold se entrena un clon del pipeline con solo el bloque de
    entrenamiento de ese fold, de forma que vocabulario, IDF, features
    auxiliares y selección chi² jamás ven filas de test (sin fugas de datos).
    Los tiempos de ajuste y predicción se miden con ``time.perf_counter``.

    Args:
        df: DataFrame con la columna ``label`` (no se muta).
        pipeline: Pipeline (normalmente de ``build_pipeline``) sin ajustar;
            no se modifica: cada fold usa un clon.
        n_splits: Número de folds.
        seed: Semilla del reparto de folds.
        group_by_premise: Si es True usa ``StratifiedGroupKFold`` con
            ``groups=df["premise"]``: ninguna premisa pasa de entrenamiento a
            test dentro de un mismo fold.
        return_fold_frames: Si es True añade la clave ``fold_frames`` con la
            lista de tuplas ``(train_df, test_df)`` de cada fold (conservan el
            índice original).

    Returns:
        Diccionario con:
            - ``folds``: DataFrame con una fila por fold y las columnas
              ``fold``, ``n_train``, ``n_test``, ``accuracy``, ``macro_f1``,
              ``fit_seconds`` y ``predict_seconds``.
            - ``summary``: métricas agregadas (medias y desviaciones con
              ``ddof=0`` más el tiempo total de ajuste).
            - ``confusion_matrix``: matriz 3x3 acumulada de todos los folds.
            - ``fold_frames``: solo si ``return_fold_frames=True``.
    """
    y = df["label"]

    if group_by_premise:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        splits = splitter.split(df, y, groups=df["premise"])
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        splits = splitter.split(df, y)

    fold_rows: list[dict] = []
    fold_frames: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    confusion = np.zeros((len(LABEL_IDS), len(LABEL_IDS)), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
        fold_train = df.iloc[train_idx]
        fold_test = df.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        estimator = clone(pipeline)

        started = time.perf_counter()
        estimator.fit(fold_train, y_train)
        fit_seconds = _quantize_seconds(time.perf_counter() - started)

        started = time.perf_counter()
        predictions = estimator.predict(fold_test)
        predict_seconds = _quantize_seconds(time.perf_counter() - started)
        accuracy, macro_f1 = _accuracy_and_macro_f1(y_test, predictions)

        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(fold_train)),
                "n_test": int(len(fold_test)),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "fit_seconds": fit_seconds,
                "predict_seconds": predict_seconds,
            }
        )
        confusion += confusion_matrix(y_test, predictions, labels=LABEL_IDS)

        if return_fold_frames:
            fold_frames.append((fold_train, fold_test))

    folds = pd.DataFrame(fold_rows, columns=list(FOLD_COLUMNS))
    summary = {
        "n_splits": int(len(folds)),
        "accuracy_mean": float(folds["accuracy"].mean()),
        "accuracy_std": float(folds["accuracy"].std(ddof=0)),
        "macro_f1_mean": float(folds["macro_f1"].mean()),
        "macro_f1_std": float(folds["macro_f1"].std(ddof=0)),
        "fit_seconds_mean": float(folds["fit_seconds"].mean()),
        "predict_seconds_mean": float(folds["predict_seconds"].mean()),
        "fit_seconds_total": float(folds["fit_seconds"].sum()),
    }

    result = {"folds": folds, "summary": summary, "confusion_matrix": confusion}
    if return_fold_frames:
        result["fold_frames"] = fold_frames
    return result
