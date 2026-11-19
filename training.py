"""Entrenamiento y evaluación del Random Forest para NLI.

Este módulo contiene el flujo completo del modelo final de la competición
"Contradictory, My Dear Watson" (Kaggle): carga de datos, partición
train/validación, preprocesado cacheable, pipeline TF-IDF + POS + relación +
selección chi² + Random Forest, validación cruzada, evaluación holdout por
clase y por lengua, persistencia de resultados y generación de la submission.

Decisiones de diseño relevantes:

- Sin fugas de datos: en validación cruzada el pipeline se clona en cada fold
  (``sklearn.base.clone``), de modo que vocabulario, IDF, features auxiliares y
  selección chi² se ajustan únicamente con el bloque de entrenamiento de ese
  fold. En el holdout el ajuste se hace solo con ``train_df`` y la predicción
  con ``val_df``.
- Determinismo: todas las particiones reciben una semilla explícita y el
  ``random_state`` se propaga al vectorizador y al Random Forest.
- Tiempos: se miden con ``time.perf_counter`` y se cuantizan a 0.1 s (ver
  ``SECONDS_DECIMALS``) para que los informes sean estables.
- Partición vacía: scikit-learn lanza ``ValueError`` al calcular métricas
  sobre conjuntos vacíos, de modo que ``evaluate_holdout`` define
  explícitamente métricas nulas cuando ``val_df`` no tiene filas.
- Dependencia de ``sentence-transformers``: ``vectorizer.py`` la importa al
  cargarse, así que figura en ``requirements.txt`` (arrastra ``torch``).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from classifiers import CLASSIFIER_NAMES, build_classifier, is_naive_bayes
from pos_features import POSRatioFeatures
from preprocessing import PREPROCESSING_PARAMS, preprocess_partition
from relation_features import RelationFeatures
from vectorizer import NLITextVectorizer

# Etiquetas oficiales de la competición, por identificador entero.
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_IDS = tuple(LABEL_NAMES)  # (0, 1, 2)
VALID_PREDICTION_LABELS = frozenset(LABEL_IDS)

# Columnas del CSV original de entrenamiento, en su orden canónico.
TRAIN_COLUMNS = ("id", "premise", "hypothesis", "lang_abv", "language", "label")

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

# Decimales con los que se guardan los tiempos (0.1 s de resolución). La
# cuantización evita que informes de ejecuciones idénticas difieran por ruido
# del reloj; la medición subyacente siempre usa ``time.perf_counter``.
SECONDS_DECIMALS = 1

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

# Columnas de la tabla de métricas por lengua.
PER_LANGUAGE_COLUMNS = ("language", "n", "accuracy", "macro_f1", "n_classes")


def _quantize_seconds(seconds: float) -> float:
    """Redondea un tiempo medido a la resolución fijada en ``SECONDS_DECIMALS``."""
    return round(float(seconds), SECONDS_DECIMALS)


def _language_series(frame: pd.DataFrame) -> pd.Series | None:
    """Devuelve la columna de lengua disponible ('language' o 'lang_abv')."""
    if "language" in frame.columns:
        return frame["language"]
    if "lang_abv" in frame.columns:
        return frame["lang_abv"]
    return None


def _accuracy_and_macro_f1(y_true, y_pred) -> tuple[float, float]:
    """Accuracy y macro-F1 sobre las tres clases oficiales."""
    return (
        float(accuracy_score(y_true, y_pred)),
        float(
            f1_score(
                y_true,
                y_pred,
                labels=LABEL_IDS,
                average="macro",
                zero_division=0,
            )
        ),
    )


def load_train_data(path: str | Path = "data/train.csv") -> pd.DataFrame:
    """Carga el CSV de entrenamiento de la competición.

    Args:
        path: Ruta del CSV (por defecto ``data/train.csv``).

    Returns:
        DataFrame con las columnas ``id``, ``premise``, ``hypothesis``,
        ``lang_abv``, ``language`` y ``label`` (entero), en ese orden.

    Raises:
        FileNotFoundError: Si el fichero no existe.
        ValueError: Si falta alguna columna obligatoria o ``label`` tiene nulos.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"No existe el fichero de entrenamiento: {data_path.resolve()}"
        )

    df = pd.read_csv(data_path)

    missing = [column for column in TRAIN_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en {data_path}: {missing}")

    df = df.loc[:, list(TRAIN_COLUMNS)].copy()

    if df["label"].isna().any():
        raise ValueError("La columna 'label' contiene valores nulos.")
    df["label"] = df["label"].astype("int64")
    return df


def make_split(
    df: pd.DataFrame,
    val_size: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
    group_by_premise: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en entrenamiento y validación.

    Args:
        df: DataFrame de entrada (no se muta).
        val_size: Proporción de validación (entre 0 y 1).
        seed: Semilla que controla la aleatoriedad de la partición.
        stratify: Si es True, la partición simple estratifica por ``label``.
            Se ignora en modo agrupado, que ya es estratificado por naturaleza.
        group_by_premise: Si es True, ninguna premisa aparece en las dos
            particiones: se usa ``StratifiedGroupKFold`` con ``1 / val_size``
            bloques (5 por defecto) y el primer bloque queda como validación.

    Returns:
        Tupla ``(train_df, val_df)`` con índices reiniciados.

    Raises:
        ValueError: Si ``val_size`` no está en (0, 1).
    """
    if not 0.0 < float(val_size) < 1.0:
        raise ValueError(f"val_size debe estar en (0, 1), pero es {val_size}.")

    if group_by_premise:
        n_splits = max(2, min(10, int(round(1.0 / float(val_size)))))
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        train_idx, val_idx = next(
            iter(splitter.split(df, df["label"], groups=df["premise"]))
        )
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=seed,
            stratify=df["label"] if stratify else None,
        )

    return (
        train_df.copy().reset_index(drop=True),
        val_df.copy().reset_index(drop=True),
    )


def split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Resume tamaños, distribuciones y solape de premisas de una partición.

    Args:
        train_df: Partición de entrenamiento.
        val_df: Partición de validación.

    Returns:
        Diccionario con ``n_train``, ``n_val``, ``class_distribution_train``,
        ``class_distribution_val`` (diccionarios con las tres etiquetas
        enteras, contando 0 si no aparece), ``language_distribution_train``,
        ``language_distribution_val`` (lenguas presentes y sus recuentos),
        ``premise_overlap`` (número de premisas compartidas) y
        ``premise_overlap_ratio`` (proporción de solape respecto a la
        partición con menos premisas únicas).
    """

    def class_distribution(series: pd.Series) -> dict:
        counts = series.value_counts()
        return {label: int(counts.get(label, 0)) for label in LABEL_IDS}

    def language_distribution(series: pd.Series | None) -> dict:
        if series is None:
            return {}
        return {
            str(key): int(value) for key, value in series.value_counts().items()
        }

    premises_train = set(train_df["premise"])
    premises_val = set(val_df["premise"])
    overlap = len(premises_train & premises_val)
    smallest = min(len(premises_train), len(premises_val))
    ratio = overlap / smallest if smallest else 0.0

    return {
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "class_distribution_train": class_distribution(train_df["label"]),
        "class_distribution_val": class_distribution(val_df["label"]),
        "language_distribution_train": language_distribution(
            _language_series(train_df)
        ),
        "language_distribution_val": language_distribution(
            _language_series(val_df)
        ),
        "premise_overlap": int(overlap),
        "premise_overlap_ratio": float(ratio),
    }


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


def _zero_classification_report() -> dict:
    """Informe de clasificación nulo, para validaciones sin filas.

    scikit-learn rechaza métricas sobre entradas vacías, así que se define
    explícitamente el resultado: precision, recall y F1 nulos y soporte 0 en
    las tres clases, manteniendo el mismo esquema que ``output_dict=True``.
    """
    zero = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0}
    report = {LABEL_NAMES[label]: dict(zero) for label in LABEL_IDS}
    report["accuracy"] = 0.0
    report["macro avg"] = dict(zero)
    report["weighted avg"] = dict(zero)
    return report


def _empty_per_language_table() -> pd.DataFrame:
    """Tabla de métricas por lengua vacía pero con sus columnas completas."""
    return pd.DataFrame(columns=list(PER_LANGUAGE_COLUMNS))


def per_language_metrics(
    y_true,
    y_pred,
    languages,
    min_samples: int = 30,
) -> pd.DataFrame:
    """Métricas de clasificación agregadas por lengua.

    Args:
        y_true: Etiquetas verdaderas (0, 1, 2).
        y_pred: Etiquetas predichas.
        languages: Lengua de cada ejemplo.
        min_samples: Tamaño mínimo del grupo para incluirlo en la tabla.

    Returns:
        DataFrame con columnas ``language``, ``n``, ``accuracy``,
        ``macro_f1`` (macro-F1 sobre las tres clases con ``zero_division=0``)
        y ``n_classes`` (clases presentes entre real y predicho). Los grupos
        con menos de ``min_samples`` ejemplos se excluyen.
    """
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true).astype(int),
            "y_pred": np.asarray(y_pred).astype(int),
            "language": np.asarray(languages, dtype=object),
        }
    )

    rows = []
    for language, group in frame.groupby("language", sort=True):
        if len(group) < min_samples:
            continue
        y_true_group = group["y_true"].to_numpy()
        y_pred_group = group["y_pred"].to_numpy()
        accuracy, macro_f1 = _accuracy_and_macro_f1(y_true_group, y_pred_group)
        rows.append(
            {
                "language": language,
                "n": int(len(group)),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "n_classes": len(set(y_true_group) | set(y_pred_group)),
            }
        )

    return pd.DataFrame(rows, columns=list(PER_LANGUAGE_COLUMNS))


def evaluate_holdout(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """Ajusta el pipeline con ``train_df`` y lo evalúa sobre ``val_df``.

    El pipeline recibido se ajusta en el propio objeto (útil para inspeccionar
    después vocabulario, IDF o features) y solo con las filas de entrenamiento:
    la validación jamás participa en el ajuste.

    Args:
        pipeline: Pipeline sin ajustar de ``build_pipeline``.
        train_df: Partición de entrenamiento con columna ``label``.
        val_df: Partición de validación con columna ``label`` (puede estar
            vacía; en ese caso se devuelven métricas nulas).

    Returns:
        Diccionario con ``accuracy``, ``macro_f1``, ``classification_report``
        (``output_dict`` con las clases ``entailment``, ``neutral`` y
        ``contradiction``), ``confusion_matrix`` 3x3, ``per_language``
        (DataFrame de ``per_language_metrics`` con ``min_samples=30``),
        ``runtime_seconds``, ``n_train`` y ``n_val``.
    """
    started = time.perf_counter()
    pipeline.fit(train_df, train_df["label"])

    n_val = int(len(val_df))
    if n_val > 0:
        y_true = val_df["label"].to_numpy()
        predictions = pipeline.predict(val_df)
        accuracy, macro_f1 = _accuracy_and_macro_f1(y_true, predictions)
        report = classification_report(
            y_true,
            predictions,
            labels=LABEL_IDS,
            target_names=[LABEL_NAMES[label] for label in LABEL_IDS],
            output_dict=True,
            zero_division=0,
        )
        confusion = confusion_matrix(y_true, predictions, labels=LABEL_IDS)
        language_series = _language_series(val_df)
        if language_series is None:
            per_language = _empty_per_language_table()
        else:
            per_language = per_language_metrics(
                y_true, predictions, language_series.to_numpy()
            )
    else:
        accuracy = 0.0
        macro_f1 = 0.0
        report = _zero_classification_report()
        confusion = np.zeros((len(LABEL_IDS), len(LABEL_IDS)), dtype=int)
        per_language = _empty_per_language_table()

    runtime_seconds = _quantize_seconds(time.perf_counter() - started)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": confusion,
        "per_language": per_language,
        "runtime_seconds": runtime_seconds,
        "n_train": int(len(train_df)),
        "n_val": n_val,
    }


def save_results(rows, path: str | Path) -> Path:
    """Escribe o añade filas de resultados en un CSV, sin duplicar cabecera.

    Si el fichero ya existe, las filas nuevas se añaden tras las existentes y
    las columnas se unifican (uniendo las de ambas versiones); las celdas sin
    dato quedan vacías (NaN al leer). No se escriben cabeceras duplicadas.

    Args:
        rows: Secuencia de diccionarios con los resultados.
        path: Ruta del CSV de resultados.

    Returns:
        La ruta del fichero escrito.
    """
    frame = pd.DataFrame(list(rows))
    results_path = Path(path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    if results_path.exists():
        existing = pd.read_csv(results_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(results_path, index=False)
    return results_path


def _json_native(value):
    """Convierte tipos numpy/pandas a tipos nativos serializables a JSON.

    Los flotantes no finitos (NaN/inf) se convierten en ``None`` porque el
    JSON estándar no los admite.
    """
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, pd.Series):
        return _json_native(value.tolist())
    if isinstance(value, pd.DataFrame):
        return _json_native(value.to_dict(orient="records"))
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_native(item) for item in value]
    return str(value)


def save_json(obj, path: str | Path) -> Path:
    """Serializa ``obj`` a JSON aceptando tipos numpy y pandas.

    Args:
        obj: Objeto a serializar (diccionarios, listas, numpy, pandas...).
        path: Ruta del fichero JSON.

    Returns:
        La ruta del fichero escrito.
    """
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_native(obj), handle, ensure_ascii=False, indent=2)
    return json_path


def write_submission(ids, predictions, path: str | Path) -> pd.DataFrame:
    """Genera el CSV de submission de Kaggle con el esquema oficial.

    Args:
        ids: Identificadores de las filas de test.
        predictions: Etiquetas predichas; deben pertenecer a {0, 1, 2}.
        path: Ruta del CSV de salida.

    Returns:
        DataFrame con las columnas ``id`` y ``prediction``.

    Raises:
        ValueError: Si la longitud de ids y predicciones no coincide o si
            alguna predicción está fuera de {0, 1, 2}.
    """
    ids_list = list(ids)
    predictions_array = np.asarray(predictions)

    if len(ids_list) != len(predictions_array):
        raise ValueError(
            "Desajuste de longitudes en la submission: "
            f"{len(ids_list)} ids frente a {len(predictions_array)} predicciones."
        )

    values = predictions_array.tolist()
    invalid = [value for value in values if value not in VALID_PREDICTION_LABELS]
    if invalid:
        raise ValueError(
            "Las predicciones deben pertenecer a {0, 1, 2}; "
            f"valor no válido: {invalid[0]!r}"
        )

    submission = pd.DataFrame(
        {"id": ids_list, "prediction": [int(value) for value in values]}
    )
    submission_path = Path(path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    return submission
