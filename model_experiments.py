"""Maquinaria común de los bancos de experimentos por clasificador.

Este módulo extrae de ``rf_experiments.py`` el protocolo y las utilidades que
comparten todos los modelos del catálogo de ``classifiers.py``, de modo que
cada algoritmo tenga su propio banco sin repetir la infraestructura:

- ``naive_bayes_experiments.py``
- ``complement_nb_experiments.py``
- ``bernoulli_nb_experiments.py``
- ``decision_tree_experiments.py``
- ``knn_experiments.py``
- ``mlp_experiments.py``
- ``xgboost_experiments.py``
- ``catboost_experiments.py``
- ``lightgbm_experiments.py``

El Random Forest conserva su banco histórico ``rf_experiments.py``.

Protocolo (idéntico al del Random Forest):

- Los datos de selección son SOLO el split de entrenamiento (9.696 pares).
- El holdout reservado no se toca aquí; se usa al final para confirmar.
- Cada configuración se evalúa con la misma CV estratificada de 5 folds y la
  misma semilla, así que las comparaciones entre configuraciones son pareadas
  fold a fold.
- Se guarda una línea JSON por ejecución en
  ``<modelo>_experiments_results.jsonl`` con las accuracy por fold, el tiempo
  de ajuste y la configuración completa.
- El preprocesado es fila a fila y se reutiliza desde
  ``0-AiDea/artifacts/preprocessed`` por hash de configuración y datos, así que
  cambiar solo el clasificador no vuelve a etiquetar el corpus.

Uso (desde la raíz del proyecto):

    python knn_experiments.py --list
    python knn_experiments.py k_25 k_50
    python knn_experiments.py            # todos los experimentos del banco
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "0-AiDea"
for _path in (ROOT, PACKAGE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd

from experiment_defaults import CLASSIFIER_SPECS

from training import (
    PREPROCESSING_PARAMS,
    VECTORIZER_PARAMS,
    build_pipeline,
    cross_validate,
    load_train_data,
    make_split,
    preprocess_partition,
)

# Caché de preprocesado compartida con el banco del Random Forest.
CACHE = PACKAGE / "artifacts/preprocessed"

# Random Forest conserva su banco histórico.
BASELINE_SPECS = {
    model: params for model, params in CLASSIFIER_SPECS.items() if model != "random_forest"
}

# Ejes de representación comunes a todos los modelos. Cada banco los combina
# con los hiperparámetros propios del clasificador; cambiar el clasificador es
# la única variable cuando se ejecuta la fila ``base``. Todos los métodos de
# vectorización elegidos producen entradas no negativas, requisito de chi² y
# de la familia Naive Bayes.
REPRESENTATION_EXPERIMENTS = {
    "base": dict(note="configuración final del proyecto con este clasificador"),
    "select_200": dict(select_k=200),
    "select_500": dict(select_k=500),
    "select_1000": dict(select_k=1000),
    "no_select": dict(select_k=None, note="sin selección chi²"),
    "mindf_20": dict(vec={"min_df": 20}, note="vocabulario más corto"),
    "ngram_12": dict(vec={"word_ngram_range": (1, 2)}, note="bigramas"),
    "count": dict(vec={"method": "count"}, note="conteos en vez de TF-IDF"),
    "pair_separate": dict(
        vec={"method": "pair_separate"}, note="sin |Xp-Xh| ni solape"
    ),
    "no_pos_features": dict(use_pos_features=False, note="sin ratios POS"),
    "no_relation_features": dict(
        use_relation_features=False, note="sin features de relación"
    ),
    "no_aux_features": dict(
        use_pos_features=False,
        use_relation_features=False,
        note="solo texto vectorizado",
    ),
}


def prep_params(**overrides) -> dict:
    """Parámetros de ``NLITextPreprocesser`` partiendo de los del proyecto."""
    params = dict(PREPROCESSING_PARAMS)
    params.update(overrides)
    return params


def vec_params(**overrides) -> dict:
    """Parámetros de ``NLITextVectorizer`` partiendo de los del proyecto."""
    params = dict(VECTORIZER_PARAMS)
    params.update(overrides)
    return params


def results_path(model: str) -> Path:
    """Ruta del JSONL de resultados del banco de un modelo."""
    return ROOT / f"{model}_experiments_results.jsonl"


def load_results(model: str) -> pd.DataFrame:
    """Carga el JSONL del banco de un modelo; vacío si aún no se ejecutó."""
    path = results_path(model)
    if not path.exists():
        return pd.DataFrame()
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(rows)


def run(
    name: str,
    model: str,
    train: pd.DataFrame,
    spec: dict | None = None,
    *,
    results: str | Path | None = None,
) -> dict:
    """Ejecuta una configuración con CV y la añade al JSONL del modelo.

    Args:
        name: Nombre del experimento dentro del banco.
        model: Nombre corto del clasificador (``classifiers.CLASSIFIER_NAMES``).
        train: Partición de entrenamiento ya dividida; el holdout no se toca.
        spec: Diccionario del experimento. Claves admitidas: ``prep``
            (overrides de preprocesado), ``vec`` (overrides del vectorizador),
            ``clf`` (overrides de los hiperparámetros baseline), ``select_k``
            (``None`` desactiva chi²), ``use_pos_features``,
            ``use_relation_features``, ``n_splits``, ``seed`` y ``note``.
        results: JSONL de salida; por defecto el del modelo.

    Returns:
        Diccionario con la configuración evaluada y sus métricas: medias y
        desviaciones de accuracy y macro-F1, las métricas por fold (para
        comparaciones pareadas) y los tiempos de preprocesado, ajuste y
        validación cruzada.
    """
    spec = dict(spec or {})
    prep = prep_params(**(spec.get("prep") or {}))
    vectorizer = vec_params(**(spec.get("vec") or {}))
    classifier = dict(BASELINE_SPECS[model])
    classifier.update(spec.get("clf") or {})

    select_k = spec.get("select_k", 300)
    use_pos_features = spec.get("use_pos_features", True)
    use_relation_features = spec.get("use_relation_features", True)
    n_splits = int(spec.get("n_splits", 5))
    seed = int(spec.get("seed", 42))

    started = time.perf_counter()
    processed = preprocess_partition(train, prep, cache_dir=CACHE)
    preprocess_seconds = time.perf_counter() - started

    pipeline = build_pipeline(
        vectorizer,
        classifier_params=classifier,
        classifier=model,
        select_k=select_k,
        use_pos_features=use_pos_features,
        use_relation_features=use_relation_features,
        random_state=seed,
    )

    started = time.perf_counter()
    cv = cross_validate(processed, pipeline, n_splits=n_splits, seed=seed)
    cv_seconds = time.perf_counter() - started

    result = {
        "model": model,
        "name": name,
        "note": spec.get("note", ""),
        "preprocessing": {
            key: prep[key]
            for key in (
                "reduction",
                "stopword_method",
                "keep_negations",
                "tokenizer",
                "strip_accents",
                "lowercase",
                "use_pos",
            )
        },
        "vectorizer": vectorizer,
        "classifier": model,
        "classifier_params": classifier,
        "select_k": select_k,
        "use_pos_features": use_pos_features,
        "use_relation_features": use_relation_features,
        "n_splits": n_splits,
        "seed": seed,
        "accuracy_mean": cv["summary"]["accuracy_mean"],
        "accuracy_std": cv["summary"]["accuracy_std"],
        "macro_f1_mean": cv["summary"]["macro_f1_mean"],
        "macro_f1_std": cv["summary"]["macro_f1_std"],
        "fold_accuracy": [float(value) for value in cv["folds"]["accuracy"]],
        "fold_macro_f1": [float(value) for value in cv["folds"]["macro_f1"]],
        "fit_seconds_mean": cv["summary"]["fit_seconds_mean"],
        "predict_seconds_mean": cv["summary"]["predict_seconds_mean"],
        "preprocess_seconds": round(preprocess_seconds, 1),
        "cv_seconds": round(cv_seconds, 1),
    }

    out_path = Path(results) if results is not None else results_path(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
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


def run_cli(model: str, experiments: dict, argv=None) -> list[dict]:
    """Punto de entrada de los scripts: ``--list`` o los nombres pedidos.

    Args:
        model: Nombre corto del clasificador del banco.
        experiments: Registro ``{nombre: spec}`` del banco.
        argv: Argumentos de línea de órdenes; por defecto ``sys.argv[1:]``.

    Returns:
        Lista de resultados en el orden de ejecución (vacía con ``--list``).

    Raises:
        SystemExit: Si se pide un experimento que no existe en el registro.
    """
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--list" in argv:
        print(f"{len(experiments)} experimentos de {model}:")
        for name, spec in experiments.items():
            print(f"  {name:28s} {spec.get('note', '')}")
        return []

    names = [arg for arg in argv if not arg.startswith("--")]
    if not names:
        names = list(experiments)

    unknown = [name for name in names if name not in experiments]
    if unknown:
        raise SystemExit(
            f"experimento desconocido en {model}: {unknown[0]!r}. "
            f"Usa --list para ver los disponibles."
        )

    df = load_train_data(PACKAGE / "data/train.csv")
    train, _ = make_split(df, val_size=0.2, seed=42)
    del df

    out = []
    for name in names:
        print(f"== {model}:{name} ==", flush=True)
        out.append(run(name, model, train, experiments[name]))
    return out
