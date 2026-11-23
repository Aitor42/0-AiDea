"""Comparación de algoritmos sobre la configuración ganadora del proyecto.

Bloque 2 del estudio: evalúa con validación cruzada estratificada los
algoritmos vistos en teoría y práctica —Naive Bayes, árboles de decisión,
métodos de ensemble (Random Forest), kNN, redes feedforward (MLP)— y los tres
gradient boosting del enunciado (XGBoost, CatBoost y LightGBM) sobre el mismo
split, el mismo preprocesado y la misma representación que el modelo final de
``training.py``. Del ganador deja evidencias reproducibles: tabla comparativa,
tuning opcional sobre los mismos folds, evaluación holdout (métricas, tabla por
lengua y matriz de confusión), submission sobre ``test.csv`` y un
``final_config.json`` con el nombre corto del clasificador y sus parámetros.

Decisiones de diseño relevantes:

- Mismos folds y misma semilla para todos los algoritmos: la única variable de
  la comparación es el clasificador (con su especificación en
  ``CLASSIFIER_SPECS``). El vocabulario, el IDF, las features auxiliares y la
  selección chi² se ajustan dentro de cada fold, sin fugas.
- El ganador se marca con ``select_winner`` (mayor ``accuracy_mean``,
  desempate por ``macro_f1_mean`` y, en último caso, por nombre), sin
  reevaluar nada.
- El catálogo incluye los boosting externos, que se importan de forma perezosa
  en ``classifiers.build_classifier``; si un paquete no está instalado, solo
  falla ese algoritmo. Se puede acotar con ``--classifiers``.

Uso:
    .venv/bin/python -m experiments.scripts.run_algorithm_comparison
    .venv/bin/python -m experiments.scripts.run_algorithm_comparison --classifiers random_forest,xgboost
    .venv/bin/python -m experiments.scripts.run_algorithm_comparison --skip-holdout --skip-submission
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402

from nli_project.models.classifiers import CLASSIFIER_NAMES  # noqa: E402
from experiments.configs.experiment_defaults import CLASSIFIER_SPECS  # noqa: E402
from nli_project.data.dataset import LABEL_NAMES, load_train_data, make_split  # noqa: E402
from nli_project.evaluation.metrics import evaluate_holdout  # noqa: E402
from nli_project.evaluation.reporting import save_json  # noqa: E402
from nli_project.preprocessing.preprocessing import preprocess_partition  # noqa: E402
from nli_project.submission.csv_output import write_submission  # noqa: E402
from nli_project.training.training import build_pipeline, cross_validate  # noqa: E402

# Catálogo evaluado por defecto: los nombres de ``CLASSIFIER_SPECS`` en orden
# (sin el alias ``multinomial_nb`` de ``classifiers.CLASSIFIER_NAMES``).
DEFAULT_CLASSIFIERS = tuple(CLASSIFIER_SPECS)

# Métrica usada para elegir el algoritmo ganador en la tabla comparativa.
SELECTION_METRIC = "accuracy_mean"


def _param_product(**axes) -> list[dict]:
    """Producto cartesiano de ejes de hiperparámetros como lista de variaciones."""
    combos = [{}]
    for key, values in axes.items():
        combos = [{**combo, key: value} for combo in combos for value in values]
    return combos


# Variaciones que el tuning combina sobre la especificación baseline de cada
# algoritmo (``CLASSIFIER_SPECS``). Solo se listan los ejes que se retocan; el
# resto de hiperparámetros del spec quedan fijos.
TUNING_VARIATIONS = {
    "naive_bayes": _param_product(alpha=[1.0, 3.0, 10.0], fit_prior=[False, True]),
    "complement_nb": _param_product(alpha=[1.0, 3.0, 10.0], norm=[True, False]),
    "bernoulli_nb": _param_product(alpha=[0.5, 1.0, 3.0], fit_prior=[False, True]),
    "decision_tree": _param_product(max_depth=[10, 20, None], min_samples_leaf=[1, 5]),
    "random_forest": _param_product(
        n_estimators=[500, 1000], min_samples_leaf=[1, 2, 5], max_depth=[50]
    ),
    "knn": _param_product(n_neighbors=[25, 50, 75], weights=["distance", "uniform"]),
    "mlp": _param_product(hidden_layer_sizes=[[256], [512]], alpha=[1e-4, 1e-3]),
    "xgboost": _param_product(
        n_estimators=[200, 300], max_depth=[4, 6], learning_rate=[0.1]
    ),
    "catboost": _param_product(iterations=[200, 300], depth=[4, 6]),
    "lightgbm": _param_product(
        n_estimators=[200, 300], num_leaves=[15, 31], learning_rate=[0.05]
    ),
}


def comparison_pipelines(
    vectorizer_params: dict,
    *,
    classifiers=None,
    select_k: int | None = 300,
    use_pos_features: bool = True,
    use_relation_features: bool = True,
    random_state: int = 42,
) -> dict[str, Pipeline]:
    """Construye un pipeline por clasificador con la misma representación.

    Args:
        vectorizer_params: Argumentos de ``NLITextVectorizer`` (no se muta).
        classifiers: Nombres de clasificadores a incluir; ``None`` usa
            ``DEFAULT_CLASSIFIERS``.
        select_k: Características que conserva ``SelectKBest(chi2)``; ``None``
            desactiva la selección.
        use_pos_features: Si es True añade los ratios POS.
        use_relation_features: Si es True añade las señales de relación.
        random_state: Semilla global de cada pipeline.

    Returns:
        Diccionario ``{nombre: Pipeline}`` en el orden pedido.

    Raises:
        ValueError: Si algún nombre de clasificador no es válido.
    """
    names = list(classifiers) if classifiers is not None else list(DEFAULT_CLASSIFIERS)
    unknown = [name for name in names if name not in CLASSIFIER_NAMES]
    if unknown:
        raise ValueError(
            f"Clasificador no válido: {unknown}. "
            f"Valores válidos: {list(CLASSIFIER_NAMES)}"
        )

    return {
        name: build_pipeline(
            dict(vectorizer_params),
            classifier=name,
            classifier_params=CLASSIFIER_SPECS.get(name),
            select_k=select_k,
            use_pos_features=use_pos_features,
            use_relation_features=use_relation_features,
            random_state=random_state,
        )
        for name in names
    }


def _classifier_params_json(name: str, pipeline: Pipeline) -> str:
    """Serializa los hiperparámetros del clasificador de un pipeline."""
    params = CLASSIFIER_SPECS.get(name)
    if params is None:
        params = pipeline.named_steps["classifier"].get_params(deep=False)
    return json.dumps(params, sort_keys=True, ensure_ascii=False)


def comparison_rows(
    df: pd.DataFrame,
    pipelines: dict,
    *,
    n_splits: int = 5,
    seed: int = 42,
    group_by_premise: bool = False,
) -> list[dict]:
    """Valida cruzadamente cada pipeline y construye las filas comparativas."""
    rows: list[dict] = []
    for name, pipeline in pipelines.items():
        summary = cross_validate(
            df,
            pipeline,
            n_splits=n_splits,
            seed=seed,
            group_by_premise=group_by_premise,
        )["summary"]
        select_step = pipeline.named_steps.get("select")
        rows.append(
            {
                "experiment_id": f"algorithm__{name}",
                "classifier": name,
                "classifier_params": _classifier_params_json(name, pipeline),
                "select_k": select_step.k if select_step is not None else None,
                "use_pos_features": isinstance(
                    pipeline.named_steps["features"], FeatureUnion
                ),
                "accuracy_mean": summary["accuracy_mean"],
                "accuracy_std": summary["accuracy_std"],
                "macro_f1_mean": summary["macro_f1_mean"],
                "macro_f1_std": summary["macro_f1_std"],
                "fit_seconds_mean": summary["fit_seconds_mean"],
                "predict_seconds_mean": summary["predict_seconds_mean"],
                "n_splits": int(n_splits),
                "seed": int(seed),
            }
        )
        print(
            f"{name:16s} acc={rows[-1]['accuracy_mean']:.4f} "
            f"± {rows[-1]['accuracy_std']:.4f} "
            f"f1={rows[-1]['macro_f1_mean']:.4f} "
            f"({rows[-1]['fit_seconds_mean']:.1f}s/fold)",
            flush=True,
        )
    return rows


def select_winner(rows, metric: str = SELECTION_METRIC) -> dict:
    """Selecciona la fila ganadora de la comparación.

    Criterio en cascada: mayor ``metric``, luego mayor ``macro_f1_mean`` y,
    como desempate determinista final, menor nombre de clasificador.
    """
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient="records")
    rows = list(rows)
    if not rows:
        raise ValueError("No hay filas de donde seleccionar el algoritmo ganador.")
    if metric not in rows[0]:
        raise ValueError(
            f"La métrica {metric!r} no existe en las filas. "
            f"Columnas disponibles: {sorted(rows[0])}"
        )

    def sort_key(row: dict):
        primary = float(row.get(metric, float("-inf")))
        fallback = float(row.get("macro_f1_mean", float("-inf")))
        return (-primary, -fallback, str(row.get("classifier", "")))

    return dict(min(rows, key=sort_key))


def tuning_grid(name: str) -> list[dict]:
    """Rejilla de hiperparámetros del tuning, con el baseline en primer lugar."""
    if name not in CLASSIFIER_NAMES:
        raise ValueError(
            f"Clasificador no válido: {name!r}. "
            f"Valores válidos: {list(CLASSIFIER_NAMES)}"
        )
    spec_name = "naive_bayes" if name == "multinomial_nb" else name
    grid: list[dict] = []
    seen: set[str] = set()
    for variation in [{}] + list(TUNING_VARIATIONS.get(spec_name, [])):
        config = dict(CLASSIFIER_SPECS[spec_name], **variation)
        key = json.dumps(config, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            grid.append(config)
    return grid


def _tuning_slug(config: dict) -> str:
    """Slug determinista de una config: prefijo SHA-1 de su JSON canónico."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def tuning_rows(
    df: pd.DataFrame,
    classifier: str,
    vectorizer_params: dict,
    *,
    grid: list[dict] | None = None,
    n_splits: int = 5,
    seed: int = 42,
    select_k: int | None = 300,
    use_pos_features: bool = True,
    use_relation_features: bool = True,
    random_state: int = 42,
    group_by_premise: bool = False,
) -> list[dict]:
    """Valida cruzadamente cada config de la rejilla de un clasificador."""
    configs = grid if grid is not None else tuning_grid(classifier)
    # Mismo splitter que ``cross_validate`` (sin agrupar): el fold de train más
    # pequeño acota las configs de kNN evaluables.
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    min_train = min(len(train_idx) for train_idx, _ in splitter.split(df, df["label"]))
    evaluable = [
        config
        for config in configs
        if "n_neighbors" not in config or config["n_neighbors"] <= min_train
    ]
    if not evaluable:
        evaluable = []
        seen: set[str] = set()
        for config in configs:
            if "n_neighbors" in config:
                config = {
                    **config,
                    "n_neighbors": min(config["n_neighbors"], min_train),
                }
            key = json.dumps(config, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                evaluable.append(config)

    rows: list[dict] = []
    for config in evaluable:
        pipeline = build_pipeline(
            dict(vectorizer_params),
            classifier=classifier,
            classifier_params=config,
            select_k=select_k,
            use_pos_features=use_pos_features,
            use_relation_features=use_relation_features,
            random_state=random_state,
        )
        summary = cross_validate(
            df,
            pipeline,
            n_splits=n_splits,
            seed=seed,
            group_by_premise=group_by_premise,
        )["summary"]
        select_step = pipeline.named_steps.get("select")
        rows.append(
            {
                "experiment_id": f"tuning__{classifier}__{_tuning_slug(config)}",
                "classifier": classifier,
                "classifier_params": json.dumps(
                    config, sort_keys=True, ensure_ascii=False
                ),
                "select_k": select_step.k if select_step is not None else None,
                "use_pos_features": isinstance(
                    pipeline.named_steps["features"], FeatureUnion
                ),
                "accuracy_mean": summary["accuracy_mean"],
                "accuracy_std": summary["accuracy_std"],
                "macro_f1_mean": summary["macro_f1_mean"],
                "macro_f1_std": summary["macro_f1_std"],
                "fit_seconds_mean": summary["fit_seconds_mean"],
                "predict_seconds_mean": summary["predict_seconds_mean"],
                "n_splits": int(n_splits),
                "seed": int(seed),
            }
        )
        print(
            f"  tuning {config} -> acc={rows[-1]['accuracy_mean']:.4f} "
            f"± {rows[-1]['accuracy_std']:.4f}",
            flush=True,
        )
    return rows


def select_tuned(rows, metric: str = SELECTION_METRIC) -> dict:
    """Selecciona la mejor configuración ajustada de la rejilla de tuning."""
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient="records")
    rows = list(rows)
    if not rows:
        raise ValueError("No hay filas de donde seleccionar la configuración ajustada.")
    if metric not in rows[0]:
        raise ValueError(
            f"La métrica {metric!r} no existe en las filas. "
            f"Columnas disponibles: {sorted(rows[0])}"
        )

    def sort_key(row: dict):
        primary = float(row.get(metric, float("-inf")))
        fallback = float(row.get("macro_f1_mean", float("-inf")))
        return (-primary, -fallback, str(row.get("classifier_params", "")))

    return dict(min(rows, key=sort_key))


def plot_confusion_matrix(confusion, path: Path) -> None:
    """Guarda la matriz de confusión 3x3 como PNG."""
    labels = [LABEL_NAMES[label] for label in sorted(LABEL_NAMES)]
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicción")
    axis.set_ylabel("Real")
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(
                column,
                row,
                str(int(confusion[row][column])),
                ha="center",
                va="center",
                color="black",
            )
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description=(
            "Compara los algoritmos del enunciado (y los tres boosting) con "
            "validación cruzada sobre la configuración final del proyecto."
        )
    )
    parser.add_argument("--train-path", default="experiments/data/train.csv")
    parser.add_argument("--test-path", default="experiments/data/test.csv")
    parser.add_argument("--output-dir", default="experiments/.local/classical_models/algorithm_comparison")
    parser.add_argument("--artifacts-dir", default="experiments/.local")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--classifiers",
        default="all",
        help='Nombres separados por comas (p. ej. "random_forest,xgboost") '
        'o "all" para todo el catálogo.',
    )
    parser.add_argument(
        "--select-k",
        type=int,
        default=300,
        help="Características que conserva SelectKBest(chi2); 0 = sin selección.",
    )
    parser.add_argument(
        "--skip-holdout", action="store_true", help="No evalúa el holdout del ganador."
    )
    parser.add_argument(
        "--skip-submission", action="store_true", help="No genera submission.csv."
    )
    parser.add_argument(
        "--skip-tune",
        action="store_true",
        help="No ajusta hiperparámetros del ganador.",
    )
    parser.add_argument(
        "--group-by-premise",
        action="store_true",
        help="Partición de validación agrupada por premisa.",
    )
    args = parser.parse_args(argv)

    raw = str(args.classifiers).strip()
    if raw.lower() == "all" or not raw:
        args.classifiers = None
    else:
        names = [part.strip() for part in raw.split(",") if part.strip()]
        unknown = [name for name in names if name not in DEFAULT_CLASSIFIERS]
        if unknown:
            raise ValueError(
                f"Clasificador no válido: {unknown}. "
                f"Valores válidos: {list(DEFAULT_CLASSIFIERS)}"
            )
        args.classifiers = names
    if args.select_k == 0:
        args.select_k = None
    return args


def main(argv=None) -> None:
    """Ejecuta la comparación completa y guarda las evidencias del ganador."""
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.artifacts_dir) / "preprocessed"

    print("Cargando y preprocesando datos...", flush=True)
    frame = load_train_data(args.train_path)
    train_df, val_df = make_split(
        frame,
        val_size=args.val_size,
        seed=args.seed,
        group_by_premise=args.group_by_premise,
    )
    processed_train = preprocess_partition(train_df, cache_dir=cache_dir)
    print(f"train={len(processed_train)}  val={len(val_df)}", flush=True)

    selected = args.classifiers or list(DEFAULT_CLASSIFIERS)
    pipelines = comparison_pipelines(
        {"method": "pair_diff", "min_df": 5, "word_ngram_range": (1, 1), "use_pos": False},
        classifiers=selected,
        select_k=args.select_k,
        random_state=args.seed,
    )

    print("\n=== Comparación de algoritmos (CV) ===", flush=True)
    rows = comparison_rows(
        processed_train,
        pipelines,
        n_splits=args.n_splits,
        seed=args.seed,
        group_by_premise=args.group_by_premise,
    )
    table = pd.DataFrame(rows).sort_values(
        ["accuracy_mean", "macro_f1_mean"], ascending=False
    )
    table.to_csv(output_dir / "algorithm_comparison.csv", index=False)
    winner = select_winner(rows)
    print(
        f"\nGanador: {winner['classifier']} "
        f"(acc={winner['accuracy_mean']:.4f}, f1={winner['macro_f1_mean']:.4f})",
        flush=True,
    )

    if not args.skip_tune:
        print(f"\n=== Tuning de {winner['classifier']} ===", flush=True)
        tuned_rows = tuning_rows(
            processed_train,
            winner["classifier"],
            {"method": "pair_diff", "min_df": 5, "word_ngram_range": (1, 1), "use_pos": False},
            n_splits=args.n_splits,
            seed=args.seed,
            select_k=args.select_k,
            random_state=args.seed,
            group_by_premise=args.group_by_premise,
        )
        pd.DataFrame(tuned_rows).to_csv(
            output_dir / "tuning_results.csv", index=False
        )
        best = select_tuned(tuned_rows)
        winner = {**winner, **best}
        print(
            f"Mejor config: {winner['classifier_params']} "
            f"(acc={winner['accuracy_mean']:.4f})",
            flush=True,
        )

    winner_params = json.loads(winner["classifier_params"])
    winner_pipeline = build_pipeline(
        {"method": "pair_diff", "min_df": 5, "word_ngram_range": (1, 1), "use_pos": False},
        classifier=winner["classifier"],
        classifier_params=winner_params,
        select_k=args.select_k,
        random_state=args.seed,
    )

    holdout_summary = None
    if not args.skip_holdout:
        print("\n=== Holdout del ganador ===", flush=True)
        processed_val = preprocess_partition(val_df, cache_dir=cache_dir)
        holdout = evaluate_holdout(clone(winner_pipeline), processed_train, processed_val)
        holdout_summary = {
            "accuracy": holdout["accuracy"],
            "macro_f1": holdout["macro_f1"],
            "runtime_seconds": holdout["runtime_seconds"],
        }
        metrics = {
            **holdout_summary,
            "experiment_id": winner["experiment_id"],
            "classifier": winner["classifier"],
            "classifier_params": winner_params,
            "select_k": args.select_k,
            "seed": int(args.seed),
            "n_train": int(len(processed_train)),
            "n_val": int(len(processed_val)),
            "classification_report": holdout["classification_report"],
            "confusion_matrix": holdout["confusion_matrix"].tolist(),
        }
        save_json(metrics, output_dir / "holdout_metrics.json")
        holdout["per_language"].to_csv(output_dir / "per_language.csv", index=False)
        plot_confusion_matrix(
            holdout["confusion_matrix"], output_dir / "confusion_matrix.png"
        )
        print(
            f"accuracy={holdout['accuracy']:.4f}  "
            f"macro_f1={holdout['macro_f1']:.4f}  "
            f"runtime={holdout['runtime_seconds']}s",
            flush=True,
        )

    if not args.skip_submission and Path(args.test_path).exists():
        print("\n=== Submission del ganador ===", flush=True)
        test_df = pd.read_csv(args.test_path)
        processed_test = preprocess_partition(test_df, cache_dir=cache_dir)
        winner_pipeline.fit(processed_train, processed_train["label"])
        predictions = winner_pipeline.predict(processed_test)
        write_submission(test_df["id"], predictions, output_dir / "submission.csv")
        print(f"submission: {output_dir / 'submission.csv'}", flush=True)

    save_json(
        {
            "experiment_id": winner["experiment_id"],
            "classifier": winner["classifier"],
            "classifier_params": winner_params,
            "select_k": args.select_k,
            "use_pos_features": True,
            "n_splits": int(args.n_splits),
            "seed": int(args.seed),
            "accuracy_mean": winner["accuracy_mean"],
            "accuracy_std": winner["accuracy_std"],
            "macro_f1_mean": winner["macro_f1_mean"],
            "macro_f1_std": winner["macro_f1_std"],
            "holdout": holdout_summary,
        },
        output_dir / "final_config.json",
    )
    print(f"\nEvidencias en {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
