"""Compare classifiers across preprocessed datasets using shared folds.

Each run compares dataset variants, text representations, and classifiers.
Each fold fits the vectorizer and classifier using its training data.
The results include cross-validation metrics and optional holdout predictions.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from nli_project.models.classifiers import CLASSIFIER_NAMES
from experiments.configs.experiment_defaults import CLASSIFIER_SPECS
from nli_project.data.dataset import LABEL_NAMES
from nli_project.training.training import (
    build_pipeline,
    cross_validate,
)
from nli_project.evaluation.metrics import evaluate_holdout
from nli_project.evaluation.reporting import save_json


DEFAULT_REPRESENTATIONS = {
    "pair_diff": {"method": "pair_diff", "min_df": 5, "word_ngram_range": (1, 1), "use_pos": False},
    "pair_separate": {"method": "pair_separate", "min_df": 5, "word_ngram_range": (1, 1), "use_pos": False},
}

# Preserve the settings of the historical runs of this matrix.
DEFAULT_CLASSIFIERS = {
    name: dict(CLASSIFIER_SPECS[name])
    for name in ("naive_bayes", "complement_nb", "random_forest", "catboost")
}
DEFAULT_CLASSIFIERS["random_forest"]["n_jobs"] = -1
DEFAULT_CLASSIFIERS["catboost"]["thread_count"] = -1

def load_processed_dataset(path: Path) -> pd.DataFrame:
    """Load processed text, IDs, original languages, and integer labels."""
    if path.suffix.lower() in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
    else:
        frame = pd.read_csv(path, dtype={"id": str}, keep_default_na=False)
    text_columns = ("premise_processed", "hypothesis_processed")
    translated_columns = ("premise_translated_processed", "hypothesis_translated_processed")
    if all(column in frame for column in translated_columns):
        if any(column in frame for column in text_columns):
            raise ValueError(f"Columnas procesadas ambiguas en {path}")
        # The pipeline uses the same column names for every dataset variant.
        for target, translated in zip(text_columns, translated_columns):
            frame[target] = frame[translated]

    missing = sorted({"label", *text_columns} - set(frame.columns))
    if missing:
        raise ValueError(f"{path} no contiene columnas requeridas: {missing}")
    if frame["label"].isna().any():
        raise ValueError(f"{path} contiene etiquetas nulas.")
    if (
        "id" not in frame
        or frame["id"].isna().any()
        or frame["id"].astype(str).str.strip().eq("").any()
        or not frame["id"].is_unique
    ):
        raise ValueError(f"{path} requiere identificadores no nulos y únicos.")
    if (
        "lang_abv" not in frame
        or frame["lang_abv"].isna().any()
        or frame["lang_abv"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"{path} requiere el idioma original en 'lang_abv'.")
    invalid = sorted(set(frame["label"]) - set(LABEL_NAMES))
    if invalid:
        raise ValueError(f"Etiquetas desconocidas en {path}: {invalid}")
    return frame.assign(label=frame["label"].astype("int64"))


def validate_dataset_alignment(
    datasets: list[tuple[str, Path, pd.DataFrame]],
    *,
    group_by_premise: bool = False,
) -> None:
    """Ensure all data variants retain the same IDs, labels and original languages."""
    if not datasets:
        raise ValueError("No hay datasets para comparar.")
    reference_name, _, reference = datasets[0]
    columns = ["id", "label", "lang_abv"]
    if group_by_premise:
        columns.append("premise")
        for _, path, frame in datasets:
            if "premise" not in frame or frame["premise"].isna().any():
                raise ValueError(f"{path} requiere la premisa original para agrupar los folds.")
    reference_rows = reference[columns].reset_index(drop=True)
    for _, path, frame in datasets[1:]:
        rows = frame[columns].reset_index(drop=True)
        if not reference_rows.equals(rows):
            raise ValueError(
                f"{path} no conserva el orden, las etiquetas, los idiomas originales "
                f"o las columnas de agrupación del dataset {reference_name}."
            )


def _save_holdout_results(
    pipeline,
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    val_path: Path,
    prediction_path: Path,
) -> dict:
    """Evaluate the reserved split and save one prediction per row."""
    holdout = evaluate_holdout(pipeline, train_frame, val_frame)
    predictions = pipeline.predict(val_frame)
    prediction_frame = pd.DataFrame({
        "id": val_frame["id"].to_numpy(),
        "label": val_frame["label"].to_numpy(),
        "prediction": pd.Series(predictions).astype("int64").to_numpy(),
        "lang_abv": val_frame["lang_abv"].to_numpy(),
    })
    prediction_frame.to_csv(prediction_path, index=False)
    return {
        "accuracy": holdout["accuracy"],
        "macro_f1": holdout["macro_f1"],
        "runtime_seconds": holdout["runtime_seconds"],
        "n_train": holdout["n_train"],
        "n_val": holdout["n_val"],
        "classification_report": holdout["classification_report"],
        "confusion_matrix": holdout["confusion_matrix"].tolist(),
        "per_language": holdout["per_language"].to_dict(orient="records"),
        "predictions_path": str(prediction_path.resolve()),
        "holdout_dataset_path": str(val_path.resolve()),
    }


def _describe_inputs(datasets: list[tuple[str, Path]]) -> list[dict]:
    inputs = []
    for name, path in datasets:
        with path.open("rb") as dataset:
            digest = hashlib.file_digest(dataset, "sha256").hexdigest()
        inputs.append({"name": name, "path": str(path.resolve()), "sha256": digest})
    return inputs


def run_matrix(
    datasets: list[tuple[str, Path]],
    representations: dict[str, dict],
    classifiers: list[str],
    output_dir: Path,
    *,
    n_splits: int = 5,
    seed: int = 42,
    select_k: int | None = 300,
    group_by_premise: bool = False,
    holdout_datasets: list[tuple[str, Path]] | None = None,
    holdout_classifiers: list[str] | None = None,
    use_pos_features: bool = False,
    use_relation_features: bool = False,
    reuse: bool = False,
) -> list[dict]:
    """Cross-validate every combination and return its saved result.

    Evaluate the reserved split only for the requested classifiers.
    With reuse=True, store each configuration in its own subdirectory.
    """
    if n_splits < 2:
        raise ValueError("n_splits debe ser al menos 2.")
    unknown_classifiers = sorted(set(classifiers) - set(CLASSIFIER_NAMES))
    if unknown_classifiers:
        raise ValueError(f"Clasificadores no válidos: {unknown_classifiers}")
    if not representations or not classifiers:
        raise ValueError("Debe indicarse al menos una representación y un modelo.")

    # Compare variants on the same examples in the same order.
    loaded = [(name, path, load_processed_dataset(path)) for name, path in datasets]
    validate_dataset_alignment(loaded, group_by_premise=group_by_premise)
    holdout_by_name = {}
    if holdout_datasets:
        if {name for name, _ in datasets} != {name for name, _ in holdout_datasets}:
            raise ValueError("Los nombres de dataset deben coincidir entre train y holdout.")
        loaded_holdout = [
            (name, path, load_processed_dataset(path)) for name, path in holdout_datasets
        ]
        validate_dataset_alignment(loaded_holdout, group_by_premise=group_by_premise)
        for name, path, frame in loaded_holdout:
            holdout_by_name.setdefault(name, (path, frame))
        for name, _, train_frame in loaded:
            _, val_frame = holdout_by_name[name]
            if set(train_frame["id"]) & set(val_frame["id"]):
                raise ValueError(f"Hay IDs compartidos entre train y holdout en {name}.")
    finalist_set = set(holdout_classifiers or [])
    unknown_finalists = sorted(finalist_set - set(classifiers))
    if unknown_finalists:
        raise ValueError(f"Finalistas ausentes de la comparación CV: {unknown_finalists}")

    feature_flags = {
        "use_pos_features": use_pos_features,
        "use_relation_features": use_relation_features,
    }
    spec = {
        "version": 1,
        "datasets": _describe_inputs(datasets),
        "holdout_datasets": _describe_inputs(holdout_datasets or []),
        "representations": list(representations.items()),
        "classifiers": [(name, DEFAULT_CLASSIFIERS.get(name, {})) for name in classifiers],
        "holdout_classifiers": sorted(finalist_set),
        "seed": seed,
        "n_splits": n_splits,
        "group_by_premise": group_by_premise,
        "select_k": select_k,
        **feature_flags,
    }
    spec_json = json.dumps(spec, sort_keys=True)
    output_dir = output_dir.resolve()
    if reuse:
        output_dir /= hashlib.sha256(spec_json.encode("utf-8")).hexdigest()[:16]
    if output_dir.exists():
        if not reuse:
            raise FileExistsError(output_dir)
        result_path = output_dir / "results.json"
        if not result_path.is_file():
            raise ValueError(f"Resultados incompletos en {output_dir}. Usa otro directorio.")
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if saved.get("spec") != json.loads(spec_json):
            raise ValueError(f"La configuración guardada no coincide en {output_dir}")
        records = saved["experiments"]
        if len(records) != len(datasets) * len(representations) * len(classifiers) or any(
            not Path(record["holdout"]["predictions_path"]).is_file()
            for record in records if "holdout" in record
        ):
            raise ValueError(f"Resultados incompletos en {output_dir}. Usa otro directorio.")
        print(f"Resultados reutilizados: {output_dir}", flush=True)
        return records

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".matrix-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        records = []
        experiments = product(loaded, representations.items(), classifiers)
        for dataset, representation, classifier_name in experiments:
            dataset_name, dataset_path, frame = dataset
            representation_name, vectorizer_params = representation
            experiment_spec = {
                "dataset": dataset_name,
                "representation": representation_name,
                "vectorizer_params": vectorizer_params,
                "classifier": classifier_name,
                "classifier_params": DEFAULT_CLASSIFIERS.get(classifier_name, {}),
                "seed": seed,
                "n_splits": n_splits,
                "group_by_premise": group_by_premise,
                "select_k": select_k,
                **feature_flags,
            }
            # Use the same ID in the results and the predictions filename.
            experiment_id = hashlib.sha256(
                json.dumps(experiment_spec, sort_keys=True, default=list).encode()
            ).hexdigest()[:16]
            pipeline = build_pipeline(
                dict(vectorizer_params),
                classifier=classifier_name,
                classifier_params=DEFAULT_CLASSIFIERS.get(classifier_name),
                select_k=select_k,
                random_state=seed,
                **feature_flags,
            )
            evaluated = cross_validate(
                frame,
                pipeline,
                n_splits=n_splits,
                seed=seed,
                group_by_premise=group_by_premise,
            )
            record = {
                **experiment_spec,
                "experiment_id": experiment_id,
                "dataset_path": str(dataset_path.resolve()),
                "n_examples": len(frame),
                "summary": evaluated["summary"],
                "confusion_matrix": evaluated["confusion_matrix"].tolist(),
                "folds": evaluated["folds"].to_dict(orient="records"),
            }
            # Use the reserved partition only for the selected classifiers.
            if holdout_by_name and classifier_name in finalist_set:
                val_path, val_frame = holdout_by_name[dataset_name]
                prediction_path = stage / f"holdout_predictions_{experiment_id}.csv"
                record["holdout"] = _save_holdout_results(
                    pipeline, frame, val_frame, val_path, prediction_path
                )
                record["holdout"]["predictions_path"] = str(output_dir / prediction_path.name)
            records.append(record)
            print(
                f"{experiment_id} {dataset_name}/{representation_name}/{classifier_name} "
                f"acc={record['summary']['accuracy_mean']:.4f} "
                f"macro-F1={record['summary']['macro_f1_mean']:.4f}",
                flush=True,
            )
        save_json({"spec": spec, "experiments": records}, stage / "results.json")
        stage.rename(output_dir)
        print(f"Resultados guardados: {output_dir}", flush=True)
        return records
