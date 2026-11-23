"""Translate train once and compare CatBoost on the same holdout IDs."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from nli_project.translation.ollama import load_translations, translate_file


def compare(model, batch_size, output=None, *, translations=None, generate=False):
    from nli_project.preprocessing.parameter_variants import materialize_variants
    from nli_project.preprocessing.preprocessing import MULTILINGUAL_PREPROCESSING_PARAMS
    from experiments.scripts.run_experiment_matrix import DEFAULT_CLASSIFIERS, DEFAULT_REPRESENTATIONS, load_processed_dataset
    from nli_project.training.training import (
        SELECT_K_FEATURES,
        build_pipeline,
    )
    from nli_project.evaluation.metrics import evaluate_holdout

    project = Path(__file__).resolve().parents[2]
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = project / "experiments/.local/ollama_comparisons" / stamp
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"El directorio de resultados ya existe: {output}")
    safe_name = model.replace(":", "_").replace("/", "_")
    archived = project / "experiments/results/translation/corpus" / f"train_{safe_name}.csv"
    generated = project / "experiments/.local/translations" / archived.name
    translated_path = Path(translations) if translations is not None else (
        archived if archived.is_file() else generated
    )
    train_path = project / "experiments/data/train.csv"
    started = time.monotonic()
    # Reuse a completed translation for this model and check its source rows below.
    if not translated_path.exists():
        if not generate:
            raise FileNotFoundError(f"Falta {translated_path}. Usa generate=True para generarlo.")
        translate_file(train_path, translated_path, model=model, batch_size=batch_size)
    options = {"dtype": {"id": str}, "keep_default_na": False}
    official = pd.read_csv(train_path, **options)
    english = load_translations(official, translated_path).set_index("id")
    elapsed = time.monotonic() - started
    splits = [pd.read_csv(project / f"experiments/results/validation/historical_split/{name}.csv", **options) for name in ("train", "val")]
    source = pd.concat(splits).set_index("id").sort_index()
    if not (source.index.is_unique and english.index.is_unique):
        raise ValueError("Los IDs de las traducciones y particiones deben ser únicos.")
    pd.testing.assert_frame_equal(source, english[source.columns].sort_index())
    features = {"use_pos_features": False, "use_relation_features": False}
    results = {"model": model, "translation_or_cache_seconds": elapsed, "seed": 42,
               "preprocessing": MULTILINGUAL_PREPROCESSING_PARAMS,
               "representation": DEFAULT_REPRESENTATIONS["pair_diff"],
               "select_k": SELECT_K_FEATURES,
               "classifier_params": DEFAULT_CLASSIFIERS["catboost"], **features,
               "translated_pairs": int(english.lang_abv.ne("en").sum()), "arms": {}}
    predictions = splits[1][["id", "label", "lang_abv"]].copy()
    for arm in ("original", "translated"):
        frames = []
        for split in splits:
            data = split
            params = dict(MULTILINGUAL_PREPROCESSING_PARAMS)
            if arm == "translated":
                data = english.loc[split.id].reset_index()
                params["columns"] = ["premise_translated", "hypothesis_translated"]
            info, = materialize_variants(data, variants=[{"name": "baseline", "params": {}}],
                                         base_params=params,
                                         output_dir=project / "experiments/.local/parameter_variants",
                                         cache_dir=project / "experiments/.local/preprocessed")
            frames.append(load_processed_dataset(Path(info["dataset_path"])))
        pipeline = build_pipeline(DEFAULT_REPRESENTATIONS["pair_diff"], classifier="catboost",
                                  classifier_params=DEFAULT_CLASSIFIERS["catboost"], random_state=42,
                                  select_k=SELECT_K_FEATURES,
                                  **features)
        metrics = evaluate_holdout(pipeline, *frames)
        predictions[arm] = pipeline.predict(frames[1])
        metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
        metrics["per_language"] = metrics["per_language"].to_dict("records")
        results["arms"][arm] = metrics
        print(arm, metrics["accuracy"], metrics["macro_f1"], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        stage = Path(directory)
        predictions.to_csv(stage / "predictions.csv", index=False)
        (stage / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        stage.rename(output)
    return output
