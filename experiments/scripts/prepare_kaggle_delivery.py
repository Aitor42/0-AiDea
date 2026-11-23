"""Prepara datos y código para el cuaderno final de Kaggle.

Importa y llama a prepare_delivery() desde un cuaderno.

Exige traducciones terminadas de train y test. La carpeta de entrega nueva se
publica sin sobrescribir una ejecución anterior.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np

from experiments.scripts.bundle import digest, make_code_bundle

from nli_project.data.dataset import LABEL_NAMES
from nli_project.preprocessing.preprocessing import (
    PREPROCESSING_PARAMS,
    preprocess_partition,
)
from nli_project.features.pos_features import POSRatioFeatures
from nli_project.features.relation_features import RelationFeatures
from nli_project.features.vectorizer import _to_text


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "experiments/.local/kaggle_delivery"
TRAIN_INPUT = ROOT / "experiments/results/translation/corpus/train_gemma3_4b.csv"
TEST_INPUT = ROOT / "experiments/results/translation/corpus/test_gemma3_4b.csv"


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_INPUT, dtype={"id": str}, keep_default_na=False)
    test = pd.read_csv(TEST_INPUT, dtype={"id": str}, keep_default_na=False)
    official_train = pd.read_csv(ROOT / "experiments/data/train.csv", dtype={"id": str}, keep_default_na=False)
    official_test = pd.read_csv(ROOT / "experiments/data/test.csv", dtype={"id": str}, keep_default_na=False)
    sample = pd.read_csv(ROOT / "experiments/data/sample_submission.csv", dtype={"id": str}, keep_default_na=False)

    for name, frame, official in (("train", train, official_train), ("test", test, official_test)):
        required = {"id", "premise", "hypothesis", "lang_abv", "premise_translated", "hypothesis_translated"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"La traducción de {name} no contiene: {missing}")
        if frame["id"].isna().any() or frame["id"].duplicated().any():
            raise ValueError(f"Los IDs traducidos de {name} deben ser únicos y no nulos")
        expected_ids = official["id"].tolist()
        if frame["id"].tolist() != expected_ids:
            raise ValueError(f"Los IDs de la traducción de {name} no coinciden con el CSV oficial en orden")
        for column in ("premise", "hypothesis", "lang_abv"):
            if not frame[column].equals(official[column]):
                raise ValueError(f"La traducción de {name} alteró la columna de origen {column}")
        if frame[["premise_translated", "hypothesis_translated"]].eq("").any().any():
            raise ValueError(f"Hay traducciones vacías en {name}")

    if "label" not in train or train["label"].isna().any():
        raise ValueError("Train debe conservar sus etiquetas")
    if "label" in test:
        raise ValueError("Test no debe incorporar etiquetas")
    if set(train["label"].astype(int)) != set(LABEL_NAMES):
        raise ValueError("Train no contiene exactamente las tres clases oficiales")
    if train["label"].astype(int).tolist() != official_train["label"].astype(int).tolist():
        raise ValueError("La traducción de train alteró las etiquetas")
    if sample.columns.tolist() != ["id", "prediction"] or sample.id.tolist() != test.id.tolist():
        raise ValueError("El esquema/orden de sample_submission no coincide con test")
    return train, test


def prepare_delivery() -> None:
    if not TRAIN_INPUT.is_file() or not TEST_INPUT.is_file():
        raise FileNotFoundError(
            "Se necesitan artifacts/translations/train_gemma3_4b.csv y "
            "test_gemma3_4b.csv. Completa primero ambas traducciones."
        )
    if OUTPUT.exists():
        raise FileExistsError(f"Se conserva la carpeta de entrega existente: {OUTPUT}")

    train, test = load_and_validate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kaggle-delivery-", dir=OUTPUT.parent) as staging_name:
        staging = Path(staging_name)
        cache_dir = ROOT / "experiments/.local/preprocessed"
        preprocess_params = {
            **PREPROCESSING_PARAMS,
            "columns": ("premise_translated", "hypothesis_translated"),
            "language": None,
        }
        # El preprocesador deriva los nombres de las columnas de entrada. El
        # pipeline de la comparación usa los mismos nombres en ambos brazos.
        model_columns = {"premise_translated_processed": "premise_processed",
                         "hypothesis_translated_processed": "hypothesis_processed"}
        processed_train = preprocess_partition(train, preprocess_params, cache_dir).rename(
            columns=model_columns, errors="raise")
        processed_test = preprocess_partition(test, preprocess_params, cache_dir).rename(
            columns=model_columns, errors="raise")
        if processed_train.id.tolist() != train.id.tolist() or processed_test.id.tolist() != test.id.tolist():
            raise ValueError("El preprocesador alteró el orden o los IDs")
        if processed_train.label.tolist() != train.label.astype("int64").tolist():
            raise ValueError("El preprocesador alteró las etiquetas de train")

        columns = [
            "id", "lang_abv", "premise_processed", "hypothesis_processed"
        ]
        train_export = processed_train.loc[:, [*columns, "label"]].copy()
        test_export = processed_test.loc[:, columns].copy()
        train_export.to_json(staging / "train_processed.jsonl", orient="records", lines=True, force_ascii=False)
        test_export.to_json(staging / "test_processed.jsonl", orient="records", lines=True, force_ascii=False)

        # JSONL serializes token/POS tuples as arrays. These are the exact text
        # and POS inputs consumed by the shared vectorizer/features.
        roundtrip = pd.read_json(staging / "train_processed.jsonl", lines=True, dtype={"id": str})
        sample_before = processed_train.head(128)
        sample_after = roundtrip.head(128)
        for column in ("premise_processed", "hypothesis_processed"):
            before = [_to_text(value) for value in sample_before[column]]
            after = [_to_text(value) for value in sample_after[column]]
            if before != after:
                raise ValueError(f"La serialización JSONL alteró los tokens de {column}")
        np.testing.assert_allclose(
            POSRatioFeatures().fit_transform(sample_before),
            POSRatioFeatures().fit_transform(sample_after),
        )
        np.testing.assert_allclose(
            RelationFeatures().fit_transform(sample_before),
            RelationFeatures().fit_transform(sample_after),
        )
        shutil.copy2(ROOT / "experiments/data/sample_submission.csv", staging / "sample_submission.csv")
        source_hashes = make_code_bundle(staging / "training_source.zip")
        manifest = {
            "schema_version": 1,
            "translation_model": "gemma3:4b",
            "translation_backend": "ollama",
            "preprocessor_params": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in preprocess_params.items()
            },
            "train_rows": len(processed_train),
            "test_rows": len(processed_test),
            "test_ids_sha256": hashlib.sha256("\n".join(test.id).encode("utf-8")).hexdigest(),
            "training_source_sha256": source_hashes,
            "artifacts": {
                name: digest(staging / name)
                for name in (
                    "train_processed.jsonl",
                    "test_processed.jsonl",
                    "sample_submission.csv",
                    "training_source.zip",
                )
            },
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, OUTPUT)
    print(f"Entrega preparada en {OUTPUT}")
    print(f"Train: {len(train)} filas · test: {len(test)} filas")


if __name__ == "__main__":
    prepare_delivery()
