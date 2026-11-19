"""Test de integración: preprocesador real + pipeline + evaluación."""

from __future__ import annotations

import pandas as pd
import pytest

from training import (
    build_pipeline,
    cross_validate,
    evaluate_holdout,
    make_split,
    preprocess_partition,
)


@pytest.mark.slow
@pytest.mark.e2e
def test_real_preprocessor_and_training_pipeline_end_to_end(
    synthetic_pairs, preprocessor_params, tmp_path
):
    processed = preprocess_partition(
        synthetic_pairs, preprocessor_params, cache_dir=tmp_path / "cache"
    )

    assert {"premise_processed", "hypothesis_processed", "label"} <= set(
        processed.columns
    )
    assert processed["label"].notna().all()
    assert (processed["lang_abv"] == "en").all()
    assert processed["premise_processed"].map(lambda value: len(value) > 0).all()

    train_df, val_df = make_split(processed, val_size=0.25, seed=42)
    pipeline = build_pipeline(
        {"method": "tfidf", "min_df": 1, "max_features": 5_000},
        classifier_params={"n_estimators": 50, "min_samples_leaf": 1},
        use_pos_features=False,
        select_k=50,
    )

    cv = cross_validate(train_df, pipeline, n_splits=3, seed=42)
    assert cv["summary"]["accuracy_mean"] > 0.4
    assert cv["confusion_matrix"].sum() == len(train_df)

    holdout = evaluate_holdout(pipeline, train_df, val_df)
    assert holdout["accuracy"] > 0.4
    assert holdout["n_val"] == len(val_df)
    assert holdout["confusion_matrix"].sum() == len(val_df)
    assert isinstance(holdout["per_language"], pd.DataFrame)


@pytest.mark.slow
def test_preprocessing_cache_is_reused(synthetic_pairs, preprocessor_params, tmp_path):
    cache_dir = tmp_path / "cache"

    first = preprocess_partition(
        synthetic_pairs, preprocessor_params, cache_dir=cache_dir
    )
    files_after_first = list(cache_dir.glob("*.pkl"))
    second = preprocess_partition(
        synthetic_pairs, preprocessor_params, cache_dir=cache_dir
    )

    assert len(files_after_first) == 1
    assert list(cache_dir.glob("*.pkl")) == files_after_first
    pd.testing.assert_frame_equal(first, second)
