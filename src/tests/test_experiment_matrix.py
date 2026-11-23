"""Tests de contratos para la matriz reproducible de experimentos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nli_project.preprocessing.parameter_variants import materialize_variants
from nli_project.translation.ollama import translate_dataframe

from experiments.scripts.run_experiment_matrix import (
    load_processed_dataset,
    run_matrix,
    validate_dataset_alignment,
)


def test_translated_variant_uses_translated_tokens_and_original_groups(tmp_path):
    raw = pd.DataFrame([{"id": "001", "premise": "Un gato", "hypothesis": "Un animal",
                         "lang_abv": "es", "label": 0}])

    class Translator:
        def translate_batch(self, rows, source_language):
            return {"001": {"premise": "A cat", "hypothesis": "An animal"}}

    translated = translate_dataframe(raw, Translator())
    variant, = materialize_variants(translated, tmp_path / "variants",
        base_params={"columns": ["premise_translated", "hypothesis_translated"]},
        variants=[{"name": "tokens", "params": {"use_pos": False, "reduction": "none"}}],
        cache_dir=tmp_path / "cache")
    frame = load_processed_dataset(Path(variant["dataset_path"]))
    assert frame.premise_processed.iloc[0] == ["a", "cat"]
    assert frame.premise.iloc[0] == "Un gato"
    assert frame.lang_abv.iloc[0] == "es"


def test_load_processed_dataset_validates_columns_and_labels(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"label": [0], "premise_processed": ["a"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="hypothesis_processed"):
        load_processed_dataset(path)


def test_dataset_alignment_rejects_changed_original_language(tmp_path):
    base = pd.DataFrame(
        {"id": ["a", "b"], "label": [0, 1], "lang_abv": ["en", "fr"]}
    )
    altered = base.copy()
    altered.loc[1, "lang_abv"] = "en"

    with pytest.raises(ValueError, match="idiomas originales"):
        validate_dataset_alignment([
            ("raw", tmp_path / "raw.pkl", base),
            ("translated", tmp_path / "translated.pkl", altered),
        ])


def test_grouped_comparison_rejects_changed_original_premise(tmp_path):
    base = pd.DataFrame({"id": ["001"], "label": [0], "lang_abv": ["es"], "premise": ["Un gato"]})
    translated = base.assign(premise="A cat")
    with pytest.raises(ValueError, match="columnas de agrupación"):
        validate_dataset_alignment(
            [("raw", tmp_path / "raw.pkl", base), ("translated", tmp_path / "translated.pkl", translated)],
            group_by_premise=True,
        )


def test_matrix_reuses_only_identical_complete_runs(tmp_path, synthetic_pairs_small, monkeypatch):
    dataset = tmp_path / "train.pkl"
    frame = synthetic_pairs_small
    frame.to_pickle(dataset)
    output = tmp_path / "runs"

    def run(**overrides):
        return run_matrix(
            [("original", dataset)], {"count": {"method": "count"}},
            ["naive_bayes"], output, n_splits=2, select_k=None, reuse=True,
            **overrides,
        )

    first, = run()
    assert not first["use_pos_features"] and not first["use_relation_features"]
    with monkeypatch.context() as context:
        context.setattr(
            "experiments.scripts.run_experiment_matrix.cross_validate",
            lambda *args, **kwargs: pytest.fail("Se repitió un resultado completo."),
        )
        assert run()[0]["experiment_id"] == first["experiment_id"]

    assert run(use_relation_features=True)[0]["use_relation_features"]
    frame.assign(premise_processed="different sample").to_pickle(dataset)
    run()
    assert len(list(output.glob("*/results.json"))) == 3


def test_matrix_does_not_publish_interrupted_results(tmp_path, synthetic_pairs_small, monkeypatch):
    dataset = tmp_path / "train.pkl"
    synthetic_pairs_small.to_pickle(dataset)
    output = tmp_path / "run"
    args = ([("original", dataset)], {"count": {"method": "count"}}, ["naive_bayes"], output)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as context:
        context.setattr("experiments.scripts.run_experiment_matrix.cross_validate", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run_matrix(*args, n_splits=2, select_k=None)

    assert not output.exists()
    assert not list(tmp_path.glob(".matrix-*"))
    assert run_matrix(*args, n_splits=2, select_k=None)
