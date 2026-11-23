"""Contratos del generador de variantes de preprocesado."""

from pathlib import Path

import pandas as pd
import pytest

import nli_project.preprocessing.parameter_variants as parameter_variants
import nli_project.preprocessing.preprocessing as preprocessing


def test_grid_checks_budget_before_materializing():
    axes = {"tokenizer": ["whitespace", "spacy"], "reduction": ["none", "snowball"]}
    with pytest.raises(ValueError, match="límite"):
        parameter_variants.grid_variants(axes, max_combinations=3)
    grid = parameter_variants.grid_variants(axes, max_combinations=4)
    assert len({tuple(row["params"].values()) for row in grid}) == 4


def test_catalog_is_small_stable_and_deduplicates_final_configs():
    first = parameter_variants.iter_variants()
    second = parameter_variants.iter_variants()

    assert len(first) == 6
    assert [item["variant_id"] for item in first] == [
        item["variant_id"] for item in second
    ]
    assert len({item["variant_id"] for item in first}) == len(first)
    assert all(item["params"]["language"] is None for item in first)

    duplicated = parameter_variants.iter_variants(
        variants=[
            {"name": "one", "params": {"reduction": "none"}},
            {"name": "same-config", "params": {"reduction": "none"}},
        ]
    )
    assert len(duplicated) == 1


def test_materialize_reuses_dataset_and_preserves_rows_and_labels(
    monkeypatch, tmp_path
):
    source = pd.DataFrame(
        {
            "id": ["a", "b"],
            "premise": ["one", "dos"],
            "hypothesis": ["two", "tres"],
            "lang_abv": ["en", "es"],
            "label": [0, 2],
        }
    )
    calls = []

    def fake_preprocess(frame, preprocessor_params, cache_dir):
        calls.append(dict(preprocessor_params))
        result = frame.copy()
        result["premise_processed"] = ["one", "dos"]
        result["hypothesis_processed"] = ["two", "tres"]
        return result

    monkeypatch.setattr(preprocessing, "preprocess_partition", fake_preprocess)
    variant = [{"name": "baseline", "params": {"reduction": "none"}}]
    first = parameter_variants.materialize_variants(
        source, tmp_path / "variants", variants=variant, cache_dir=tmp_path / "cache"
    )
    second = parameter_variants.materialize_variants(
        source, tmp_path / "variants", variants=variant, cache_dir=tmp_path / "cache"
    )

    loaded = parameter_variants.load_variant_dataset(first[0])
    assert len(calls) == 1
    assert first == second
    assert loaded["id"].tolist() == source["id"].tolist()
    assert loaded["label"].tolist() == source["label"].tolist()
    dataset_path = Path(first[0]["dataset_path"])
    assert (dataset_path.parent / "manifest.json").exists()


def test_materialize_rejects_a_language_filter_that_drops_rows(monkeypatch, tmp_path):
    source = pd.DataFrame(
        {
            "id": ["a", "b"],
            "premise": ["one", "dos"],
            "hypothesis": ["two", "tres"],
            "lang_abv": ["en", "es"],
            "label": [0, 2],
        }
    )

    def filtered(frame, preprocessor_params, cache_dir):
        return frame.iloc[:1].copy()

    monkeypatch.setattr(preprocessing, "preprocess_partition", filtered)
    with pytest.raises(ValueError, match="filtró filas"):
        parameter_variants.materialize_variants(
            source,
            tmp_path / "variants",
            variants=[{"name": "english-only", "params": {"language": "en"}}],
        )
