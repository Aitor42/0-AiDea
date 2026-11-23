import json

import pandas as pd
import pytest

from experiments.scripts.compare_ollama import compare
from nli_project.translation.ollama import load_translations, translate_dataframe, translate_file


class RecordingTranslator:
    def __init__(self):
        self.calls = []

    def translate_batch(self, rows, source_language):
        self.calls.append((source_language, [row["id"] for row in rows]))
        return {
            row["id"]: {
                "premise": f"EN: {row['premise']}",
                "hypothesis": f"EN: {row['hypothesis']}",
            }
            for row in rows
        }


def test_translation_writes_new_csv_and_refuses_overwrite(corpus, tmp_path):
    source = tmp_path / "source.csv"
    destination = tmp_path / "english.csv"
    corpus = corpus.assign(id=["0001", "0002", "0003", "0004"])
    corpus.to_csv(source, index=False)
    assert translate_file(source, destination, translator=RecordingTranslator()) == destination
    with pytest.raises(FileExistsError):
        translate_file(source, destination)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["english.csv", "source.csv"]
    saved = pd.read_csv(destination, dtype={"id": str})
    pd.testing.assert_frame_equal(saved[corpus.columns], corpus)


@pytest.fixture
def corpus():
    return pd.DataFrame(
        [
            {"id": "en-1", "premise": "An English sentence", "hypothesis": "English", "lang_abv": "en", "label": 1},
            {"id": "es-1", "premise": "Una frase", "hypothesis": "Español", "lang_abv": "es", "label": 0},
            {"id": "es-2", "premise": "Otra frase", "hypothesis": "También español", "lang_abv": "es", "label": 2},
            {"id": "fr-1", "premise": "Une phrase", "hypothesis": "Français", "lang_abv": "fr", "label": 1},
        ]
    )


def test_translation_preserves_rows_labels_and_source_text(corpus):
    translator = RecordingTranslator()
    translated = translate_dataframe(corpus, translator, batch_size=2)

    assert translated.id.tolist() == corpus.id.tolist()
    assert translated.label.tolist() == corpus.label.tolist()
    assert translated.lang_abv.tolist() == corpus.lang_abv.tolist()
    assert translated.loc[0, "premise"] == "An English sentence"
    pd.testing.assert_frame_equal(translated[corpus.columns], corpus)
    assert translated.loc[0, "premise_translated"] == "An English sentence"
    assert translated.loc[1, "premise_translated"] == "EN: Una frase"
    assert translated.loc[1, "hypothesis_translated"] == "EN: Español"
    assert translator.calls == [("Spanish", ["es-1", "es-2"]), ("French", ["fr-1"])]


@pytest.mark.parametrize(
    ("response", "message"),
    [({}, "one translation per batch ID"),
     ({"es-1": {"premise": "A sentence"}}, "missing or invalid premise or hypothesis")],
)
def test_invalid_batch_does_not_publish_csv(corpus, tmp_path, monkeypatch, response, message):
    source = tmp_path / "source.csv"
    destination = tmp_path / "english.csv"
    corpus.iloc[[1]].to_csv(source, index=False)
    translator = RecordingTranslator()
    monkeypatch.setattr(translator, "translate_batch", lambda rows, language: response)

    with pytest.raises(ValueError, match=message):
        translate_file(source, destination, translator=translator)

    assert not destination.exists()


def test_duplicate_ids_are_rejected(corpus):
    duplicate = pd.concat([corpus, corpus.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="present and unique"):
        translate_dataframe(duplicate, RecordingTranslator())


def test_cached_translations_verify_source_metadata_and_english(corpus, tmp_path):
    path = tmp_path / "translated.csv"
    translated = translate_dataframe(corpus, RecordingTranslator())
    translated.to_csv(path, index=False)
    pd.testing.assert_frame_equal(load_translations(corpus, path), translated)

    altered = translated.copy()
    altered.loc[0, "premise_translated"] = "Different English text"
    altered.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="Textos de origen inglés"):
        load_translations(corpus, path)

    altered = translated.assign(label=2)
    altered.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="Filas originales"):
        load_translations(corpus, path)


def test_comparator_requires_opt_in_for_missing_translations(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "experiments.scripts.compare_ollama.translate_file",
        lambda *args, **kwargs: pytest.fail("Ollama se debe solicitar explícitamente."),
    )
    with pytest.raises(FileNotFoundError, match="generate=True"):
        compare("gemma3:4b", 6, translations=tmp_path / "missing.csv")


def test_comparator_writes_new_runs_with_equal_conditions(tmp_path, synthetic_pairs_small, monkeypatch):
    from experiments.scripts.run_experiment_matrix import DEFAULT_CLASSIFIERS

    frame = synthetic_pairs_small.drop(columns=["premise_processed", "hypothesis_processed"])
    (tmp_path / "experiments/data").mkdir(parents=True)
    (tmp_path / "experiments/results/validation/historical_split").mkdir(parents=True)
    historical = tmp_path / "experiments/results/translation/catboost_holdout"
    historical.mkdir(parents=True)
    frame.to_csv(tmp_path / "experiments/data/train.csv", index=False)
    frame.iloc[:54].to_csv(tmp_path / "experiments/results/validation/historical_split/train.csv", index=False)
    frame.iloc[54:].to_csv(tmp_path / "experiments/results/validation/historical_split/val.csv", index=False)
    translation = tmp_path / "translations.csv"
    translate_dataframe(frame, RecordingTranslator()).to_csv(translation, index=False)

    monkeypatch.setattr("experiments.scripts.compare_ollama.__file__", str(tmp_path / "experiments/scripts/compare_ollama.py"))
    monkeypatch.setattr("nli_project.training.training.SELECT_K_FEATURES", None)
    monkeypatch.setitem(DEFAULT_CLASSIFIERS, "catboost", {"iterations": 2, "thread_count": 1})
    first = compare("gemma3:4b", 6, translations=translation)
    second = compare("gemma3:4b", 6, translations=translation)

    assert first != second
    assert first.parent == second.parent == tmp_path / "experiments/.local/ollama_comparisons"
    assert historical.is_dir()
    metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    assert not metrics["use_pos_features"] and not metrics["use_relation_features"]
    assert {name: arm["n_val"] for name, arm in metrics["arms"].items()} == {
        "original": 18, "translated": 18,
    }
    predictions = pd.read_csv(first / "predictions.csv", dtype={"id": str})
    assert predictions.id.tolist() == frame.id.iloc[54:].tolist()
