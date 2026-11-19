"""Translate an NLI dataset to English with Ollama."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd


DEFAULT_MODEL = "gemma3:4b"
DEFAULT_BATCH_SIZE = 6

LANGUAGES = {
    "ar": "Arabic", "bg": "Bulgarian", "de": "German", "el": "Greek",
    "es": "Spanish", "fr": "French", "hi": "Hindi", "ru": "Russian",
    "sw": "Swahili", "th": "Thai", "tr": "Turkish", "ur": "Urdu",
    "vi": "Vietnamese", "zh": "Chinese",
}

TRANSLATION_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "premise": {"type": "string"},
                    "hypothesis": {"type": "string"},
                },
                "required": ["id", "premise", "hypothesis"],
            },
        }
    },
    "required": ["translations"],
}


class OllamaTranslator:
    """Small HTTP client for Ollama's local /api/chat endpoint."""

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", timeout: int = 300):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def translate_batch(self, rows: list[dict], source_language: str) -> dict[str, dict[str, str]]:
        # Use short indices so the model does not have to copy hashed IDs.
        numbered = [dict(row, id=str(i)) for i, row in enumerate(rows)]
        payload = {
            "model": self.model,
            "stream": False,
            "format": TRANSLATION_SCHEMA,
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Translate each premise and hypothesis faithfully into English. "
                        "Preserve meaning, negation, named entities, and NLI-relevant details. "
                        "Return exactly one translation for each supplied id. Do not add commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"source_language": source_language, "rows": numbered},
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        request = Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            translated = json.loads(result["message"]["content"])["translations"]
            if not isinstance(translated, list) or len(translated) != len(rows):
                raise ValueError("Ollama returned an incomplete translation batch.")
            output = {item["id"]: item for item in translated}
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"Could not contact Ollama: {error}") from error
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("Ollama returned malformed translation data.") from error

        if set(output) != {row["id"] for row in numbered}:
            raise ValueError("Ollama returned incorrect or duplicate batch IDs.")
        return {
            row["id"]: {
                "premise": output[str(index)].get("premise"),
                "hypothesis": output[str(index)].get("hypothesis"),
            }
            for index, row in enumerate(rows)
        }


TEXT_COLUMNS = ("premise", "hypothesis")
TRANSLATED_COLUMNS = ("premise_translated", "hypothesis_translated")


def load_translations(source: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Reuse a translation CSV only if it preserves the ordered source rows."""
    text_types = dict.fromkeys(("id", *TEXT_COLUMNS, *TRANSLATED_COLUMNS), str)
    translated = pd.read_csv(path, dtype=text_types, keep_default_na=False)
    required = {*source.columns, *TRANSLATED_COLUMNS}
    if missing := required - set(translated.columns):
        raise ValueError(f"{path}: faltan columnas {sorted(missing)}.")
    pd.testing.assert_frame_equal(
        source.reset_index(drop=True), translated.loc[:, source.columns],
        obj=f"Filas originales en {path}",
    )
    if translated[list(TRANSLATED_COLUMNS)].apply(
        lambda column: column.str.strip().eq("")
    ).any().any():
        raise ValueError(f"{path}: hay traducciones vacías.")
    english = source["lang_abv"].eq("en").to_numpy()
    pd.testing.assert_frame_equal(
        source.loc[english, list(TEXT_COLUMNS)].reset_index(drop=True),
        translated.loc[english, list(TRANSLATED_COLUMNS)].set_axis(TEXT_COLUMNS, axis=1)
        .reset_index(drop=True),
        obj=f"Textos de origen inglés en {path}",
    )
    return translated


def translate_dataframe(
    dataframe: pd.DataFrame,
    translator: OllamaTranslator,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Translate non-English pairs and keep all source columns in their original order."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    required = {"id", "lang_abv", *TEXT_COLUMNS}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if set(TRANSLATED_COLUMNS) & set(dataframe.columns):
        raise ValueError("The dataset already contains translation columns.")
    ids = dataframe["id"].astype(str)
    if dataframe["id"].isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("Dataset IDs must be present and unique.")
    if not dataframe["lang_abv"].isin({"en", *LANGUAGES}).all():
        raise ValueError("The dataset contains an unsupported or missing language code.")

    translations = {}
    non_english = dataframe[dataframe["lang_abv"] != "en"]
    # Keep each batch in one source language.
    for code, group in non_english.groupby("lang_abv", sort=False):
        records = group[["id", *TEXT_COLUMNS]].astype({"id": str}).to_dict("records")
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            result = translator.translate_batch(batch, LANGUAGES[code])
            if not isinstance(result, dict) or set(result) != {row["id"] for row in batch}:
                raise ValueError("The response must contain one translation per batch ID.")
            for pair in result.values():
                if not isinstance(pair, dict) or any(
                    not isinstance(pair.get(column), str) or not pair[column].strip()
                    for column in TEXT_COLUMNS
                ):
                    raise ValueError("The response contains a missing or invalid premise or hypothesis.")
            translations.update(result)

    output = dataframe.copy()
    # Build the new columns in input order, including the original English text.
    for column, target in zip(TEXT_COLUMNS, TRANSLATED_COLUMNS):
        output[target] = [
            translations[identifier][column] if identifier in translations else original
            for identifier, original in zip(ids, dataframe[column])
        ]
    return output


def translate_file(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    translator: OllamaTranslator | None = None,
) -> Path:
    """Translate a dataset and publish it as a new CSV."""
    output_path = Path(output_csv)
    if output_path.exists():
        raise FileExistsError(f"The output file already exists: {output_path}")
    dataframe = pd.read_csv(input_csv, dtype={"id": str, "premise": str, "hypothesis": str}, keep_default_na=False)
    if translator is None:
        translator = OllamaTranslator(model)
    result = translate_dataframe(dataframe, translator, batch_size=batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish the complete CSV without overwriting another run's output.
    with TemporaryDirectory(dir=output_path.parent) as directory:
        temporary = Path(directory) / output_path.name
        result.to_csv(temporary, index=False)
        os.link(temporary, output_path)
    return output_path
