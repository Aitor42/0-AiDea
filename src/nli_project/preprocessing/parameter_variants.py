"""Generate and cache reproducible NLI preprocessing variants.

Each variant changes discrete preprocessor settings. Vectorization stays
separate so the pipeline fits vocabulary and IDF within each training fold.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


# Use the project baseline first, then change one setting or a related pair.
PREPROCESSING_VARIANTS = (
    {"name": "baseline", "params": {}},
    {"name": "without_reduction", "params": {"reduction": "none"}},
    {"name": "snowball_stemming", "params": {"reduction": "snowball"}},
    {"name": "sklearn_stopwords", "params": {"stopword_method": "sklearn"}},
    {
        "name": "whitespace_no_reduction",
        "params": {"tokenizer": "whitespace", "reduction": "none"},
    },
    {"name": "strip_accents", "params": {"strip_accents": True}},
)

GRID_CATEGORIES = {
    "tokenizer": ("whitespace", "nltk", "spacy"),
    "stopword_method": ("none", "nltk", "sklearn", "custom"),
    "reduction": ("none", "porter", "lancaster", "snowball", "wordnet", "spacy_lemma"),
    "lowercase": (True, False),
    "strip_accents": (True, False),
}
PREPROCESSOR_PARAMETERS = frozenset(
    {
        "columns",
        "language",
        "language_column",
        "lowercase",
        "strip_accents",
        "tokenizer",
        "stopword_method",
        "keep_negations",
        "custom_stopwords",
        "reduction",
        "ngram_orders",
        "top_n",
        "use_pos",
    }
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _jsonable(value):
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    return value


def _variant_id(params: Mapping) -> str:
    digest = hashlib.sha256(_canonical_json(params).encode("utf-8")).hexdigest()[:12]
    return f"preprocess_{digest}"


def _effective_params(params: Mapping) -> dict:
    """Drop settings that do not affect the columns consumed by the model."""
    effective = dict(params)
    # Frequency reports and auxiliary n-gram columns do not affect model input.
    effective.pop("top_n", None)
    effective.pop("ngram_orders", None)
    if effective.get("stopword_method") == "none" or (
        effective.get("stopword_method") == "custom"
        and not effective.get("custom_stopwords")
    ):
        effective["keep_negations"] = True
    return effective


def grid_variants(
    grid: Mapping[str, list],
    *,
    max_combinations: int = 12,
    base_params: Mapping | None = None,
) -> list[dict]:
    """Expand a grid of discrete settings from ``GRID_CATEGORIES``.

    Check the size limit before building the Cartesian product. The integer
    ``top_n`` only controls frequency reports.
    """
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("La rejilla debe ser un objeto JSON con al menos un eje.")
    if isinstance(max_combinations, bool) or not isinstance(max_combinations, int):
        raise ValueError("max_combinations debe ser un entero positivo.")
    if max_combinations < 1:
        raise ValueError("max_combinations debe ser un entero positivo.")

    axes = []
    count = 1
    for name, values in grid.items():
        if name not in GRID_CATEGORIES:
            raise ValueError(
                f"El parámetro {name!r} no se puede permutar. "
                f"Ejes admitidos: {sorted(GRID_CATEGORIES)}"
            )
        if not isinstance(values, list) or not values:
            raise ValueError(f"El eje {name!r} debe ser una lista no vacía.")
        allowed = GRID_CATEGORIES[name]
        for value in values:
            if type(value) is not type(allowed[0]) or value not in allowed:
                raise ValueError(
                    f"Valor {value!r} no válido para {name!r}. "
                    f"Valores admitidos: {list(allowed)}"
                )
        if len({_canonical_json(value) for value in values}) != len(values):
            raise ValueError(f"El eje {name!r} contiene valores duplicados.")
        count *= len(values)
        if count > max_combinations:
            raise ValueError(
                f"La rejilla generaría {count} combinaciones, por encima del "
                f"límite {max_combinations}. Reduce los ejes o eleva el límite explícitamente."
            )
        axes.append((name, values))

    result = []
    for combination in itertools.product(*(values for _, values in axes)):
        overrides = dict(zip((name for name, _ in axes), combination))
        label = "_".join(f"{key}-{value}" for key, value in overrides.items())
        result.append({"name": f"grid_{label}", "params": overrides})
    return result


def iter_variants(
    base_params: Mapping | None = None,
    variants: Iterable[Mapping] | None = None,
) -> list[dict]:
    """Yield unique variants with stable IDs and complete settings.

    Merge ``base_params`` over the shared preprocessing defaults, then apply each variant's
    optional name and parameter overrides. Skip duplicate final configurations.
    """
    from nli_project.preprocessing.preprocessing import PREPROCESSING_PARAMS

    base = dict(PREPROCESSING_PARAMS)
    if base_params:
        base.update(base_params)

    rows = PREPROCESSING_VARIANTS if variants is None else variants
    result = []
    seen = set()
    for item in rows:
        if "params" in item:
            name = str(item.get("name", "custom"))
            overrides = dict(item["params"])
        else:
            # Plain dictionaries of parameter overrides are also accepted.
            name = "custom"
            overrides = dict(item)
        unknown = sorted(set(overrides).difference(PREPROCESSOR_PARAMETERS))
        if unknown:
            raise ValueError(f"Parámetros desconocidos del preprocesador: {unknown}")
        params = _jsonable({**base, **overrides})
        identity = _canonical_json(_effective_params(params))
        if identity in seen:
            continue
        seen.add(identity)
        variant_id = _variant_id(params)
        result.append({"variant_id": variant_id, "name": name, "params": params})
    return result


def _dataframe_digest(frame: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(_canonical_json([str(column) for column in frame.columns]).encode("utf-8"))
    hashes = pd.util.hash_pandas_object(frame, index=False, categorize=True)
    hasher.update(hashes.to_numpy().tobytes())
    return hasher.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as temporary:
        json.dump(value, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _validate_materialized(source: pd.DataFrame, processed: pd.DataFrame) -> None:
    """Check row alignment and confirm that the original data is preserved."""
    if len(processed) != len(source):
        raise ValueError(
            "El preprocesador filtró filas. Para comparar variantes deben conservarse "
            "todas las lenguas y filas. Usa language=None."
        )
    processed_ids = processed["id"].reset_index(drop=True)
    source_ids = source["id"].reset_index(drop=True)
    if not processed_ids.equals(source_ids):
        raise ValueError("El preprocesador alteró el orden o los identificadores de fila.")
    missing_columns = [column for column in source.columns if column not in processed]
    changed_columns = [
        column
        for column in source.columns
        if column in processed
        and not processed[column].reset_index(drop=True).equals(
            source[column].reset_index(drop=True)
        )
    ]
    if missing_columns or changed_columns:
        raise ValueError(
            "El preprocesador debe conservar los datos de entrada. "
            f"Faltan columnas: {missing_columns}; cambiaron: {changed_columns}"
        )


def materialize_variants(
    dataframe: pd.DataFrame | None = None,
    output_dir: str | Path = "experiments/.local/parameter_variants",
    *,
    base_params: Mapping | None = None,
    variants: Iterable[Mapping] | None = None,
    cache_dir: str | Path = "experiments/.local/preprocessed",
) -> list[dict]:
    """Preprocess each variant and return metadata with its dataset path.

    Load the official training data when no DataFrame is supplied. Reuse the
    preprocessing cache and key the output by configuration and input contents.
    Pickle preserves token lists and POS tags.
    """
    from nli_project.preprocessing.preprocessing import preprocess_partition

    source = (
        pd.read_csv("experiments/data/train.csv", dtype={"id": str}, keep_default_na=False)
        if dataframe is None
        else dataframe.copy()
    )
    required = {"id", "premise", "hypothesis", "lang_abv"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas para las variantes: {missing}")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    source_digest = _dataframe_digest(source)
    results = []
    for variant in iter_variants(base_params=base_params, variants=variants):
        dataset_dir = root / f"{variant['variant_id']}_{source_digest[:12]}"
        dataset_path = dataset_dir / "dataset.pkl"
        manifest_path = dataset_dir / "manifest.json"
        metadata = {
            **variant,
            "source_digest": source_digest,
            "n_rows": int(len(source)),
            "dataset_path": str(dataset_path),
            "source_columns": [str(column) for column in source.columns],
        }
        if dataset_dir.exists():
            if not manifest_path.exists() or not dataset_path.exists():
                raise ValueError(
                    f"Directorio de variante incompleto. Se conserva para revisión: {dataset_dir}"
                )
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("schema_version") != 1:
                raise ValueError(f"Versión de manifest no compatible: {manifest_path}")
            saved = existing.get("dataset", {})
            if (
                saved.get("variant_id") != metadata["variant_id"]
                or saved.get("params") != json.loads(_canonical_json(metadata["params"]))
                or saved.get("source_digest") != source_digest
                or saved.get("source_columns") != metadata["source_columns"]
                or saved.get("n_rows") != len(source)
            ):
                raise ValueError(f"El artefacto existente no coincide. Se conserva: {dataset_dir}")
            _validate_materialized(source, pd.read_pickle(dataset_path))
            results.append(metadata)
            continue

        processed = preprocess_partition(
            source,
            preprocessor_params=variant["params"],
            cache_dir=cache_dir,
        )
        _validate_materialized(source, processed)

        try:
            with tempfile.TemporaryDirectory(prefix=".variant-", dir=root) as staging:
                staging_dir = Path(staging)
                processed.to_pickle(staging_dir / "dataset.pkl")
                _atomic_json(
                    staging_dir / "manifest.json",
                    {"schema_version": 1, "dataset": metadata},
                )
                os.replace(staging_dir, dataset_dir)
        except OSError:
            # Reuse another run's output only if the completed artifact is valid.
            if not dataset_dir.exists() or not manifest_path.exists() or not dataset_path.exists():
                raise
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            saved_dataset = saved.get("dataset", {})
            if (
                saved_dataset.get("variant_id") != metadata["variant_id"]
                or saved_dataset.get("params")
                != json.loads(_canonical_json(metadata["params"]))
                or saved_dataset.get("source_digest") != source_digest
                or saved_dataset.get("source_columns") != metadata["source_columns"]
                or saved_dataset.get("n_rows") != len(source)
            ):
                raise
            _validate_materialized(source, pd.read_pickle(dataset_path))
        results.append(metadata)
    return results


def load_variant_dataset(metadata_or_path: Mapping | str | Path) -> pd.DataFrame:
    """Load the processed DataFrame from a metadata record or path."""
    path = (
        metadata_or_path["dataset_path"]
        if isinstance(metadata_or_path, Mapping)
        else metadata_or_path
    )
    return pd.read_pickle(path)
