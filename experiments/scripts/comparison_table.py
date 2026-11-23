"""Tabla comparativa de los mejores experimentos de cada modelo.

Lee los JSONL que escriben los bancos por modelo
(``<modelo>_experiments_results.jsonl``) más el histórico del Random Forest
(``experiments/results/random_forest/experiments.jsonl``), se queda con la última ejecución de cada
configuración y construye dos tablas:

- ``experiments/.local/classical_models/model_comparison.csv``: mejor configuración de cada clasificador según
  accuracy media de la validación cruzada (desempate por macro-F1).
- ``experiments/.local/classical_models/model_comparison_base.csv``: la fila ``base`` de cada modelo, es decir, la
  misma representación y los mismos folds cambiando solo el clasificador.

Uso (desde la raíz del proyecto):

    python -m experiments.scripts.comparison_table
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import experiments.scripts.model_experiments as me
from experiments.scripts.rf_experiments import HISTORY as RF_HISTORY, RESULTS as RF_RESULTS

MODELS = (
    "naive_bayes",
    "complement_nb",
    "bernoulli_nb",
    "decision_tree",
    "knn",
    "mlp",
    "xgboost",
    "catboost",
    "lightgbm",
)

COLUMNS = (
    "model",
    "config",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "select_k",
    "fit_seconds_mean",
    "params",
    "source",
)

# Orden de columnas de los CSV; las que solo existen en el histórico del
# Random Forest quedan fuera para que la tabla sea homogénea.
CSV_ORDER = (
    "model",
    "config",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "select_k",
    "use_pos_features",
    "use_relation_features",
    "fit_seconds_mean",
    "params",
    "source",
    "note",
)


def _records(path: Path) -> list[dict]:
    """Filas JSON de un JSONL, ignorando líneas vacías."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_model_frames() -> pd.DataFrame:
    """Concatena los resultados de todos los bancos por modelo."""
    frames = []
    for model in MODELS:
        rows = me.load_results(model).to_dict("records")
        if not rows:
            continue
        frame = pd.DataFrame(rows).drop_duplicates(subset="name", keep="last")
        frame["model"] = model
        frame["config"] = frame["name"]
        frame["source"] = f"{model}_experiments.py"
        frame["params"] = frame["classifier_params"].map(
            lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
            if isinstance(value, dict)
            else str(value)
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_random_forest_frame() -> pd.DataFrame:
    """Resultados del banco del Random Forest (sin las repeticiones ``@seed``)."""
    rows = [
        row
        for path in (RF_HISTORY, RF_RESULTS)
        for row in _records(path)
        if "@" not in str(row.get("name", ""))
    ]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).drop_duplicates(subset="name", keep="last")
    frame["model"] = "random_forest"
    frame["config"] = frame["name"]
    frame["source"] = "rf_experiments.py"
    frame["classifier_params"] = frame["random_forest"]
    frame["params"] = frame["random_forest"].map(
        lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
        if isinstance(value, dict)
        else str(value)
    )
    return frame


def best_per_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Mejor fila de cada modelo por accuracy (desempate por macro-F1)."""
    ordered = frame.sort_values(
        ["accuracy_mean", "macro_f1_mean"], ascending=False
    )
    return ordered.drop_duplicates(subset="model", keep="first").sort_values(
        "accuracy_mean", ascending=False
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Escribe la tabla con el orden de columnas fijo y sin columnas vacías."""
    columns = [column for column in CSV_ORDER if column in frame.columns]
    view = frame.loc[:, columns].dropna(axis=1, how="all")
    path.parent.mkdir(parents=True, exist_ok=True)
    view.to_csv(path, index=False)


def _show(frame: pd.DataFrame) -> None:
    """Imprime la tabla en Markdown con las métricas redondeadas."""
    view = frame.loc[:, list(COLUMNS)].copy()
    for column in ("accuracy_mean", "accuracy_std", "macro_f1_mean", "fit_seconds_mean"):
        view[column] = view[column].astype(float).round(4)
    view["select_k"] = view["select_k"].map(
        lambda value: "todas" if pd.isna(value) else str(int(float(value)))
    )
    print("| " + " | ".join(view.columns) + " |")
    print("|" + "|".join(["---"] * len(view.columns)) + "|")
    for row in view.itertuples(index=False):
        print("| " + " | ".join(str(value) for value in row) + " |")


def main() -> None:
    """Construye y escribe las dos tablas comparativas."""
    frames = [frame for frame in (load_model_frames(), load_random_forest_frame()) if not frame.empty]
    if not frames:
        raise SystemExit("No hay resultados: ejecuta antes run_all_experiments.py")

    frame = pd.concat(frames, ignore_index=True)
    if "use_pos_features" in frame:
        frame["use_pos_features"] = frame["use_pos_features"].fillna(True)

    best = best_per_model(frame)
    _write_csv(best, me.ROOT / "experiments/.local/classical_models/model_comparison.csv")

    # La fila ``base`` del banco histórico del Random Forest es la
    # configuración previa a las mejoras (whitespace, bigramas, min_df=50), así
    # que no pertenece a la comparación de "misma representación".
    base = frame[
        (frame["config"] == "base") & (frame["model"] != "random_forest")
    ].sort_values("accuracy_mean", ascending=False)
    _write_csv(base, me.ROOT / "experiments/.local/classical_models/model_comparison_base.csv")

    print(f"ejecuciones registradas: {len(frame)} configuraciones únicas")
    print(f"modelos con resultados: {frame['model'].nunique()}\n")

    print("## Mejor configuración por modelo (CV 5-fold, seed 42)\n")
    _show(best)

    if not base.empty:
        print("\n## Misma representación, distinto clasificador (fila base)\n")
        _show(base)
        print(
            "\nNota: el Random Forest queda fuera de esta tabla porque la fila "
            "``base`` de su banco histórico es la configuración anterior a las "
            "mejoras; su equivalente actual está en la tabla anterior."
        )

    print("\nEscrito: model_comparison.csv y model_comparison_base.csv")


if __name__ == "__main__":
    main()
