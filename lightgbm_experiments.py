"""Banco de experimentos para LightGBM (solo investigación).

Gradient boosting por hojas (leaf-wise) con histogramas: crece el árbol por la
hoja de mayor ganancia, lo que da modelos más profundos con menos nodos. Se
evalúa con la representación y los folds de ``rf_experiments.py`` y el holdout
reservado sigue intacto.

Ejes que se exploran:

- ``num_leaves`` (15/31/63), que es el control real de capacidad en leaf-wise
  (``max_depth`` solo actúa como tope adicional).
- ``n_estimators`` (200/500/1000) y ``learning_rate`` (0,05/0,1).
- ``colsample_bytree`` como regularización estocástica.
- Interacción con chi² y con el tamaño del vocabulario.

Uso (desde la raíz del proyecto):

    python lightgbm_experiments.py --list
    python lightgbm_experiments.py base leaves_15 leaves_63
"""

from __future__ import annotations

import sys

from model_experiments import REPRESENTATION_EXPERIMENTS, run_cli

MODEL = "lightgbm"

MODEL_EXPERIMENTS = {
    "est_200": dict(clf={"n_estimators": 200}),
    "est_500": dict(clf={"n_estimators": 500}),
    "est_1000": dict(clf={"n_estimators": 1000}),
    "leaves_15": dict(clf={"num_leaves": 15}),
    "leaves_63": dict(clf={"num_leaves": 63}),
    "lr_0.1": dict(clf={"learning_rate": 0.1}),
    "colsample_0.6": dict(clf={"colsample_bytree": 0.6}),
    "max_depth_8": dict(clf={"max_depth": 8}),
    "leaves_15_est_500": dict(clf={"num_leaves": 15, "n_estimators": 500}),
}

EXPERIMENTS = {**REPRESENTATION_EXPERIMENTS, **MODEL_EXPERIMENTS}


if __name__ == "__main__":
    run_cli(MODEL, EXPERIMENTS)
    sys.exit(0)
