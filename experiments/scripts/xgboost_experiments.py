"""Banco de experimentos para XGBoost (solo investigación).

Gradient boosting por niveles: cada árbol corrige el residuo de los anteriores.
Se entrena con la misma representación y los mismos folds que
``rf_experiments.py``, de modo que la comparación con el Random Forest es
directa (holdout reservado, sin tocar).

Ejes que se exploran:

- ``n_estimators`` (200/500/1000): con boosting, más árboles sin bajar la tasa
  de aprendizaje suele sobreajustar.
- ``learning_rate`` (0,05/0,1/0,2) y ``max_depth`` (4/6/8): el compromiso
  clásico entre número de árboles y profundidad.
- Submuestreo de filas (``subsample``) y de columnas (``colsample_bytree``),
  que es el mecanismo de regularización estocástica del boosting.
- Interacción con chi² y con el tamaño del vocabulario.

Uso (desde la raíz del proyecto):

    python -m experiments.scripts.xgboost_experiments --list
    python -m experiments.scripts.xgboost_experiments base lr_0.05 depth_8
"""

from __future__ import annotations

import sys

from experiments.scripts.model_experiments import REPRESENTATION_EXPERIMENTS, run_cli

MODEL = "xgboost"

MODEL_EXPERIMENTS = {
    "est_200": dict(clf={"n_estimators": 200}),
    "est_500": dict(clf={"n_estimators": 500}),
    "est_1000": dict(clf={"n_estimators": 1000}),
    "depth_4": dict(clf={"max_depth": 4}),
    "depth_8": dict(clf={"max_depth": 8}),
    "lr_0.05": dict(clf={"learning_rate": 0.05}),
    "lr_0.2": dict(clf={"learning_rate": 0.2}),
    "subsample_0.6": dict(clf={"subsample": 0.6}),
    "colsample_0.6": dict(clf={"colsample_bytree": 0.6}),
    "lr_0.05_est_500": dict(clf={"learning_rate": 0.05, "n_estimators": 500}),
}

EXPERIMENTS = {**REPRESENTATION_EXPERIMENTS, **MODEL_EXPERIMENTS}


if __name__ == "__main__":
    run_cli(MODEL, EXPERIMENTS)
    sys.exit(0)
