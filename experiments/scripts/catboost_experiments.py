"""Banco de experimentos para CatBoost (solo investigación).

Gradient boosting ordenado (ordered boosting), pensado para reducir el sesgo
de los estimadores sobre pocos datos. Se evalúa con la representación y los
folds de ``rf_experiments.py``, con el holdout reservado y sin tocar.

El estimador que construye ``classifiers.build_classifier`` aplana la salida
de ``predict`` a 1-D, necesario porque CatBoost puede devolver ``(n, 1)`` en
multiclase.

Ejes que se exploran:

- ``iterations`` (200/500/1000) y ``learning_rate`` (0,05/0,1/0,2).
- ``depth`` (4/6/8) y regularización ``l2_leaf_reg``.
- Interacción con chi² y con ``min_df``.

Uso (desde la raíz del proyecto):

    python -m experiments.scripts.catboost_experiments --list
    python -m experiments.scripts.catboost_experiments base depth_4 lr_0.05
"""

from __future__ import annotations

import sys

from experiments.scripts.model_experiments import REPRESENTATION_EXPERIMENTS, run_cli

MODEL = "catboost"

MODEL_EXPERIMENTS = {
    "iter_200": dict(clf={"iterations": 200}),
    "iter_500": dict(clf={"iterations": 500}),
    "iter_1000": dict(clf={"iterations": 1000}),
    "depth_4": dict(clf={"depth": 4}),
    "depth_8": dict(clf={"depth": 8}),
    "lr_0.05": dict(clf={"learning_rate": 0.05}),
    "lr_0.2": dict(clf={"learning_rate": 0.2}),
    "l2_3": dict(clf={"l2_leaf_reg": 3.0}),
    "l2_10": dict(clf={"l2_leaf_reg": 10.0}, note="regularización fuerte"),
    "lr_0.05_iter_500": dict(clf={"learning_rate": 0.05, "iterations": 500}),
}

EXPERIMENTS = {**REPRESENTATION_EXPERIMENTS, **MODEL_EXPERIMENTS}


if __name__ == "__main__":
    run_cli(MODEL, EXPERIMENTS)
    sys.exit(0)
