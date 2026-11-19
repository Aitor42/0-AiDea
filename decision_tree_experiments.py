"""Banco de experimentos para el árbol de decisión (solo investigación).

El árbol es el modelo base de los ensembles (S05): un único árbol sin poda
sobreajusta con 19.000 features y suele quedarse muy por debajo del bosque,
pero sirve de referencia y de control de varianza. Se evalúa con el mismo
protocolo que ``rf_experiments.py``: CV de 5 folds sobre el split de
entrenamiento y holdout reservado para la confirmación.

Ejes que se exploran:

- Profundidad (``max_depth`` 10/20/50/``None``) y tamaño mínimo de hoja
  (``min_samples_leaf`` 1/5/10/20): controlan la poda, que es el mecanismo
  básico contra el sobreajuste.
- Criterio de partición (``gini`` frente a ``entropy``): la diferencia es
  pequeña en la práctica, pero es el eje clásico de la teoría.
- ``min_samples_split``, ``class_weight="balanced"`` y la interacción con la
  selección chi².

Uso (desde la raíz del proyecto):

    python decision_tree_experiments.py --list
    python decision_tree_experiments.py base depth_10 leaf_10 entropy
"""

from __future__ import annotations

import sys

from model_experiments import REPRESENTATION_EXPERIMENTS, run_cli

MODEL = "decision_tree"

MODEL_EXPERIMENTS = {
    "depth_10": dict(clf={"max_depth": 10}),
    "depth_50": dict(clf={"max_depth": 50}),
    "depth_none": dict(clf={"max_depth": None}, note="sin poda por profundidad"),
    "leaf_1": dict(clf={"min_samples_leaf": 1}),
    "leaf_10": dict(clf={"min_samples_leaf": 10}),
    "leaf_20": dict(clf={"min_samples_leaf": 20}),
    "split_2": dict(clf={"min_samples_split": 2}),
    "entropy": dict(clf={"criterion": "entropy"}),
    "balanced": dict(clf={"class_weight": "balanced"}),
    "select_500_leaf_1": dict(select_k=500, clf={"min_samples_leaf": 1}),
    "depth_10_leaf_20": dict(clf={"max_depth": 10, "min_samples_leaf": 20}),
}

EXPERIMENTS = {**REPRESENTATION_EXPERIMENTS, **MODEL_EXPERIMENTS}


if __name__ == "__main__":
    run_cli(MODEL, EXPERIMENTS)
    sys.exit(0)
