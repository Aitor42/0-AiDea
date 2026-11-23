"""Banco de experimentos para el MLP (solo investigación).

Red feedforward entrenada con backpropagation (S07): es el modelo con más
hiperparámetros de la comparación y el más caro por ajuste, así que el banco se
centra en los ejes que de verdad mueven la accuracy sobre esta representación.
Protocolo idéntico al de ``rf_experiments.py``: CV de 5 folds sobre el split de
entrenamiento y holdout reservado.

Ejes que se exploran:

- Arquitectura: una capa de 128/256/512 unidades frente a dos capas
  (512, 256).
- Regularización L2 (``alpha``), tamaño de lote, tasa de aprendizaje y
  ``early_stopping`` (con ``validation_fraction`` interno, que no toca el fold
  de test).
- Presupuesto de features: con 300 features chi² el MLP ve una entrada
  pequeña, así que se compara con `k=500/1000` y sin selección.

Uso (desde la raíz del proyecto):

    python -m experiments.scripts.mlp_experiments --list
    python -m experiments.scripts.mlp_experiments base hidden_256 alpha_1e-3
"""

from __future__ import annotations

import sys

from experiments.scripts.model_experiments import REPRESENTATION_EXPERIMENTS, run_cli

MODEL = "mlp"

MODEL_EXPERIMENTS = {
    "hidden_128": dict(clf={"hidden_layer_sizes": [128]}),
    "hidden_256": dict(clf={"hidden_layer_sizes": [256]}),
    "hidden_512_256": dict(
        clf={"hidden_layer_sizes": [512, 256]}, note="dos capas ocultas"
    ),
    "alpha_1e-3": dict(clf={"alpha": 1e-3}),
    "alpha_1e-2": dict(clf={"alpha": 1e-2}, note="regularización fuerte"),
    "batch_512": dict(clf={"batch_size": 512}),
    "lr_1e-3": dict(clf={"learning_rate_init": 1e-3}),
    "no_early_stopping": dict(
        clf={"early_stopping": False}, note="sin parada temprana"
    ),
    "select_500_hidden_512": dict(select_k=500, clf={"hidden_layer_sizes": [512]}),
    "select_1000_hidden_256": dict(
        select_k=1000, clf={"hidden_layer_sizes": [256]}
    ),
}

EXPERIMENTS = {**REPRESENTATION_EXPERIMENTS, **MODEL_EXPERIMENTS}


if __name__ == "__main__":
    run_cli(MODEL, EXPERIMENTS)
    sys.exit(0)
