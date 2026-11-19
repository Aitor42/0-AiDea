"""Banco de experimentos para kNN (solo investigación).

kNN no entrena: guarda las representaciones y decide por vecindad, así que su
calidad depende por completo de la métrica de distancia y de cuántas features
se conservan (con 19.000 dimensiones la maldición de la dimensionalidad domina
y los vecinos dejan de ser informativos). Se evalúa con el mismo protocolo que
``rf_experiments.py``: CV de 5 folds sobre el split de entrenamiento, holdout
intacto.

Ejes que se exploran:

- ``n_neighbors`` (5 a 200): el eje de sesgo-varianza del modelo.
- ``weights``: voto uniforme frente a ponderado por distancia.
- ``metric``: coseno (adecuado para TF-IDF normalizado) frente a euclídea y
  Manhattan.
- Efecto de la selección chi² y de la poda de vocabulario (``min_df``), que es
  lo que hace viable kNN en este espacio.

Uso (desde la raíz del proyecto):

    python knn_experiments.py --list
    python knn_experiments.py base k_25 k_100
"""

from __future__ import annotations

import sys

from model_experiments import REPRESENTATION_EXPERIMENTS, run_cli

MODEL = "knn"

MODEL_EXPERIMENTS = {
    "k_5": dict(clf={"n_neighbors": 5}),
    "k_10": dict(clf={"n_neighbors": 10}),
    "k_25": dict(clf={"n_neighbors": 25}),
    "k_100": dict(clf={"n_neighbors": 100}),
    "k_200": dict(clf={"n_neighbors": 200}),
    "uniform": dict(clf={"weights": "uniform"}),
    "metric_euclidean": dict(clf={"metric": "euclidean"}),
    "metric_manhattan": dict(clf={"metric": "manhattan"}),
    "k_10_distance": dict(clf={"n_neighbors": 10, "weights": "distance"}),
    "select_500_k_25": dict(select_k=500, clf={"n_neighbors": 25}),
    "select_1000_k_50": dict(select_k=1000, clf={"n_neighbors": 50}),
}

EXPERIMENTS = {**REPRESENTATION_EXPERIMENTS, **MODEL_EXPERIMENTS}


if __name__ == "__main__":
    run_cli(MODEL, EXPERIMENTS)
    sys.exit(0)
