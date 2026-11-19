"""Banco de experimentos para Bernoulli Naive Bayes (solo investigación).

Bernoulli NB modela la *presencia* de cada término, no su frecuencia: con el
TF-IDF del proyecto binariza cualquier valor positivo, así que mide si la
palabra aparece. Es la variante que penaliza explícitamente la ausencia de un
término y por eso suele rendir peor en documentos largos, pero conviene
comprobarlo con el mismo protocolo que ``rf_experiments.py`` (CV sobre el split
de entrenamiento, holdout intacto).

Ejes que se exploran:

- ``alpha``: suavizado, especialmente sensible porque el número de features
  tras la selección es pequeño (300).
- ``fit_prior``: prior estimado frente a uniforme.
- ``count`` frente a TF-IDF binarizado y tamaño de vocabulario.

Uso (desde la raíz del proyecto):

    python bernoulli_nb_experiments.py --list
    python bernoulli_nb_experiments.py base alpha_0.5 alpha_3
"""

from __future__ import annotations

import sys

from model_experiments import REPRESENTATION_EXPERIMENTS, run_cli

MODEL = "bernoulli_nb"

MODEL_EXPERIMENTS = {
    "alpha_0.1": dict(clf={"alpha": 0.1}),
    "alpha_0.5": dict(clf={"alpha": 0.5}),
    "alpha_3": dict(clf={"alpha": 3.0}),
    "alpha_10": dict(clf={"alpha": 10.0}),
    "fit_prior_true": dict(clf={"fit_prior": True}),
    "select_500_alpha_1": dict(select_k=500, clf={"alpha": 1.0}),
    "no_select_alpha_1": dict(select_k=None, clf={"alpha": 1.0}),
}

EXPERIMENTS = {**REPRESENTATION_EXPERIMENTS, **MODEL_EXPERIMENTS}


if __name__ == "__main__":
    run_cli(MODEL, EXPERIMENTS)
    sys.exit(0)
