"""Banco de experimentos para Multinomial Naive Bayes (solo investigación).

Mismo protocolo que ``rf_experiments.py``: selección solo sobre el split de
entrenamiento, CV estratificada de 5 folds con la misma semilla y holdout
reservado intacto.

Conceptos de teoría (L04/S04) que se exploran:

- ``alpha``: suavizado de Laplace. Con TF-IDF, valores bajos sobreajustan a
  términos raros (el vocabulario tiene ~19.000 features) y valores altos
  aplanan las probabilidades.
- ``fit_prior``: si se estiman las probabilidades a priori de clase o se usa
  una uniforme. En este dataset las clases están casi equilibradas, así que se
  espera poco efecto.
- Selección chi² y tamaño de vocabulario (``min_df``): la familia Naive Bayes
  exige entradas no negativas, de modo que los métodos con signo
  (``tfidf_svd``, ``embeddings``) quedan fuera y se evalúan conteos, TF-IDF y
  ``pair_diff``.

Uso (desde la raíz del proyecto):

    python -m experiments.scripts.naive_bayes_experiments --list
    python -m experiments.scripts.naive_bayes_experiments base alpha_0.5 alpha_3
"""

from __future__ import annotations

import sys

from experiments.scripts.model_experiments import REPRESENTATION_EXPERIMENTS, run_cli

MODEL = "naive_bayes"

MODEL_EXPERIMENTS = {
    "alpha_0.1": dict(clf={"alpha": 0.1}, note="suavizado mínimo: sobreajuste"),
    "alpha_0.5": dict(clf={"alpha": 0.5}),
    "alpha_3": dict(clf={"alpha": 3.0}),
    "alpha_10": dict(clf={"alpha": 10.0}, note="suavizado fuerte"),
    "fit_prior_true": dict(clf={"fit_prior": True}),
    "count_alpha_0.5": dict(
        vec={"method": "count"}, clf={"alpha": 0.5}, note="conteos + suavizado bajo"
    ),
    "select_500_alpha_0.5": dict(select_k=500, clf={"alpha": 0.5}),
    "mindf_20_alpha_0.5": dict(vec={"min_df": 20}, clf={"alpha": 0.5}),
    "ngram_12_alpha_1": dict(vec={"word_ngram_range": (1, 2)}, clf={"alpha": 1.0}),
}

EXPERIMENTS = {**REPRESENTATION_EXPERIMENTS, **MODEL_EXPERIMENTS}


if __name__ == "__main__":
    run_cli(MODEL, EXPERIMENTS)
    sys.exit(0)
