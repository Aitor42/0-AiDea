"""Banco de experimentos para Complement Naive Bayes (solo investigación).

Complement NB estima los parámetros de cada clase con las estadísticas de las
*otras* clases, lo que lo hace menos sensible al desequilibrio y, según la
literatura, algo más robusto que MultinomialNB en texto. Aquí se comprueba con
el mismo protocolo que ``rf_experiments.py``: solo el split de entrenamiento,
CV de 5 folds con semilla fija y holdout reservado sin tocar.

Ejes que se exploran:

- ``alpha``: suavizado; el baseline del proyecto usa 3,0.
- ``norm``: si se normaliza la contribución de cada clase por la longitud de
  sus pesos (``True`` en el baseline).
- ``fit_prior``: prior estimado frente a uniforme.
- Interacción con la selección chi², ``min_df`` y n-gramas.

Uso (desde la raíz del proyecto):

    python complement_nb_experiments.py --list
    python complement_nb_experiments.py base alpha_1 norm_false
"""

from __future__ import annotations

import sys

from model_experiments import REPRESENTATION_EXPERIMENTS, run_cli

MODEL = "complement_nb"

MODEL_EXPERIMENTS = {
    "alpha_0.5": dict(clf={"alpha": 0.5}),
    "alpha_1": dict(clf={"alpha": 1.0}),
    "alpha_10": dict(clf={"alpha": 10.0}, note="suavizado fuerte"),
    "norm_false": dict(clf={"norm": False}, note="sin normalizar por longitud"),
    "fit_prior_true": dict(clf={"fit_prior": True}),
    "select_500_alpha_3": dict(select_k=500, clf={"alpha": 3.0}),
    "select_200_alpha_3": dict(select_k=200, clf={"alpha": 3.0}),
    "count_alpha_1": dict(vec={"method": "count"}, clf={"alpha": 1.0}),
}

EXPERIMENTS = {**REPRESENTATION_EXPERIMENTS, **MODEL_EXPERIMENTS}


if __name__ == "__main__":
    run_cli(MODEL, EXPERIMENTS)
    sys.exit(0)
