"""Fábrica de clasificadores para la comparación de algoritmos (Bloque 2).

Centraliza la construcción de los algoritmos vistos en teoría y práctica
sobre la misma representación: la familia Naive Bayes (multinomial, complement
y Bernoulli), árbol de decisión, Random Forest (ensemble), kNN y MLP (red
neuronal feedforward). Añade los tres gradient boosting externos que evalúa el
enunciado de la Parte 3 (XGBoost, CatBoost y LightGBM). Todos se exponen bajo
un nombre corto que viaja en ``final_config.json``, de modo que la
configuración ganadora sea reconstruible sin conocer la clase de scikit-learn
subyacente.

Decisiones de diseño relevantes:

- Defaults explícitos con precedencia del usuario: los parámetros del
  llamador (``**params``) se mezclan sobre los valores por defecto, de modo
  que un ``classifier_params`` de ``final_config.json`` siempre gana.
- ``random_state`` es un argumento nombrado, no un parámetro de clasificador:
  se inyecta solo en los modelos estocásticos (árbol, Random Forest, MLP y los
  tres boosting). kNN y Naive Bayes son deterministas y no aceptan la semilla,
  así que jamás se les pasa.
- ``hidden_layer_sizes`` llega como lista cuando el pipeline se reconstruye
  desde JSON; para MLP se normaliza a tupla, el formato que espera
  ``MLPClassifier``.
- Los boosting externos se importan de forma perezosa, dentro de su rama: el
  catálogo y el resto de algoritmos funcionan aunque el paquete no esté
  instalado, y solo falla el clasificador que se pide. CatBoost puede devolver
  ``predict`` con forma ``(n, 1)``; se envuelve para que el pipeline siempre
  produzca un vector 1-D.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Catálogo de nombres admitidos por ``build_classifier``. ``multinomial_nb`` es
# un alias explícito de ``naive_bayes`` (mismo estimador, nombre más descriptivo
# para las tablas de comparación de algoritmos).
CLASSIFIER_NAMES = (
    "naive_bayes",
    "multinomial_nb",
    "complement_nb",
    "bernoulli_nb",
    "decision_tree",
    "random_forest",
    "knn",
    "mlp",
    "xgboost",
    "catboost",
    "lightgbm",
)

# Nombres de la familia Naive Bayes: comparten la restricción de admitir solo
# características no negativas (conteos y TF-IDF).
NAIVE_BAYES_NAMES = frozenset(
    {"naive_bayes", "multinomial_nb", "complement_nb", "bernoulli_nb"}
)

# Gradient boosting externos: dependen de paquetes opcionales y comparten el
# tratamiento de la semilla global.
BOOSTING_NAMES = frozenset({"xgboost", "catboost", "lightgbm"})

# Clase de CatBoost creada perezosamente para corregir la forma de ``predict``.
_RAVEL_CATBOOST_CLASS: type | None = None


def is_naive_bayes(name: str) -> bool:
    """Indica si un nombre de clasificador pertenece a la familia Naive Bayes.

    Args:
        name: Nombre corto del clasificador (miembro de ``CLASSIFIER_NAMES``).

    Returns:
        True si el clasificador es de la familia Naive Bayes (incluido el
        alias ``multinomial_nb``); False en cualquier otro caso.
    """
    return name in NAIVE_BAYES_NAMES


def is_boosting(name: str) -> bool:
    """Indica si un nombre corresponde a un gradient boosting externo.

    Args:
        name: Nombre corto del clasificador.

    Returns:
        True para ``xgboost``, ``catboost`` y ``lightgbm``; False en el resto.
    """
    return name in BOOSTING_NAMES


def _ravel_catboost_class() -> type:
    """Devuelve una subclase de ``CatBoostClassifier`` con ``predict`` 1-D.

    CatBoost puede devolver una matriz ``(n, 1)`` al predecir multiclase; el
    resto del pipeline (métricas, submission) espera un vector. La clase se
    crea una sola vez y se reutiliza, para que ``sklearn.clone`` reconstruya
    estimadores del mismo tipo.
    """
    global _RAVEL_CATBOOST_CLASS
    if _RAVEL_CATBOOST_CLASS is None:
        from catboost import CatBoostClassifier

        class RavelCatBoostClassifier(CatBoostClassifier):
            """CatBoost que aplana la salida de ``predict`` a 1-D."""

            def predict(self, X, *args, **kwargs):
                return np.asarray(super().predict(X, *args, **kwargs)).ravel()

        _RAVEL_CATBOOST_CLASS = RavelCatBoostClassifier
    return _RAVEL_CATBOOST_CLASS


def build_classifier(
    name: str = "random_forest",
    *,
    random_state: int = 42,
    **params,
) -> BaseEstimator:
    """Construye el estimador asociado a un nombre corto.

    Los valores por defecto son explícitos y los ``**params`` del llamador
    tienen precedencia sobre ellos, de modo que ``classifier_params`` de
    ``final_config.json`` se aplica siempre. ``params`` no se muta.

    Args:
        name: Nombre corto del clasificador; debe pertenecer a
            ``CLASSIFIER_NAMES``. ``multinomial_nb`` es alias de
            ``naive_bayes``.
        random_state: Semilla global; solo se inyecta en los clasificadores
            estocásticos (árbol, Random Forest, MLP, XGBoost, CatBoost y
            LightGBM). Si ``params`` incluye la semilla propia del modelo, ese
            valor tiene precedencia.
        **params: Hiperparámetros adicionales del estimador subyacente (por
            ejemplo ``alpha`` para Naive Bayes o ``n_estimators`` para Random
            Forest).

    Returns:
        Estimador sin ajustar correspondiente al nombre pedido.

    Raises:
        ValueError: Si ``name`` no pertenece a ``CLASSIFIER_NAMES``.
        ImportError: Si se pide un boosting cuyo paquete no está instalado.
    """
    if name not in CLASSIFIER_NAMES:
        raise ValueError(
            f"Clasificador no válido: {name!r}. "
            f"Valores válidos: {list(CLASSIFIER_NAMES)}"
        )

    if name in ("naive_bayes", "multinomial_nb"):
        settings = {"alpha": 1.0, "fit_prior": True, "class_prior": None, **params}
        return MultinomialNB(**settings)

    if name == "complement_nb":
        settings = {"alpha": 1.0, "norm": False, "fit_prior": True, **params}
        return ComplementNB(**settings)

    if name == "bernoulli_nb":
        settings = {"alpha": 1.0, "fit_prior": True, **params}
        return BernoulliNB(**settings)

    if name == "decision_tree":
        settings = {"criterion": "gini", "random_state": random_state, **params}
        return DecisionTreeClassifier(**settings)

    if name == "random_forest":
        settings = {
            "n_estimators": 100,
            "n_jobs": -1,
            "random_state": random_state,
            **params,
        }
        return RandomForestClassifier(**settings)

    if name == "knn":
        # KNeighborsClassifier es determinista y no acepta random_state: la
        # semilla global jamás se inyecta aquí.
        settings = {"n_neighbors": 5, "n_jobs": -1, **params}
        return KNeighborsClassifier(**settings)

    if name == "mlp":
        # ``hidden_layer_sizes`` puede llegar como lista (JSON de
        # final_config.json) y MLPClassifier espera una tupla de capas.
        settings = {
            "hidden_layer_sizes": (128,),
            "max_iter": 200,
            "early_stopping": True,
            "random_state": random_state,
            **params,
        }
        if isinstance(settings.get("hidden_layer_sizes"), list):
            settings["hidden_layer_sizes"] = tuple(settings["hidden_layer_sizes"])
        return MLPClassifier(**settings)

    if name == "xgboost":
        from xgboost import XGBClassifier

        settings = {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": random_state,
            **params,
        }
        return XGBClassifier(**settings)

    if name == "catboost":
        classifier_class = _ravel_catboost_class()
        settings = {
            "iterations": 300,
            "learning_rate": 0.1,
            "depth": 6,
            "loss_function": "MultiClass",
            "thread_count": -1,
            "verbose": False,
            "allow_writing_files": False,
            "random_seed": random_state,
            **params,
        }
        return classifier_class(**settings)

    # lightgbm
    from lightgbm import LGBMClassifier

    settings = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "colsample_bytree": 0.8,
        "n_jobs": -1,
        "verbose": -1,
        "random_state": random_state,
        **params,
    }
    return LGBMClassifier(**settings)
