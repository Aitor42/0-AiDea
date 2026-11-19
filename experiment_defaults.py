"""Hiperparámetros de referencia de los bancos de experimentos."""

# Especificaciones de hiperparámetros por algoritmo. No incluyen la semilla:
# la semilla global se inyecta al construir cada pipeline. ``hidden_layer_sizes``
# va como lista JSON y ``build_classifier`` la normaliza a tupla para MLP.
CLASSIFIER_SPECS = {
    "naive_bayes": {"alpha": 1.0, "fit_prior": False},
    "complement_nb": {"alpha": 3.0, "norm": True, "fit_prior": False},
    "bernoulli_nb": {"alpha": 1.0, "fit_prior": False},
    "decision_tree": {"max_depth": 20, "min_samples_leaf": 5},
    "random_forest": {
        "n_estimators": 1000,
        "max_depth": 50,
        "min_samples_leaf": 5,
        "min_samples_split": 5,
        "max_features": "sqrt",
        "criterion": "gini",
    },
    "knn": {"n_neighbors": 50, "weights": "distance", "metric": "cosine"},
    "mlp": {
        "hidden_layer_sizes": [512],
        "alpha": 1e-4,
        "batch_size": 256,
        "max_iter": 50,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 5,
    },
    "xgboost": {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
    },
    "catboost": {
        "iterations": 300,
        "learning_rate": 0.1,
        "depth": 6,
        "loss_function": "MultiClass",
        "verbose": False,
        "allow_writing_files": False,
    },
    "lightgbm": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "colsample_bytree": 0.8,
        "verbose": -1,
    },
}
