"""Evalúa un clasificador NLI ajustado sobre matrices ya construidas."""

from collections.abc import Mapping
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_consistent_length


class GenericNLIEvaluator:
    """Predice y calcula métricas de clasificación de etiqueta única."""

    def __init__(self, label_names: Mapping[Any, str] | None = None):
        """Usa las etiquetas y el orden del mapa si se proporciona."""
        self.label_names = dict(label_names or {})

    def predict(self, model, X) -> np.ndarray:
        """Devuelve una etiqueta por fila de la matriz."""
        predictions = np.asarray(model.predict(X))
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            predictions = predictions[:, 0]
        if predictions.ndim != 1:
            raise ValueError("El modelo debe devolver una etiqueta por muestra.")
        check_consistent_length(X, predictions)
        return predictions

    def evaluate(self, model, dataset) -> dict[str, Any]:
        """Evalúa un par (X, y) y conserva las predicciones para revisar errores.
        """
        X, y_true = dataset
        check_consistent_length(X, y_true)

        start = perf_counter()
        predictions = self.predict(model, X)
        elapsed = perf_counter() - start
        return {**self.evaluate_predictions(y_true, predictions), "predict_time_sec": elapsed}

    def evaluate_predictions(self, y_true, predictions) -> dict[str, Any]:
        """Calcula las mismas métricas para predicciones ya reunidas, como las OOF.

        Con un mapa de clases, F1 macro incluye todas las clases indicadas y
        la matriz de confusión mantiene ese mismo orden.
        """
        predictions = np.asarray(predictions)
        check_consistent_length(y_true, predictions)

        observed = unique_labels(y_true, predictions)
        if self.label_names and not set(observed).issubset(self.label_names):
            raise ValueError("El mapa de clases debe incluir todas las etiquetas observadas.")
        labels = list(self.label_names) if self.label_names else observed.tolist()
        names = [self.label_names.get(label, label) for label in labels]

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, predictions, labels=labels, zero_division=0
        )
        per_class = pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support},
            index=pd.Index(names, name="class"),
        )
        matrix = pd.DataFrame(
            confusion_matrix(y_true, predictions, labels=labels),
            index=pd.Index(names, name="actual"),
            columns=pd.Index(names, name="predicted"),
        )
        return {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "f1_macro": float(f1.mean()),
            "f1_weighted": float(np.average(f1, weights=support)),
            "per_class": per_class,
            "confusion_matrix": matrix,
            "predictions": predictions,
        }
