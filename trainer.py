"""Entrena clasificadores NLI con un grid y particiones ya vectorizadas.

El código que llama al trainer prepara los datos. Este módulo compara las
configuraciones del estimador sobre los mismos folds y reajusta la ganadora.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import ParameterGrid
from sklearn.utils.validation import check_consistent_length


# X es la matriz devuelta por el builder e y contiene las etiquetas en el mismo orden.
# La matriz puede ser densa o dispersa, según el estimador que la reciba.
Dataset = tuple[Any, Any]
METRICS = ("accuracy", "f1_macro", "f1_weighted", "log_loss")


def _compute_step_metrics(model, X, y_true, *, y_pred=None) -> dict[str, float]:
    """Mide el rendimiento de un modelo ajustado sobre un dataset.

    F1 macro da el mismo peso a cada clase. F1 ponderado usa su número de
    muestras. Log loss queda en NaN cuando faltan probabilidades o el modelo
    carece de la información de clases necesaria para calcularla.
    """
    if y_pred is None:
        y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "log_loss": np.nan,
    }
    if hasattr(model, "predict_proba"):
        classes = np.asarray(model.classes_)
        # El fold puede contener una sola clase observada. Las clases del modelo
        # permiten calcular la pérdida si conoce todas las etiquetas del fold.
        if len(classes) >= 2 and np.isin(y_true, classes).all():
            # Scikit-learn espera las columnas de probabilidad ordenadas por clase.
            order = np.argsort(classes)
            probabilities = model.predict_proba(X)[:, order]
            metrics["log_loss"] = float(
                log_loss(y_true, probabilities, labels=classes[order])
            )
    return metrics


class GenericNLITrainer:
    """Compara configuraciones del mismo estimador sobre folds preparados."""

    def __init__(
        self,
        scoring: str = "f1_macro",
        random_state: int | None = 42,
        verbose: bool = True,
    ):
        """Configura la métrica de selección, la semilla y los mensajes de progreso.

        La semilla se aplica al estimador cuando expone un ``random_state`` sin
        asignar. Quien prepara las particiones fija la semilla del reparto.
        """
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose

    def _new_model(self, base_model, params):
        """Crea un estimador independiente para un fold o para el reajuste final."""
        # El grid puede incluir estimadores, por ejemplo, un paso de un Pipeline.
        # Se clonan junto con el modelo base para evitar compartir estado entre ajustes.
        model = clone(base_model).set_params(**clone(params, safe=False))
        # get_params(deep=True) expone también los parámetros de pasos anidados.
        seeds = {
            name: self.random_state
            for name, value in model.get_params(deep=True).items()
            if name.rsplit("__", 1)[-1] == "random_state" and value is None
        }
        return model.set_params(**seeds)

    def fit(
        self,
        base_model: Any,
        param_grid: dict[str, list] | None,
        train_dataset: Dataset,
        cv_folds: Iterable[tuple[Dataset, Dataset]],
        *,
        return_fold_predictions: bool = False,
    ) -> dict[str, Any]:
        """Evalúa un grid de scikit-learn y reajusta el ganador con todo el train.

        Cada dataset es un par ``(X, y)`` con la matriz de ``build_dataset`` y
        sus etiquetas en el mismo orden. Cada elemento de ``cv_folds`` contiene
        ``(train_fold, validation_fold)``. Para un único split se pasa una lista
        con un par. Para el reajuste final, ``train_dataset`` contiene el conjunto
        de entrenamiento completo, vectorizado con su propio builder ajustado.

        Quien llama prepara los splits, ajusta el builder en el train de cada
        fold y construye la validación con ese mismo builder. Las matrices se
        pasan directamente al estimador. Un grid vacío evalúa ``base_model``.

        ``random_state`` completa las semillas sin asignar del estimador y sus
        pasos. Se conservan las semillas explícitas del modelo y del grid.
        Para elegir por ``log_loss`` se requieren probabilidades y etiquetas
        conocidas por el modelo en todos los folds.

        Devuelve el modelo reajustado en ``best_model``, los hiperparámetros del
        grid en ``best_params`` y la media de validación ganadora en ``best_cv_score``.
        ``cv_results`` contiene una fila por combinación y ``train_metrics``
        mide el rendimiento del modelo final sobre ``train_dataset``.
        Con ``return_fold_predictions=True`` también devuelve las predicciones
        de validación de la configuración ganadora, en el orden de los folds.
        """
        if self.scoring not in METRICS:
            raise ValueError(f"La métrica {self.scoring!r} debe ser una de {METRICS}.")

        # Materializar el iterable permite pasar un generador y usar los mismos
        # folds para todas las combinaciones del grid.
        folds = tuple(cv_folds)
        if not folds:
            raise ValueError("cv_folds debe incluir al menos un par de train y validación.")

        # Comprobar que cada matriz y su vector de etiquetas tienen las mismas filas.
        check_consistent_length(*train_dataset)
        for train_fold, validation_fold in folds:
            check_consistent_length(*train_fold)
            check_consistent_length(*validation_fold)

        # ParameterGrid combina los valores de cada hiperparámetro. Con {} hay
        # una sola configuración, la que trae el estimador base.
        grid = ParameterGrid(param_grid or {})
        minimize_score = self.scoring == "log_loss"
        results_rows = []
        predictions_by_candidate = []

        for idx, params in enumerate(grid, start=1):
            start = time.perf_counter()
            fold_train_scores = []
            fold_val_scores = []
            fold_predictions = []

            for (X_train, y_train), (X_val, y_val) in folds:
                # El modelo aprende del train del fold. Medir también en ese
                # train permite comparar el rendimiento con el de validación.
                model = self._new_model(base_model, params)
                model.fit(X_train, y_train)
                fold_train_scores.append(_compute_step_metrics(model, X_train, y_train))
                y_pred = model.predict(X_val)
                fold_val_scores.append(_compute_step_metrics(model, X_val, y_val, y_pred=y_pred))
                if return_fold_predictions:
                    fold_predictions.append(y_pred)

            if return_fold_predictions:
                predictions_by_candidate.append(fold_predictions)

            # El tiempo de cada configuración incluye sus ajustes y métricas.
            # Las medias dan el mismo peso a cada fold.
            elapsed = time.perf_counter() - start
            val_scores = [metrics[self.scoring] for metrics in fold_val_scores]
            mean_val_metrics = {
                f"mean_val_{metric}": float(
                    np.mean([scores[metric] for scores in fold_val_scores])
                )
                for metric in METRICS
            }
            mean_val_score = mean_val_metrics[f"mean_val_{self.scoring}"]
            std_val_score = float(np.std(val_scores))
            mean_train_score = float(
                np.mean([metrics[self.scoring] for metrics in fold_train_scores])
            )
            # Para una pérdida se invierte el signo, de modo que un valor
            # positivo indique mejor rendimiento en train en ambos casos.
            gap = mean_train_score - mean_val_score
            results_rows.append({
                "params": params,
                f"mean_train_{self.scoring}": mean_train_score,
                f"std_val_{self.scoring}": std_val_score,
                **mean_val_metrics,
                "overfit_gap": -gap if minimize_score else gap,
                "fit_time_sec": round(elapsed, 3),
            })

            if self.verbose:
                print(
                    f"[{idx}/{len(grid)}] {params} -> "
                    f"Val {self.scoring}={mean_val_score:.4f} (±{std_val_score:.4f}) | "
                    f"Train {self.scoring}={mean_train_score:.4f} | {elapsed:.2f}s"
                )

        # Se maximiza accuracy o F1 y se minimiza log loss. En caso de empate,
        # el orden estable conserva la primera combinación del grid.
        score_column = f"mean_val_{self.scoring}"
        cv_results = pd.DataFrame(results_rows).sort_values(
            by=score_column, ascending=minimize_score, kind="stable"
        )
        best_row = cv_results.iloc[0]
        best_score = float(best_row[score_column])
        # Por ejemplo, al seleccionar por log loss con modelos sin probabilidades.
        if not np.isfinite(best_score):
            raise ValueError(f"Ninguna configuración produjo una media válida de {self.scoring}.")

        # Entrenar un nuevo clon con todo el conjunto de desarrollo una vez
        # escogidos los hiperparámetros en validación cruzada.
        best_params = best_row["params"]
        best_model = self._new_model(base_model, best_params)
        best_model.fit(*train_dataset)

        result = {
            "best_model": best_model,
            "best_params": best_params,
            "best_cv_score": best_score,
            "cv_results": cv_results.reset_index(drop=True),
            "train_metrics": _compute_step_metrics(best_model, *train_dataset),
        }
        if return_fold_predictions:
            result["fold_predictions"] = predictions_by_candidate[cv_results.index[0]]
        return result
