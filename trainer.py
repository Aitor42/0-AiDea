# trainer.py
from __future__ import annotations
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, log_loss

from dataset_utils import create_builder_for_mode


DENSE_REQUIRED_MODELS = {
    "GaussianNB",
    "HistGradientBoostingClassifier",
    "QuadraticDiscriminantAnalysis",
    "LinearDiscriminantAnalysis"
}


def _adapt_matrix_for_model(X, model):
    """Convierte a matriz densa solo si el estimador lo requiere."""
    model_name = type(model).__name__
    if model_name in DENSE_REQUIRED_MODELS and sparse.issparse(X):
        return X.toarray()
    return X


def _compute_step_metrics(model, X, y_true) -> Dict[str, float]:
    """Calcula métricas de clasificación para train o validation."""
    y_pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            metrics["log_loss"] = np.nan
    else:
        metrics["log_loss"] = np.nan
    return metrics


class GenericNLITrainer:
    """
    Loop de entrenamiento genérico con Cross-Validation + Grid Search
    integrado directamente con NLIDatasetBuilder sin data leakage.
    """

    def __init__(
        self,
        cv_splits: int = 5,
        scoring: str = "f1_macro",
        random_state: int = 42,
        verbose: bool = True
    ):
        self.cv_splits = cv_splits
        self.scoring = scoring
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        base_model: Any,
        param_grid: Dict[str, list],
        train_df: pd.DataFrame,
        preprocessor_params: Dict[str, Any],
        vectorizer_params: Dict[str, Any],
        input_mode: str = "two_vectors",
        balancer: Optional[Any] = None,
        balance_strategy: str = "joint_oversample"
    ) -> Dict[str, Any]:
        """
        Ejecuta Grid Search con StratifiedGroupKFold sobre `train_df`.

        Args:
            base_model: Estimador compatible con scikit-learn (e.g., LogisticRegression()).
            param_grid: Diccionario de hiperparámetros para explorar.
            train_df: DataFrame de entrenamiento ya filtrado (con columnas premise, hypothesis, label).
            preprocessor_params: Configuración para NLITextPreprocesser.
            vectorizer_params: Configuración para NLITextVectorizer.
            input_mode: 'two_vectors' o 'one_vector'.
            balancer: Instancia opcional de NLIClusterBalancer para aplicar dentro de cada train fold.
            balance_strategy: Estrategia de balanceo si `balancer` está activo.

        Returns:
            Diccionario con:
                - 'best_model': Modelo entrenado con los mejores hiperparámetros en todo train_df.
                - 'best_builder': NLIDatasetBuilder ajustado en todo train_df (listo para test).
                - 'best_params': Diccionario con los mejores hiperparámetros.
                - 'best_cv_score': Mejor puntuación media en validación cruzada.
                - 'cv_results': DataFrame con los resultados de todas las combinaciones del grid.
                - 'train_metrics': Métricas del modelo final sobre train_df.
        """
        prep_cfg = dict(preprocessor_params)
        # Evitamos refiltrar idioma dentro de los folds si train_df ya viene preparado
        prep_cfg["language"] = None

        sgkf = StratifiedGroupKFold(
            n_splits=self.cv_splits,
            shuffle=True,
            random_state=self.random_state
        )

        # Precomputar folds (y opcionalmente matrices si embeddings es stateless)
        is_stateless_embeddings = (vectorizer_params.get("method") == "embeddings")
        precomputed_folds = []

        if is_stateless_embeddings:
            global_builder = create_builder_for_mode(prep_cfg, vectorizer_params, input_mode)
            X_all = global_builder.build_dataset(train_df, mode="train")
            y_all = train_df["label"].to_numpy()
            for tr_idx, va_idx in sgkf.split(train_df, y_all, groups=train_df["premise"]):
                precomputed_folds.append((X_all[tr_idx], y_all[tr_idx], X_all[va_idx], y_all[va_idx]))
        else:
            for tr_idx, va_idx in sgkf.split(train_df, train_df["label"], groups=train_df["premise"]):
                tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
                va_df = train_df.iloc[va_idx].reset_index(drop=True)

                if balancer is not None:
                    clusters, _ = balancer.fit_predict_clusters(tr_df, prep_cfg)
                    tr_df = balancer.balance_dataframe(tr_df, clusters, strategy=balance_strategy)

                fold_builder = create_builder_for_mode(prep_cfg, vectorizer_params, input_mode)
                X_tr = fold_builder.build_dataset(tr_df, mode="train")
                y_tr = tr_df["label"].to_numpy()
                X_va = fold_builder.build_dataset(va_df, mode="val")
                y_va = va_df["label"].to_numpy()
                precomputed_folds.append((X_tr, y_tr, X_va, y_va))

        grid = list(ParameterGrid(param_grid)) if param_grid else [{}]
        results_rows = []
        best_score = -np.inf
        best_params = None

        for idx, params in enumerate(grid, start=1):
            t0 = time.perf_counter()
            fold_train_scores = []
            fold_val_scores = []

            for X_tr, y_tr, X_va, y_va in precomputed_folds:
                model = clone(base_model).set_params(**params)
                X_tr_adapted = _adapt_matrix_for_model(X_tr, model)
                X_va_adapted = _adapt_matrix_for_model(X_va, model)

                model.fit(X_tr_adapted, y_tr)

                tr_m = _compute_step_metrics(model, X_tr_adapted, y_tr)
                va_m = _compute_step_metrics(model, X_va_adapted, y_va)
                fold_train_scores.append(tr_m)
                fold_val_scores.append(va_m)

            elapsed = time.perf_counter() - t0
            mean_val_score = float(np.mean([m[self.scoring] for m in fold_val_scores]))
            std_val_score = float(np.std([m[self.scoring] for m in fold_val_scores]))
            mean_train_score = float(np.mean([m[self.scoring] for m in fold_train_scores]))

            row = {
                "params": params,
                f"mean_val_{self.scoring}": mean_val_score,
                f"std_val_{self.scoring}": std_val_score,
                f"mean_train_{self.scoring}": mean_train_score,
                "mean_val_accuracy": float(np.mean([m["accuracy"] for m in fold_val_scores])),
                "mean_val_f1_macro": float(np.mean([m["f1_macro"] for m in fold_val_scores])),
                "mean_val_log_loss": float(np.nanmean([m["log_loss"] for m in fold_val_scores])),
                "overfit_gap": mean_train_score - mean_val_score,
                "fit_time_sec": round(elapsed, 3),
            }
            results_rows.append(row)

            if self.verbose:
                print(
                    f"[{idx}/{len(grid)}] {params} -> "
                    f"Val {self.scoring}: {mean_val_score:.4f} (±{std_val_score:.4f}) | "
                    f"Train {self.scoring}: {mean_train_score:.4f} | {elapsed:.2f}s"
                )

            if mean_val_score > best_score:
                best_score = mean_val_score
                best_params = params

        # Reentrenamiento final con todo train_df
        final_train_df = train_df.copy()
        if balancer is not None:
            clusters, _ = balancer.fit_predict_clusters(final_train_df, prep_cfg)
            final_train_df = balancer.balance_dataframe(
                final_train_df, clusters, strategy=balance_strategy
            )

        best_builder = create_builder_for_mode(prep_cfg, vectorizer_params, input_mode)
        X_full_train = best_builder.build_dataset(final_train_df, mode="train")
        y_full_train = final_train_df["label"].to_numpy()

        best_model = clone(base_model).set_params(**best_params)
        X_full_adapted = _adapt_matrix_for_model(X_full_train, best_model)
        best_model.fit(X_full_adapted, y_full_train)

        train_metrics = _compute_step_metrics(best_model, X_full_adapted, y_full_train)
        cv_results_df = pd.DataFrame(results_rows).sort_values(
            by=f"mean_val_{self.scoring}", ascending=False
        ).reset_index(drop=True)

        return {
            "best_model": best_model,
            "best_builder": best_builder,
            "best_params": best_params,
            "best_cv_score": best_score,
            "cv_results": cv_results_df,
            "train_metrics": train_metrics,
        }