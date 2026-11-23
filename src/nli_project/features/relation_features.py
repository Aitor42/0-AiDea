"""Señales de relación premisa-hipótesis para NLI.

El TF-IDF de ``pair_diff`` mide solape léxico, pero al normalizar por la
longitud pierde información que en NLI es muy informativa: cuántas negaciones
lleva cada frase, cuánto más larga es una que la otra y si comparten o no los
números/entidades. ``RelationFeatures`` recupera esas señales de forma
explícita y barata: todo se calcula fila a fila, sin aprender nada del corpus,
por lo que puede aplicarse a train y a validación por separado sin fugas y es
compatible con la selección por chi² (todas las columnas son no negativas).

El transformador admite una o dos columnas. Con dos (premisa e hipótesis)
genera el bloque relacional completo de 14 features; con una sola frase
—por ejemplo, si el dataset trae únicamente premisas— genera las 3 señales
propias de esa frase (negaciones, longitud y números) y ninguna diferencia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Negaciones que cuentan como tales. Se incluyen contracciones sueltas
# ("n't") y adverbios de negación parcial (hardly, barely, without), que en
# NLI invierten la relación lógica igual que "not". El preprocesado conserva
# las negaciones (``keep_negations=True``), así que llegan hasta aquí.
NEGATIONS = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nor",
        "nowhere",
        "cannot",
        "n't",
        "without",
        "hardly",
        "barely",
    }
)

# Nombres de las 14 columnas de salida del caso de dos frases, en el mismo
# orden que ``transform``. Con una sola columna los nombres se derivan de esa
# columna (``rel_neg_<columna>``, ``rel_len_<columna>``, ``rel_num_<columna>``)
# y solo se emiten 3 features.
RELATION_FEATURE_NAMES = (
    "rel_neg_premise",
    "rel_neg_hypothesis",
    "rel_neg_abs_diff",
    "rel_neg_xor",
    "rel_len_premise",
    "rel_len_hypothesis",
    "rel_len_abs_diff",
    "rel_len_ratio",
    "rel_hypothesis_longer",
    "rel_premise_longer",
    "rel_num_premise",
    "rel_num_hypothesis",
    "rel_num_abs_diff",
    "rel_num_jaccard",
)


def _tokens(cell) -> list[str]:
    """Extrae la lista de tokens de una celda ``*_processed``.

    Acepta listas de tuplas ``(token, POS)`` —lo habitual cuando el
    preprocesado tiene ``use_pos=True``—, listas de cadenas, cadenas planas en
    las que se separa por espacios y celdas vacías (``None``/``NaN``).

    Args:
        cell: Valor de la columna procesada.

    Returns:
        Lista de tokens como cadenas; vacía si la celda no tiene texto.
    """
    if cell is None:
        return []
    if isinstance(cell, str):
        return cell.split()
    if isinstance(cell, (float, np.floating)) and np.isnan(cell):
        return []
    if isinstance(cell, (list, tuple, np.ndarray)):
        tokens = []
        for token in cell:
            if isinstance(token, (list, tuple, np.ndarray)) and len(token) >= 1:
                tokens.append(str(token[0]))
            elif token is not None:
                tokens.append(str(token))
        return tokens
    return [str(cell)]


def _count_negations(tokens: list[str]) -> int:
    """Cuenta negaciones en una lista de tokens (sin distinguir mayúsculas)."""
    return sum(
        token.lower() in NEGATIONS or token.lower().endswith("n't")
        for token in tokens
    )


def _numbers(tokens: list[str]) -> set[str]:
    """Devuelve el conjunto de tokens que contienen algún dígito."""
    return {token for token in tokens if any(char.isdigit() for char in token)}


class RelationFeatures(BaseEstimator, TransformerMixin):
    """Resume la relación entre premisa e hipótesis en 14 columnas.

    A partir de las columnas ``*_processed`` calcula, para cada par:

    - negaciones: recuento en premisa y en hipótesis, diferencia absoluta y
      XOR (solo una de las dos frases niega);
    - longitudes: tokens de cada frase, diferencia absoluta, ratio
      hipótesis/premisa y dos indicadores de cuál es más larga;
    - números: recuento en cada frase, diferencia absoluta y Jaccard de los
      conjuntos de números compartidos.

    Con una sola columna (únicamente premisas) emite las tres señales propias
    de esa frase —negaciones, longitud y números— y ninguna comparación.

    Conceptos de teoría: en NLI la contradicción suele venir de una negación
    añadida o de entidades/números incompatibles, y la neutralidad de una
    hipótesis que introduce información no cubierta por la premisa. El modelo
    recibe esas señales sin que el TF-IDF tenga que reconstruirlas.

    Attributes:
        columns: Nombres de una o dos columnas de entrada.
        columns_: Columnas efectivas fijadas en ``fit``.
        feature_names_: Nombres de las características de salida, alineados con
            el orden de columnas del array de ``transform`` (14 con dos
            columnas, 3 con una).
    """

    def __init__(self, columns=("premise_processed", "hypothesis_processed")):
        """Inicializa el transformador sin mutar los argumentos recibidos.

        Args:
            columns: Secuencia con una o dos nombres de columna del DataFrame.
                Con dos, el orden es (premisa, hipótesis).
        """
        self.columns = columns

    def _feature_names(self, columns) -> list[str]:
        """Construye los nombres de salida para las columnas efectivas."""
        if len(columns) == 1:
            column = columns[0]
            return [
                f"rel_neg_{column}",
                f"rel_len_{column}",
                f"rel_num_{column}",
            ]
        return list(RELATION_FEATURE_NAMES)

    def fit(self, X, y=None):
        """Valida las columnas y fija el estado de ajuste.

        El transformador no aprende del corpus: ``fit`` solo comprueba que las
        columnas existen para dar un error claro en vez de un ``KeyError``.

        Args:
            X: DataFrame con las columnas ``*_processed``.
            y: Ignorado (compatibilidad con la API de scikit-learn).

        Returns:
            ``self``.

        Raises:
            TypeError: Si ``X`` no es un ``pandas.DataFrame``.
            ValueError: Si ``columns`` no tiene entre una y dos entradas o
                falta alguna columna en ``X``.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un pandas.DataFrame.")
        n_columns = len(self.columns)
        if not 1 <= n_columns <= 2:
            raise ValueError(
                "columns debe contener entre 1 y 2 columnas; "
                f"recibidas {n_columns}: {list(self.columns)}"
            )
        missing = [column for column in self.columns if column not in X.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas en X: {missing}")

        self.columns_ = tuple(self.columns)
        self.feature_names_ = self._feature_names(self.columns_)
        self.n_features_in_ = len(self.columns_)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Ajusta y transforma, adaptando las columnas a las disponibles.

        Si el pipeline se declaró con dos columnas (premisa e hipótesis) pero
        ``X`` solo trae una —el caso de un dataset de premisas sueltas—, el
        ajuste usa las presentes y emite las features de esa frase. Así el
        transformador no es el que impide procesar entradas parciales; con
        todas las columnas presentes el comportamiento es el heredado.

        Args:
            X: DataFrame con las columnas ``*_processed``.
            y: Etiquetas (ignoradas; por compatibilidad con la API).
            **fit_params: Parámetros de ajuste (reenviados al camino
                heredado).

        Returns:
            Array de ``transform`` sobre las columnas efectivas.
        """
        requested = tuple(self.columns)
        available = tuple(column for column in requested if column in X.columns)
        if not available or len(available) == len(requested):
            return super().fit_transform(X, y, **fit_params)

        # Ajusta con las columnas disponibles y restaura el argumento original
        # para que ``get_params``/``clone`` sigan viendo los parámetros tal
        # cual se construyó el transformador.
        try:
            self.columns = available
            return super().fit_transform(X, y, **fit_params)
        finally:
            self.columns = requested

    def transform(self, X) -> np.ndarray:
        """Calcula las features de relación de cada fila.

        Args:
            X: DataFrame con las columnas efectivas fijadas en ``fit``.

        Returns:
            Array ``float64`` de forma ``(n_filas, 14)`` con dos columnas o
            ``(n_filas, 3)`` con una, y valores >= 0.

        Raises:
            NotFittedError: Si se llama antes de ``fit``.
        """
        check_is_fitted(self, "feature_names_")
        premise_cells = X[self.columns_[0]].tolist()
        hypothesis_cells = (
            X[self.columns_[1]].tolist() if len(self.columns_) == 2 else None
        )

        output = np.zeros((len(X), len(self.feature_names_)), dtype=np.float64)
        for row, premise_cell in enumerate(premise_cells):
            premise_tokens = _tokens(premise_cell)
            premise_negations = _count_negations(premise_tokens)
            premise_length = len(premise_tokens)
            premise_numbers = _numbers(premise_tokens)

            if hypothesis_cells is None:
                output[row] = (premise_negations, premise_length, len(premise_numbers))
                continue

            hypothesis_tokens = _tokens(hypothesis_cells[row])
            hypothesis_negations = _count_negations(hypothesis_tokens)
            hypothesis_length = len(hypothesis_tokens)
            hypothesis_numbers = _numbers(hypothesis_tokens)
            shared_numbers = premise_numbers & hypothesis_numbers
            all_numbers = premise_numbers | hypothesis_numbers

            output[row] = (
                premise_negations,
                hypothesis_negations,
                abs(premise_negations - hypothesis_negations),
                float((premise_negations > 0) != (hypothesis_negations > 0)),
                premise_length,
                hypothesis_length,
                abs(hypothesis_length - premise_length),
                hypothesis_length / (premise_length + 1.0),
                float(hypothesis_length > premise_length),
                float(premise_length > hypothesis_length),
                len(premise_numbers),
                len(hypothesis_numbers),
                abs(len(premise_numbers) - len(hypothesis_numbers)),
                len(shared_numbers) / len(all_numbers) if all_numbers else 0.0,
            )
        return output

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Devuelve los nombres de las características de salida.

        Args:
            input_features: Ignorado; los nombres son fijos.

        Returns:
            Array de cadenas con los nombres de las features efectivas: los 14
            de ``RELATION_FEATURE_NAMES`` con dos columnas, o los 3 de la
            columna presente con una.

        Raises:
            NotFittedError: Si se llama antes de ``fit``.
        """
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)
