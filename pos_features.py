"""Ratios de etiquetas POS como características auxiliares para NLI.

El preprocesador con ``use_pos=True`` guarda las columnas ``*_processed``
como listas de tuplas ``(token, POS)``. ``POSRatioFeatures`` las resume en
frecuencias relativas por etiqueta Universal: una por columna (premisa e
hipótesis) y, cuando hay dos columnas, la diferencia absoluta entre ambas.
Esto aporta al Random Forest una señal morfológica barata y no negativa,
complementaria al TF-IDF del texto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# Etiquetas POS Universales que se convierten en características. Se excluyen
# las etiquetas casi nulas o sin señal para NLI: PUNCT (puntuación), SPACE
# (espacios), X (tokens sin clasificar), SYM (símbolos) e INTJ
# (interjecciones) aparecen con frecuencia despreciable tras el preprocesado
# o son ruido de tokenización, de modo que sus ratios serían casi siempre 0 y
# solo añadirían columnas irrelevantes al modelo.
POS_RATIO_TAGS = ("NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "AUX")


def _tag_counts(cell) -> tuple[int, dict[str, int]]:
    """Cuenta los tokens etiquetados de una celda ``*_processed``.

    Args:
        cell: Valor de la celda. Puede ser una lista de tuplas
            ``(token, POS)``, una lista de cadenas planas (datos preprocesados
            sin POS), ``None``, ``NaN`` o una cadena plana.

    Returns:
        Tupla ``(n, counts)``: número de tokens etiquetados (elementos que
        son tupla/lista de longitud >= 2, sin importar si su etiqueta tiene
        columna propia) y un diccionario ``{etiqueta: apariciones}`` con las
        etiquetas observadas.
    """
    counts: dict[str, int] = {}
    if cell is None or isinstance(cell, str):
        return 0, counts
    if isinstance(cell, (float, np.floating)) and np.isnan(cell):
        return 0, counts
    if not isinstance(cell, (list, tuple, np.ndarray)):
        return 0, counts

    n = 0
    for token in cell:
        # Solo los pares (palabra, POS) cuentan como tokens etiquetados; las
        # cadenas planas u otros formatos no aportan nada al denominador.
        if isinstance(token, (list, tuple)) and len(token) >= 2:
            n += 1
            tag = str(token[1])
            counts[tag] = counts.get(tag, 0) + 1
    return n, counts


class POSRatioFeatures(BaseEstimator, TransformerMixin):
    """Transformador de ratios POS por columna y su diferencia absoluta.

    Convierte columnas ``*_processed`` de tuplas ``(token, POS)`` en una
    matriz densa: para cada columna y etiqueta del conjunto efectivo, la
    proporción de tokens con esa etiqueta sobre el total de tokens etiquetados
    de la celda; las etiquetas fuera del conjunto restan en el denominador
    pero no generan columna. Con dos columnas añade, por etiqueta, la
    diferencia absoluta entre premisa e hipótesis.

    Attributes:
        columns: Nombres de las columnas de entrada (1 o 2).
        tags: Etiquetas POS solicitadas o ``None`` para usar ``POS_RATIO_TAGS``.
        columns_: Columnas efectivas fijadas en ``fit`` (iguales a ``columns``
            en el ajuste estricto).
        tags_: Conjunto efectivo de etiquetas fijado en ``fit``.
        feature_names_: Nombres de las características de salida, alineados
            con el orden de columnas del array de ``transform``.
    """

    def __init__(
        self,
        columns=("premise_processed", "hypothesis_processed"),
        tags=None,
    ):
        """Inicializa el transformador sin mutar los argumentos recibidos.

        Args:
            columns: Secuencia con 1 o 2 nombres de columna del DataFrame.
            tags: Secuencia de etiquetas POS; ``None`` usa ``POS_RATIO_TAGS``.
        """
        self.columns = columns
        self.tags = tags

    def _effective_tags(self) -> tuple[str, ...]:
        """Devuelve el conjunto efectivo de etiquetas (copia si viene dada)."""
        return tuple(self.tags) if self.tags is not None else POS_RATIO_TAGS

    def _feature_names(self, columns) -> list[str]:
        """Construye los nombres de salida en el orden del array de ``transform``.

        Args:
            columns: Columnas efectivas (tuple/list) sobre las que generar
                los nombres.
        """
        tags = self._effective_tags()
        names = [f"pos_{column}_{tag}" for column in columns for tag in tags]
        if len(columns) == 2:
            names.extend(f"pos_diff_{tag}" for tag in tags)
        return names

    def fit(self, X, y=None):
        """Valida las columnas y fija el estado de ajuste.

        Args:
            X: DataFrame con las columnas ``*_processed`` de tuplas
                ``(token, POS)``.
            y: Ignorado (compatibilidad con la API de scikit-learn).

        Returns:
            ``self``.

        Raises:
            TypeError: Si ``X`` no es un ``pandas.DataFrame``.
            ValueError: Si ``columns`` no tiene entre 1 y 2 entradas o falta
                alguna columna en ``X``.
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
        self.tags_ = self._effective_tags()
        self.feature_names_ = self._feature_names(self.columns_)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Ajusta y transforma, adaptando las columnas a las disponibles.

        A diferencia de ``fit`` — que exige todas las columnas declaradas —,
        si alguna columna pedida no está en ``X`` el ajuste usa solo las
        presentes (siempre que quede al menos una): así un DataFrame parcial
        (p. ej. con solo ``premise_processed``) genera sus ratios sin
        diferencias en vez de romper el pipeline. Con todas las columnas
        presentes el comportamiento es idéntico al ``fit_transform``
        heredado de ``TransformerMixin``.

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
        """Calcula los ratios POS de cada fila.

        Args:
            X: DataFrame con las columnas efectivas fijadas en ``fit``.

        Returns:
            Array ``float64`` de forma ``(n_filas, ancho)`` con valores >= 0,
            donde ``ancho`` es ``len(columns_) * n_tags`` más ``n_tags``
            adicionales de diferencias cuando hay dos columnas.

        Raises:
            NotFittedError: Si se llama antes de ``fit``.
        """
        check_is_fitted(self, "feature_names_")
        tags = self.tags_
        n_tags = len(tags)
        n_columns = len(self.columns_)
        width = n_columns * n_tags + (n_tags if n_columns == 2 else 0)
        output = np.zeros((len(X), width), dtype=np.float64)

        # Una lista de valores por columna evita el coste de iterrows con
        # celdas de listas de tuplas.
        column_cells = [X[column].tolist() for column in self.columns_]
        for row_pos in range(len(X)):
            ratios = []
            for cells in column_cells:
                n, counts = _tag_counts(cells[row_pos])
                if n > 0:
                    ratios.append(
                        np.array(
                            [counts.get(tag, 0) / n for tag in tags],
                            dtype=np.float64,
                        )
                    )
                else:
                    ratios.append(np.zeros(n_tags, dtype=np.float64))

            offset = 0
            for column_ratios in ratios:
                output[row_pos, offset : offset + n_tags] = column_ratios
                offset += n_tags
            if n_columns == 2:
                output[row_pos, offset:] = np.abs(ratios[0] - ratios[1])
        return output

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Devuelve los nombres de las características de salida.

        Args:
            input_features: Ignorado; los nombres derivan de las columnas
                efectivas fijadas en ``fit`` y de las etiquetas efectivas.

        Returns:
            Array de cadenas ``"pos_{columna}_{TAG}"`` por columna y etiqueta,
            más ``"pos_diff_{TAG}"`` cuando hay dos columnas.

        Raises:
            NotFittedError: Si se llama antes de ``fit``.
        """
        check_is_fitted(self, "feature_names_")
        return np.asarray(self.feature_names_, dtype=object)
