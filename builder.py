"""Wrapper para preprocesar y vectorizar datasets NLI sin data leakage.

Dependencias esperadas:
    - NLITextPreprocesser_POS.py, que define NLITextPreprocesser.
    - NLITextVectorizer.py, que define NLITextVectorizer.

El vectorizador se ajusta exclusivamente en mode='train'. Las llamadas con
mode='val' reutilizan el vocabulario, IDF, SVD o modelo ya ajustado.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from sklearn.utils.validation import check_is_fitted

from preprocesser import NLITextPreprocesser
from vectorizer import NLITextVectorizer

# Esta clase debe definirse/importarse en el mismo entorno antes de usar el builder:
# NLITextPreprocesser y NLITextVectorizer.


class NLIDatasetBuilder:
    """Coordina NLITextPreprocesser y NLITextVectorizer.

    Todas las opciones de ambos componentes se pueden configurar mediante
    ``preprocessor_params`` y ``vectorizer_params``. Esto permite utilizar
    cualquier tokenizador, filtrado, reducción, POS y método de vectorización
    que implementen las clases originales, sin duplicar su lógica.

    Args:
        preprocessor_params: Argumentos del constructor de NLITextPreprocesser.
        vectorizer_params: Argumentos del constructor de NLITextVectorizer.

    Nota:
        Si no se especifican ``vectorizer_params['columns']``, se utilizan
        automáticamente las columnas ``<columna>_processed`` del preprocesador.
        Si el preprocesador usa POS y no se indica ``use_pos`` en el vectorizador,
        se activa automáticamente para mantener la compatibilidad.
    """

    def __init__(
        self,
        preprocessor_params: Optional[Mapping[str, Any]] = None,
        vectorizer_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        preprocessor_params = dict(preprocessor_params or {})
        vectorizer_params = dict(vectorizer_params or {})

        # Crear primero el preprocesador para conocer sus columnas y configuración POS.
        self.preprocessor = NLITextPreprocesser(**preprocessor_params)

        processed_columns = tuple(
            f"{column}_processed" for column in self.preprocessor.columns
        )
        vectorizer_params.setdefault("columns", processed_columns)
        vectorizer_params.setdefault("use_pos", self.preprocessor.use_pos)

        # Las dos clases deben interpretar las tuplas POS del mismo modo.
        if bool(vectorizer_params["use_pos"]) != bool(self.preprocessor.use_pos):
            raise ValueError(
                "NLITextPreprocesser.use_pos y NLITextVectorizer.use_pos "
                "deben tener el mismo valor."
            )

        self.vectorizer = NLITextVectorizer(**vectorizer_params)
        self._is_fitted = False

    def build_dataset(self, dataframe, mode: str = "train"):
        """Devuelve los vectores del vectorizador para train o validation.

        Args:
            dataframe: DataFrame original con las columnas de texto requeridas.
            mode: ``'train'`` ajusta el vectorizador y transforma los datos;
                ``'val'`` transforma usando exclusivamente el ajuste de train.

        Returns:
            La salida original de NLITextVectorizer: normalmente una matriz
            scipy sparse para métodos Count/TF-IDF y un ndarray para embeddings
            o TF-IDF + SVD. No se convierte ni se envuelve el resultado.

        Raises:
            ValueError: Si mode no es 'train' o 'val', o si se solicita
                validation antes de ajustar con train.
        """
        if mode not in {"train", "val"}:
            raise ValueError("mode debe ser 'train' o 'val'.")

        # El preprocesamiento no aprende vocabulario ni estadísticas del corpus.
        # Se aplica de forma independiente a cada partición.
        processed_dataframe = self.preprocessor.preprocess_dataset(dataframe)

        if mode == "train":
            # Único punto en el que se ajustan vocabulario, IDF, SVD, etc. (train dataset)
            vectors = self.vectorizer.fit_transform(processed_dataframe)
            self._is_fitted = True
            return vectors

        # En validation jamás se llama a fit ni fit_transform.
        if not self._is_fitted:
            raise ValueError(
                "El builder todavía no está ajustado. Ejecuta primero "
                "build_dataset(train_dataframe, mode='train')."
            )

        check_is_fitted(self.vectorizer, "n_features_in_")
        return self.vectorizer.transform(processed_dataframe)

    def get_config(self) -> dict:
        """Devuelve las configuraciones efectivas de ambos componentes."""
        return {
            "preprocessor": self.preprocessor.get_config(),
            "vectorizer": {
                key: value
                for key, value in vars(self.vectorizer).items()
                if not key.endswith("_")
            },
        }

# Se recomienda conservar una misma instancia de builder entre train y val.
def build_dataset(dataframe, mode: str = "train", *, builder: NLIDatasetBuilder):
    """Construye los vectores usando una instancia persistente de NLIDatasetBuilder.

    Ejemplo:
        builder = NLIDatasetBuilder(...)
        X_train = build_dataset(train_df, mode="train", builder=builder)
        X_val = build_dataset(val_df, mode="val", builder=builder)
    """
    if not isinstance(builder, NLIDatasetBuilder):
        raise TypeError("builder debe ser una instancia de NLIDatasetBuilder.")
    return builder.build_dataset(dataframe, mode=mode)