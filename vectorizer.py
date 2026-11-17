import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_is_fitted
from sentence_transformers import SentenceTransformer


def _to_text(value, use_pos=False):
    """Convierte texto o una lista de tokens (con o sin POS tags) en una cadena."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        tokens = []
        for t in value:
            if t is None:
                continue
            # Si el token viene como tupla/lista (palabra, tag_pos)
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                word, pos = str(t[0]), str(t[1])
                if use_pos:
                    tokens.append(f"{word}_{pos}")
                else:
                    tokens.append(word)
            else:
                # Si es un string plano (u otro formato)
                tokens.append(str(t))
        return " ".join(tokens)
    return str(value)


class NLITextVectorizer(BaseEstimator, TransformerMixin):
    """
    Transformador personalizado compatible con scikit-learn para tareas de Inferencia 
    en Lenguaje Natural (NLI), diseñado para procesar columnas de texto o tokens 
    (con o sin etiquetas POS) y convertirlas en matrices de características numéricas 
    o vectores densos.

    Attributes:
        vectorizer_ (CountVectorizer o TfidfVectorizer): Ajustado según el método elegido.
        word_vectorizer_ (TfidfVectorizer): Usado en 'word_char' para palabras.
        char_vectorizer_ (TfidfVectorizer): Usado en 'word_char' para caracteres.
        svd_components_ (TruncatedSVD): Usado en 'tfidf_svd' para reducción dimensional.
        embedding_model_ (SentenceTransformer): Usado en 'embeddings'.
        n_features_in_ (int): Número de columnas de entrada procesadas.
    """

    VALID_METHODS = {
        "count", "binary", "relative", "tfidf",
        "tfidf_bigram", "tfidf_trigram", "char_tfidf",
        "word_char", "pair_separate", "pair_diff",
        "tfidf_svd", "embeddings"
    }

    def __init__(
        self,
        columns=("tokens",),
        method="tfidf",
        max_features=30000,
        min_df=1,
        word_ngram_range=(1, 1),
        char_ngram_range=(3, 5),
        analyzer="word",
        svd_components=300,
        normalize_embeddings=True,
        embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        embedding_batch_size=32,
        sublinear_tf=True,
        random_state=42,
        use_pos=False
    ):
        """
        Inicializa el vectorizador NLI.

        Args:
            columns (tuple o list of str): Nombres de las columnas de tokens del DataFrame de entrada 
                a procesar (por ejemplo, `("tokens",)` o `("premise", "hypothesis")`).
            method (str): Estrategia de vectorización a emplear. Opciones disponibles en `VALID_METHODS`:
                
                - **Enfoques de Recuento y TF-IDF clásicos:**
                  * `"count"`: Matriz dispersa con frecuencias absolutas de los términos.
                  * `"binary"`: Matriz dispersa binaria (1 si el término aparece, 0 si no).
                  * `"relative"`: Frecuencias relativas normalizadas (norma L1).
                  * `"tfidf"`: Ponderación clásica Term Frequency - Inverse Document Frequency.
                  * `"tfidf_bigram"`: TF-IDF analizando unigramas y bigramas (1, 2).
                  * `"tfidf_trigram"`: TF-IDF analizando unigramas hasta trigramas (1, 3).
                  * `"char_tfidf"`: TF-IDF basado en n-gramas de caracteres.
                  * `"word_char"`: Concatena horizontalmente(hstack) las características TF-IDF a nivel de palabras y de caracteres.

                - **Enfoques para Pares de Textos (NLI):**
                  * `"pair_separate"`: Vectoriza de forma independiente dos columnas (premise y hypothesis) 
                    y concatena horizontalmente sus matrices `[Xp, Xh]`.
                  * `"pair_diff"`: Combina premisa e hipótesis añadiendo la diferencia absoluta (`|Xp - Xh|`) 
                    y métricas de solapamiento semántico (Jaccard, coberturas y similitud de coseno).

                - **Reducción de Dimensionalidad y Modelos Profundos:**
                  * `"tfidf_svd"`: Aplica TF-IDF y reduce la dimensionalidad mediante Truncated SVD 
                    (Latent Semantic Analysis) generando una matriz densa.
                  * `"embeddings"`: Utiliza un modelo de Sentence Transformers para generar vectores densos 
                    semánticos. Si hay dos columnas, devuelve la concatenación: `[ep, eh, |ep - eh|, ep * eh]`.

            max_features (int): Número máximo de términos en el vocabulario (ordena por frecuencia). 
                Ayuda a controlar el consumo de memoria.
            min_df (int o float): Frecuencia mínima de documento. Descarta términos que aparezcan 
                en menos documentos que este valor.
            word_ngram_range (tuple): Rango `(min_n, max_n)` para n-gramas de palabras.
            char_ngram_range (tuple): Rango `(min_n, max_n)` para n-gramas de caracteres.
            analyzer (str): Nivel de análisis para el vectorizador base (`"word"` o `"char"`).
            svd_components (int): Número de componentes de salida al usar `tfidf_svd`.
            normalize_embeddings (bool): Si es True, normaliza los vectores de embeddings a norma L2.
            embedding_model (str): Identificador del modelo preentrenado de Hugging Face para SentenceTransformers.
            embedding_batch_size (int): Tamaño del lote para la generación de embeddings neuronales.
            sublinear_tf (bool): Aplica escala logarítmica a la frecuencia de término `1 + log(tf)`.
            random_state (int): Semilla aleatoria para la reproducibilidad en Truncated SVD.
            use_pos (bool): Si es True y los tokens vienen estructurados como tuplas `(palabra, pos_tag)`, 
                combina ambos como `"palabra_POS"` para tratarlos como entidades independientes. 
                Si es False, omite la etiqueta POS y usa únicamente la palabra.
        """
        self.columns = columns
        self.method = method
        self.max_features = max_features
        self.min_df = min_df
        self.word_ngram_range = word_ngram_range
        self.char_ngram_range = char_ngram_range
        self.analyzer = analyzer
        self.svd_components = svd_components
        self.normalize_embeddings = normalize_embeddings
        self.embedding_model = embedding_model
        self.embedding_batch_size = embedding_batch_size
        self.sublinear_tf = sublinear_tf
        self.random_state = random_state
        self.use_pos = use_pos

    def _get_texts(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X debe ser un DataFrame.")

        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Faltan columnas: {missing}")
        
        return [
            pd.Series([_to_text(v, use_pos=self.use_pos) for v in X[c]]).reset_index(drop=True)
            for c in self.columns
        ]

    def _fit_corpus(self, vectorizer, texts):
        corpus = pd.concat(texts, ignore_index=True)
        vectorizer.fit(corpus)
        return vectorizer

    def _make_tfidf(self, analyzer="word", ngrams=None):
        return TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngrams or self.word_ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            dtype=np.float32,
        )

    def fit(self, X, y=None):
        if self.method not in self.VALID_METHODS:
            raise ValueError(f"Método no reconocido: {self.method}")

        texts = self._get_texts(X)
        self.n_features_in_ = len(self.columns)

        if self.method in {"pair_separate", "pair_diff"} and len(texts) != 2:
            raise ValueError(f"{self.method} necesita dos columnas.")

        # Vertical stack para tokens de premisa e hipótesis 
        # (len (corpus) = 2 * len(X)) si hay 2 columnas (premisa + hipotesis)
        corpus = pd.concat(texts, ignore_index=True)

        if self.method in {"count", "binary", "relative"}:
            self.vectorizer_ = CountVectorizer(
                binary=self.method == "binary",
                ngram_range=self.word_ngram_range,
                min_df=self.min_df,
                max_features=self.max_features,
                dtype=np.float32,
            )
            self.vectorizer_.fit(corpus)

        elif self.method == "tfidf":
            self.vectorizer_ = self._make_tfidf()
            self.vectorizer_.fit(corpus)

        elif self.method == "tfidf_bigram":
            self.vectorizer_ = self._make_tfidf(ngrams=(1, 2))
            self.vectorizer_.fit(corpus)

        elif self.method == "tfidf_trigram":
            self.vectorizer_ = self._make_tfidf(ngrams=(1, 3))
            self.vectorizer_.fit(corpus)

        elif self.method == "char_tfidf":
            self.vectorizer_ = self._make_tfidf(
                analyzer="char",
                ngrams=self.char_ngram_range
            )
            self.vectorizer_.fit(corpus)

        elif self.method == "word_char":
            self.word_vectorizer_ = self._make_tfidf()
            self.char_vectorizer_ = self._make_tfidf(
                analyzer="char",
                ngrams=self.char_ngram_range
            )
            self.word_vectorizer_.fit(corpus)
            self.char_vectorizer_.fit(corpus)

        elif self.method in {"pair_separate", "pair_diff"}:
            self.vectorizer_ = self._make_tfidf()
            self.vectorizer_.fit(corpus)

        elif self.method == "tfidf_svd":
            self.vectorizer_ = self._make_tfidf()
            self.vectorizer_.fit(corpus)

            matrix = self.vectorizer_.transform(corpus)
            max_valid = min(matrix.shape[0] - 1, matrix.shape[1] - 1)

            if max_valid < 1:
                raise ValueError("No hay suficientes datos para aplicar SVD.")

            self.svd_ = TruncatedSVD(
                n_components=min(self.svd_components, max_valid),
                random_state=self.random_state
            )
            self.svd_.fit(matrix)

        elif self.method == "embeddings":
            self.embedding_model_ = SentenceTransformer(
                self.embedding_model
            )

        return self

    def _encode(self, texts):
        return self.embedding_model_.encode(
            list(texts),
            batch_size=self.embedding_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        ).astype(np.float32)

    def _transform_one(self, texts, vectorizer):
        matrix = vectorizer.transform(texts)

        if self.method == "relative":
            return normalize(matrix, norm="l1")

        return matrix.tocsr()

    def _overlap_features(self, p, h, Xp, Xh):
        rows = []

        for a, b in zip(p, h):
            ta = set(a.split())
            tb = set(b.split())

            intersection = ta & tb
            union = ta | tb

            jaccard = len(intersection) / len(union) if union else 0
            coverage_p = len(intersection) / len(ta) if ta else 0
            coverage_h = len(intersection) / len(tb) if tb else 0

            rows.append([jaccard, coverage_p, coverage_h])

        overlap = sparse.csr_matrix(
            np.asarray(rows, dtype=np.float32)
        )

        numerator = np.asarray(
            Xp.multiply(Xh).sum(axis=1)
        ).ravel()

        norm_p = np.sqrt(
            np.asarray(Xp.multiply(Xp).sum(axis=1)).ravel()
        )
        norm_h = np.sqrt(
            np.asarray(Xh.multiply(Xh).sum(axis=1)).ravel()
        )

        denominator = norm_p * norm_h
        cosine = np.zeros_like(numerator)

        valid = denominator > 0
        cosine[valid] = numerator[valid] / denominator[valid]

        cosine = sparse.csr_matrix(cosine.reshape(-1, 1))

        return sparse.hstack(
            [overlap, cosine],
            format="csr"
        )

    def transform(self, X):
        check_is_fitted(self, "n_features_in_")
        texts = self._get_texts(X)

        if self.method in {
            "count", "binary", "relative", "tfidf",
            "tfidf_bigram", "tfidf_trigram", "char_tfidf"
        }:
            matrices = [
                self._transform_one(t, self.vectorizer_)
                for t in texts
            ]

            if len(matrices) == 1:
                return matrices[0]

            return sparse.hstack(matrices, format="csr")

        if self.method == "word_char":
            blocks = []

            for t in texts:
                word = self.word_vectorizer_.transform(t)
                char = self.char_vectorizer_.transform(t)
                blocks.extend([word, char])

            return sparse.hstack(blocks, format="csr")

        if self.method in {"pair_separate", "pair_diff"}:
            p, h = texts

            Xp = self.vectorizer_.transform(p).tocsr()
            Xh = self.vectorizer_.transform(h).tocsr()

            blocks = [Xp, Xh]

            if self.method == "pair_diff":
                blocks.append(abs(Xp - Xh))
                blocks.append(
                    self._overlap_features(p, h, Xp, Xh)
                )

            return sparse.hstack(blocks, format="csr")

        if self.method == "tfidf_svd":
            matrices = [
                self.svd_.transform(
                    self.vectorizer_.transform(t)
                )
                for t in texts
            ]
            return np.hstack(matrices).astype(np.float32)

        if self.method == "embeddings":
            embeddings = [self._encode(t) for t in texts]

            if len(embeddings) == 1:
                return embeddings[0]

            ep, eh = embeddings

            return np.hstack([
                ep,
                eh,
                np.abs(ep - eh),
                ep * eh
            ]).astype(np.float32)

        raise ValueError("Método no implementado.")