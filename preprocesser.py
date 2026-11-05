"""Preprocesamiento de texto en inglés para conjuntos de Inferencia en Lenguaje Natural."""

import json
import unicodedata
from collections import Counter

import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords, wordnet
from nltk.stem import LancasterStemmer, PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from spacy.tokens import Doc


class NLITextPreprocesser():
    """Preprocesa columnas de texto NLI en inglés y devuelve un DataFrame. """

    TOKENIZERS = ("whitespace", "nltk", "spacy")
    STOPWORD_METHODS = ("none", "nltk", "sklearn", "custom")
    REDUCTION_METHODS = ("none", "porter", "lancaster", "snowball", "wordnet", "spacy_lemma")
    ENGLISH_NEGATIONS = frozenset({
        "no", "not", "nor", "never", "none", "nobody", "nothing", "nowhere",
        "neither", "cannot", "without", "hardly", "barely", "scarcely", "n't"})
    APOSTROPHES = str.maketrans({
        "\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'", "\u00b4": "'"})

    def __init__(
        self,
        columns=("premise", "hypothesis"),
        language="en",
        language_column="lang_abv",
        lowercase=True,
        strip_accents=True,
        tokenizer="whitespace",
        stopword_method="sklearn",
        keep_negations=True,
        custom_stopwords=None,
        reduction="none",
        ngram_orders=(2, 3),
        top_n=20,
    ):
        self.columns = columns
        self.language = language
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.tokenizer = tokenizer
        self.stopword_method = stopword_method
        self.keep_negations = keep_negations
        self.custom_stopwords = custom_stopwords
        self.reduction = reduction
        self.ngram_orders = ngram_orders
        self.top_n = top_n

        self.spacy_tokenizer = spacy.blank("en").tokenizer
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def standardize_text(self, text) -> str:
        """Pasa a minúsculas y, opcionalmente, elimina marcas de acento Unicode."""
        if text is None:
            text = str(text)
        if self.lowercase:
            text = text.lower()

        text = text.translate(self.APOSTROPHES)

        if self.strip_accents:
            # canonical decomposition, separa acentos de letra base
            text = unicodedata.normalize("NFD", text)

            text = "".join(character for character in text if unicodedata.category(
                character) != "Mn")  # mark, nonspacing (quitar acentos)

        return " ".join(text.split())

    def tokenize_text(self, text, method=None) -> list[str]:
        """Tokeniza texto usando whitespace, NLTK o spaCy."""
        method = method or self.tokenizer
        text = self.standardize_text(text)

        if method not in self.TOKENIZERS:
            raise ValueError(f"method debe ser uno de {self.TOKENIZERS}.")

        if method == "whitespace":
            return text.split()

        if method == "nltk":
            return [w.lower() for w in word_tokenize(text) if w.isalpha()]

        return [
            token.text.lower()
            for token in self.spacy_tokenizer(text)
            if not token.is_space and not token.is_punct]

    def get_stopwords(self, method=None, custom_words=None) -> set[str]:
        """Carga una lista de stopwords."""
        if method is None:
            method = self.stopword_method
        if self.custom_words is None:
            custom_words = self.custom_stopwords
        if method == "none":
            return set()
        if method == "custom":
            return {str(word).lower() for word in (custom_words or ())}
        if method == "sklearn":
            words = set(ENGLISH_STOP_WORDS)
        elif method == "nltk":
            words = set(stopwords.words("english"))
        else:
            raise ValueError(f"method debe ser uno de {self.STOPWORD_METHODS}")
        if custom_words:
            words.update(str(word).lower() for word in custom_words)
        return words

    def is_negation(self, token: str) -> bool:
        token = token.lower()
        return token in self.ENGLISH_NEGATIONS or token.endswith("n't")

    def remove_stopwords(self, tokens, stopword_set=None, keep_negations=None) -> list[str]:
        """Elimina stopwords. Conserva negaciones por defecto."""
        if stopword_set is None:
            stopword_set = self.get_stopwords()
        if keep_negations is None:
            keep_negations = self.keep_negations

        return [w for w in tokens if w not in stopword_set
                or (keep_negations and self.is_negation(w))]

    def make_ngrams(self, tokens, n) -> list[str]:
        """Crea n-gramas dentro de un texto."""
        if n < 1:
            raise ValueError("n debe ser mayor o igual que uno")
        return [" ".join(ngram) for ngram in ngrams(tokens, n)]

    def top_tokens(self, tokenized_texts, top_n=None) -> pd.DataFrame:
        """Devuelve los tokens más frecuentes."""
        top_n = top_n or self.top_n
        if top_n < 1:
            raise ValueError("top_n debe ser mayor que cero.")

        counts = Counter(token for text in tokenized_texts for token in text)
        freq = Counter(token for text in tokenized_texts for token in set(text))

        total_tokens = sum(counts.values())
        total_texts = len(tokenized_texts)

        rows = []
        for token, count in counts.most_common(top_n):
            rows.append({
                "token": token,
                "count": count,
                "token_percentage": count / total_tokens if total_tokens else 0,
                "text_frequency": freq[token],
                "text_presence_freq": freq[token] / total_texts if total_texts else 0,
            })

        return pd.DataFrame(rows)

    def top_ngrams(self, texts, n=10, top_n=None) -> pd.DataFrame:
        """Devuelve los n-gramas más frecuentes, calculados por texto."""
        ngram_texts = []
        for text in texts:
            ngram_texts.append(self.make_ngrams(text, n))

        return self.top_tokens(ngram_texts, top_n).rename(columns={"token": "ngram"})

    def reduce_tokens(self, tokens, method=None) -> list[str]:
        """Aplica stemming o lematización a una secuencia de tokens."""
        method = method or self.reduction

        if method not in self.REDUCTION_METHODS:
            raise ValueError(f"method debe ser uno de {self.REDUCTION_METHODS}.")

        tokens = list(tokens)

        if method == "none":
            return tokens

        if method == "porter":
            stemmer = PorterStemmer()
            return [stemmer.stem(token) for token in tokens]

        if method == "lancaster":
            stemmer = LancasterStemmer()
            return [stemmer.stem(token) for token in tokens]

        if method == "snowball":
            stemmer = SnowballStemmer("english")
            return [stemmer.stem(token) for token in tokens]

        if method == "wordnet":
            pos_tags = nltk.pos_tag(tokens)  # pos tagging para mejores resultados

            # solo guardamos la primera letra, que representa la categoria
            pos_map = {"J": wordnet.ADJ,
                       "V": wordnet.VERB,
                       "N": wordnet.NOUN,
                       "R": wordnet.ADV}

            lemmatizer = WordNetLemmatizer()
            res = []
            for token, (_, tag) in zip(tokens, pos_tags):
                res.append(lemmatizer.lemmatize(token, pos_map.get(tag[0], wordnet.NOUN)))

            return res

        # spacy
        nlp = self.nlp
        doc = Doc(nlp.vocab, words=tokens)
        doc = nlp(doc)

        return [(token.lemma_ or token.text).lower() for token in doc]
