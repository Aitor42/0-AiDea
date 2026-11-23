# Investigación de Random Forest para NLI

[Mapa del repositorio](../../README.md) · [Guía del proyecto](../PROJECT_GUIDE.md) · [Entorno y ejecución](../reproducibility.md)

Estudio histórico de *Contradictory, My Dear Watson* para Machine Learning 2026/27 (ESI-UCLM). El [cuaderno de Random Forest](../../experiments/notebooks/random_forest_training.ipynb) conserva el entrenamiento de la configuración léxica.

## Resumen de resultados

| Configuración | Accuracy CV (3 semillas × 5 folds) | Holdout reservado (accuracy / macro-F1) |
|---|---:|---:|
| Baseline histórico | 0,4527 | 0,4814 / 0,4806 |
| **TF-IDF ajustado** | **0,5146** | **0,5206 / 0,5193** |
| Diferencia | **+6,19 pp** | **+3,92 / +3,87 pp** |

La mejora léxica combina tokenización de spaCy, selección chi-cuadrado de 300 características, unigramas con `min_df=5` y rasgos de relación. Se mantuvieron los hiperparámetros del bosque.

Con el mismo bosque y el mismo holdout, MiniLM `full` llegó a 0,6052 de accuracy y 0,6062 de macro-F1. El mejor resultado del control E5 fue 0,5660 / 0,5662 con el híbrido. El cuaderno de Random Forest conserva la configuración léxica. La entrega MiniLM de la guía usa un voto de otros clasificadores. [Comparación de embeddings](#embeddings-multilingües) · [Entrega seleccionada](../PROJECT_GUIDE.md#configuración-seleccionada-para-kaggle)

## Índice

- [Resumen de resultados](#resumen-de-resultados)
- [Protocolo](#protocolo)
- [Ablaciones léxicas](#ablaciones-léxicas)
- [Configuración léxica seleccionada](#configuración-léxica-seleccionada)
- [Confirmación de TF-IDF](#confirmación-de-tf-idf)
- [Embeddings multilingües](#embeddings-multilingües)
- [Limitaciones y decisiones finales](#limitaciones-y-decisiones-finales)
- [Reproducción](#reproducción)

## Protocolo

- **Selección.** CV estratificada de cinco folds sobre los 9.696 pares de desarrollo. Las comparaciones emparejadas usan los mismos folds. La confirmación repite la CV con tres semillas y reporta un t-test pareado.
- **Holdout histórico.** Los 2.424 pares reservados se usaron para evaluar las configuraciones seleccionadas. Hay solapamiento de premisas entre ese holdout y desarrollo. Con ese tamaño, el informe estimó un error estándar de alrededor de 1 pp para accuracy y tomó diferencias de unos 2 pp como referencia aproximada.
- **Ajuste por fold.** El preprocesado se guarda fila a fila. El vectorizador y `SelectKBest` se ajustan en el bloque de entrenamiento de cada fold. [Partición histórica y validación](../PROJECT_GUIDE.md#entrenamiento-validación-y-métricas)

## Ablaciones léxicas

### Cambios acumulados

Cada paso parte del anterior. CV de 5 folds con semilla 42.

| Paso | Configuración | CV accuracy | Δ frente al baseline (pp) |
|---|---|---:|---:|
| 0 | Base: whitespace, (1,2), `min_df=50` | 0,4525 | — |
| 1 | + chi² `k=500` | 0,4652 | +1,28 |
| 2 | + unigramas y `k=300` | 0,4668 | +1,43 |
| 3 | + tokenizador spaCy | 0,4868 | +3,43 |
| 4 | + features de negación | 0,4893 | +3,68 |
| 5 | + features de longitud | 0,4966 | +4,41 |
| 6 | + números (bloque de relación completo) | 0,5028 | +5,03 |
| 7 | + `min_df=5` | 0,5068 | +5,44 |
| 8 | + tokens sin sufijo `_POS` | **0,5133** | **+6,09** |

Los bloques de relación se compararon sobre el paso 3.

| Bloque | CV accuracy | Δ sobre el paso 3 (pp) |
|---|---:|---:|
| Negaciones | 0,4893 | +0,25 |
| Números | 0,4896 | +0,28 |
| Longitudes | 0,4966 | +0,98 |
| Relación completa (14 features) | 0,5028 | +1,60 |

Con la configuración final (`min_df=5` y tokens sin `_POS`) el pipeline genera
**19.107 características**. Chi-cuadrado selecciona 300, repartidas en 287 de texto, 1 de ratios POS y 12 de relación. Los ratios POS aportaron 0,12 pp, dentro de la variación entre folds. Se conservaron por su bajo coste y para cubrir el requisito de POS del enunciado.

### Alternativas probadas

| Cambio | CV accuracy | Δ frente a su control (pp) | Lectura |
|---|---:|---:|---|
| `tfidf_svd` (LSA, 300 componentes) | 0,3719 | −8,05 | pierde la señal léxica exacta |
| `word_char` (palabra + carácter) | 0,4068 | −4,57 | mucho ruido y 16 s por fit |
| `pair_separate` (sin `\|Xp−Xh\|` ni solape) | 0,4141 | −3,84 | las features de diferencia son clave |
| Quitar stopwords | 0,4341 | −1,84 | destruye palabras función y negaciones |
| Variantes de n-gramas | 0,4558–0,4566 | +0,33 a +0,41 | Probar (1,3) o solo unigramas dio cambios pequeños. |
| `class_weight="balanced"` | 0,4635 | −0,17 | las clases ya están casi balanceadas |
| `f_classif` en vez de chi² | 0,4645 | −0,22 | chi² encaja mejor con TF-IDF no negativo |
| One-hot de idioma | 0,4660 | −0,08 | el idioma no aporta tras el texto |
| `min_df` 20 / 100 | 0,4513 / 0,4506 | ≈0 | insensible en ese rango |
| `min_samples_leaf` 1 / 10, `max_depth=None`, `entropy` | 0,4628–0,4648 | ≤0 | la poda actual ya es adecuada |
| `max_features=0.33` | 0,4596 | −0,56 | 18 s por fit, peor |

Los deltas de esta tabla usan el control de cada variante del banco de pruebas. [Tabla completa de ablaciones](../../experiments/results/random_forest/rf_table.csv)

### Conclusiones de las ablaciones

- **Tokenización y reducción.** spaCy separó puntuación y contracciones. El paso de tokenización añadió 2,00 pp sobre la configuración anterior, que ya incluía unigramas y chi-cuadrado. La mejora acumulada fue de 3,43 pp frente al baseline. Porter, Snowball, WordNet y spaCy quedaron a ±0,4 pp. Se mantuvo `spacy_lemma` por su coste y porque ya estaba integrado.
- **Stopwords y POS.** Quitar stopwords costó entre 1,8 y 2,0 pp. El sufijo `_POS` en el token de TF-IDF costó 0,65 pp. El preprocesado conserva el etiquetado POS para calcular los ratios.
- **Selección.** En un diagnóstico con 2.041 características, 311 tenían importancia casi nula y las 200 primeras concentraban el 76 %. Chi-cuadrado con unas 300 características ganó 1,3 pp y redujo el ajuste de 5,1 a 2,6 s por fold.
- **Relación.** Cuando solo la hipótesis lleva negación, el 71 % de los pares son contradicción, frente al 29 % de referencia. Las longitudes ayudan a separar neutralidad y los números aportan una señal de contradicción. Los tres bloques se calculan fila a fila.
- **Bosque.** Las variaciones de tamaño y poda se quedaron cerca de la configuración de referencia. Las mayores mejoras vinieron de la representación.

## Configuración léxica seleccionada

El bosque conserva `n_estimators=1000`, `max_depth=50`, `min_samples_leaf=5`, `min_samples_split=5`, `max_features="sqrt"` y `criterion="gini"`.

Parámetros que distinguen la receta. Las opciones restantes están en el [preprocesado](../../src/nli_project/preprocessing/preprocessing.py) y en la [configuración archivada](../../experiments/results/random_forest/tfidf/final_config.json).

```python
PREPROCESSING_PARAMS = {
    "tokenizer": "spacy",          # antes: "whitespace"
    "reduction": "spacy_lemma",    # sin cambios
    "stopword_method": "none",     # sin cambios
    "use_pos": True,               # sigue alimentando los ratios POS
}

VECTORIZER_PARAMS = {
    "method": "pair_diff",
    "columns": ("premise_processed", "hypothesis_processed"),
    "min_df": 5,                   # antes: 50
    "word_ngram_range": (1, 1),    # antes: (1, 2)
    "use_pos": False,              # antes: True (tokens sin sufijo _POS)
}

# Pipeline: features -> SelectKBest(chi2, k=300) -> RandomForestClassifier
# features = FeatureUnion([NLITextVectorizer, POSRatioFeatures, RelationFeatures])
```

`RelationFeatures` genera 14 columnas no negativas compatibles con chi-cuadrado.

- Negaciones en premisa e hipótesis, diferencia absoluta y XOR.
- Longitudes de premisa e hipótesis, diferencia absoluta, ratio y qué frase es más larga.
- Números en cada frase, diferencia y Jaccard de números compartidos.

## Confirmación de TF-IDF

### CV repetida (3 semillas × 5 folds)

| Configuración | Media | Desv. entre semillas |
|---|---:|---:|
| Base | 0,4527 | 0,0006 |
| Unigramas + spaCy + relación + `min_df=5` | 0,5071 | 0,0002 |
| **+ tokens sin `_POS`** | **0,5146** | 0,0016 |

La comparación pareada dio **+6,19 pp**, con mejora en los **15 folds** y **p ≈ 1,2·10⁻¹¹**.

### Holdout histórico de 2.424 pares

| Clase | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| entailment | 0,5423 | 0,5760 | 0,5587 | 835 |
| neutral | 0,4932 | 0,5155 | 0,5041 | 776 |
| contradiction | 0,5248 | 0,4686 | 0,4951 | 813 |

- Accuracy **0,5206** (base 0,4814), macro-F1 **0,5193** (base 0,4806).
- 1.262 aciertos frente a 1.167, con una ganancia de 95 pares sobre todo en `entailment` y `neutral`.
- La matriz de confusión registra errores en las seis direcciones y predicciones para las tres clases.

### Accuracy por lengua en holdout

| Lengua | n | Base | TF-IDF ajustado | Δ (pp) |
|---|---:|---:|---:|---:|
| English | 1.354 | 0,513 | 0,579 | +6,6 |
| Russian | 94 | 0,394 | 0,468 | +7,4 |
| Hindi | 86 | 0,616 | 0,500 | −11,6 |
| Spanish | 83 | 0,530 | 0,530 | 0,0 |
| Arabic | 79 | 0,367 | 0,418 | +5,1 |
| Thai | 77 | 0,286 | 0,351 | +6,5 |
| Bulgarian | 71 | 0,465 | 0,549 | +8,5 |
| French | 59 | 0,407 | 0,458 | +5,1 |

El inglés concentra la mayor parte de la mejora y supone alrededor del 57 % del corpus. Las muestras de otros idiomas rondan los 60–100 pares y tienen un error estándar aproximado de 5–6 pp. Las variaciones por idioma deben leerse con ese soporte y con la limitación de usar análisis lingüístico inglés. [Métricas por idioma](../../experiments/results/random_forest/tfidf/per_language.csv) · [Confirmación completa](../../experiments/results/random_forest/rf_confirm.json)

## Embeddings multilingües

Se probó si un encoder multilingüe reducía la brecha de 13,2 pp entre el inglés (0,579) y el resto (0,447) en TF-IDF. Se conservaron el split (9.696 / 2.424) y los hiperparámetros del bosque. El encoder recibe los textos originales con puntuación y sin lematizar. Los vectores se normalizan en L2.

| Representación | Composición | Dimensiones |
| --- | --- | ---: |
| `concat` | Premisa e hipótesis concatenadas. | 768 |
| `full` | Concatenación, diferencia absoluta y producto por componentes. | 1.536 |
| `diff+product` | Diferencia absoluta y producto por componentes. | 768 |
| `concat + cosine` | Concatenación y similitud coseno. | 769 |

También se probaron PCA a 200 dimensiones, `max_features=0.1` y un híbrido de TF-IDF, POS, relación y embeddings. El híbrido usa `f_classif` porque los embeddings admiten valores negativos.

### MiniLM

| Variante | Dims | Acc | F1 | Inglés | No inglés | Fit |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF `pair_diff` (referencia) | 300 | 0,5206 | 0,5193 | 0,5790 | 0,4467 | 2 s |
| `concat` | 768 | 0,4769 | 0,4615 | 0,4609 | 0,4972 | 21 s |
| `diff+product` | 768 | 0,5837 | 0,5804 | 0,6108 | 0,5495 | 22 s |
| `concat + cosine` | 769 | 0,5763 | 0,5725 | 0,5835 | 0,5673 | 23 s |
| **`full`** | **1.536** | **0,6052** | **0,6062** | **0,6337** | **0,5692** | **32 s** |
| `full` + `max_features=0.1` | 1.536 | 0,5986 | 0,6001 | 0,6248 | 0,5654 | 132 s |
| PCA 200 | 200 | 0,5268 | 0,5186 | 0,5140 | 0,5430 | 13 s |
| Híbrido TF-IDF + embeddings | 400 | 0,5771 | 0,5780 | 0,5975 | 0,5514 | 28 s |

Con `paraphrase-multilingual-MiniLM-L12-v2`, añadir diferencia y producto a `concat` mejoró 12,8 pp. PCA a 200 dimensiones quedó en 0,527 y el híbrido en 0,577 con un presupuesto de 400 características. [Resultados completos de MiniLM](../../experiments/results/random_forest/embeddings/rf_embeddings_results_sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2.json)

### Por lengua (MiniLM `full` frente a TF-IDF)

| Lengua | n | TF-IDF | Embeddings | Δ (pp) |
|---|---:|---:|---:|---:|
| English | 1.354 | 0,579 | 0,634 | +5,5 |
| Chinese | 103 | 0,340 | 0,670 | **+33,0** |
| Russian | 94 | 0,468 | 0,532 | +6,4 |
| Hindi | 86 | 0,500 | 0,547 | +4,7 |
| Spanish | 83 | 0,530 | 0,554 | +2,4 |
| Arabic | 79 | 0,418 | 0,506 | +8,9 |
| Greek | 77 | 0,455 | 0,649 | +19,5 |
| Thai | 77 | 0,351 | 0,584 | +23,4 |
| Swahili | 72 | 0,444 | 0,417 | −2,8 |
| Bulgarian | 71 | 0,549 | 0,507 | −4,2 |
| Urdu | 70 | 0,500 | 0,543 | +4,3 |
| Vietnamese | 67 | 0,567 | 0,672 | +10,4 |
| German | 67 | 0,343 | 0,612 | **+26,9** |
| Turkish | 65 | 0,354 | 0,569 | +21,5 |
| French | 59 | 0,458 | 0,593 | +13,6 |

Mejoran 13 de 15 lenguas. La media ponderada por el número de pares sube 8,5 pp y la brecha entre inglés y el resto se reduce de 13,2 a 6,5 pp.

### Control con E5

El estudio consideró que MiniLM podía estar familiarizado con frases de XNLI por su entrenamiento con datos de paráfrasis y NLI. Repitió la prueba con `intfloat/multilingual-e5-small` como control más conservador, usando el prefijo `query: ` en ambos textos.

| Variante | Dims | Acc | F1 | Inglés | No inglés |
|---|---:|---:|---:|---:|---:|
| TF-IDF (referencia) | 300 | 0,5206 | 0,5193 | 0,5790 | 0,4467 |
| E5 `full` | 1.536 | 0,5602 | 0,5589 | 0,5731 | 0,5439 |
| E5 `diff+product` | 768 | 0,5503 | 0,5456 | 0,5731 | 0,5215 |
| E5 híbrido | 400 | 0,5660 | 0,5662 | 0,5753 | 0,5542 |

E5 `full` mejoró 4,0 pp en el total y 9,7 pp en los pares de origen no inglés. En inglés obtuvo 0,573 frente a 0,579 con TF-IDF. El patrón es compatible con transferencia entre lenguas. La ventaja adicional de MiniLM en inglés motiva revisar la posible familiaridad con NLI. El estudio planteó declarar el encoder y contrastar la decisión en Kaggle. [Resultados completos de E5](../../experiments/results/random_forest/embeddings/rf_embeddings_results_intfloat__multilingual-e5-small.json)

### Robustez y coste

- **CV agrupada por premisa.** En cinco folds con cero premisas compartidas, el informe histórico registró 0,5997 ± 0,0082 para MiniLM frente a 0,5151 para TF-IDF. Los archivos de embeddings enlazados contienen las métricas del holdout. [Referencia agrupada de TF-IDF](../../experiments/results/random_forest/tfidf/grouped_cv.json)
- **Coste.** Codificar las 24.240 entradas de texto llevó unos 8 min en CPU. Los vectores se reutilizan desde caché. El ajuste pasó de 2 s a 32 s con 1.536 dimensiones.
- **Híbrido.** TF-IDF más embeddings con 400 características seleccionadas mediante `f_classif` obtuvo 0,577 frente a 0,605 con MiniLM `full`.

### Conclusión de embeddings

MiniLM `full` ganó 8,5 pp en holdout y redujo la brecha entre inglés y el resto de 13,2 a 6,5 pp. E5 `full` aportó 4,0 pp sobre el TF-IDF. El contraste agrupado conservó la ventaja de MiniLM. La elección del encoder quedaba condicionada al coste adicional y a la posible familiaridad con NLI.

## Limitaciones y decisiones finales

- **Selección.** El estudio probó unas 45 configuraciones sobre el mismo CV. El informe estimó un posible sesgo de unos 0,3 pp. La CV repetida y la mejora de 3,9 pp en el holdout aportan evidencia adicional, con el solapamiento de premisas propio de ese split.
- **Señales superficiales y lengua.** Parte de la mejora se asocia a las longitudes. Train y test conservan idiomas y proporciones similares, pero la transferencia de esa señal a Kaggle quedó por evaluar. Las herramientas inglesas limitan la representación de los demás idiomas.
- **Escalas.** Chi-cuadrado puede favorecer los rasgos de relación de mayor magnitud frente a TF-IDF. La receta selecciona 300 características y ajusta en unos 2,6 s por fold frente a 5,1 s del baseline.
- **Decisión del estudio.** Se conservó TF-IDF ajustado en el cuaderno de Random Forest. MiniLM y E5 quedaron como alternativas experimentales con distinto coste y riesgo de familiaridad con NLI. La comparación con un segundo clasificador quedó fuera del estudio de RF y se aborda en el [informe de experimentos](team_experiments.md). El enunciado pide dos algoritmos supervisados sobre los mismos ejemplos. El criterio de comparación y ajuste justificados aporta 8 de 30 puntos.

## Reproducción

### Reproducción de TF-IDF

Ejecuta los módulos desde la raíz con el [entorno del proyecto](../reproducibility.md#instalación-y-dependencias) activo y los [CSV oficiales preparados](../reproducibility.md#preparación-de-los-datos).

```sh
python -m experiments.scripts.rf_experiments --list
python -m experiments.scripts.rf_experiments sel300_uni_spacy_rel_mindf5_noposvec
python -m experiments.scripts.rf_confirm base sel300_uni_spacy_rel_mindf5_noposvec
python -m experiments.scripts.rf_summary
```

| Artefacto archivado | Contenido |
| --- | --- |
| [`experiments.jsonl`](../../experiments/results/random_forest/experiments.jsonl) | Resultados por fold de cada configuración. |
| [`rf_table.csv`](../../experiments/results/random_forest/rf_table.csv) | Tabla consolidada. |
| [`rf_confirm.json`](../../experiments/results/random_forest/rf_confirm.json) | CV repetida y holdout de las configuraciones finalistas. |

Los módulos escriben los resultados nuevos en `experiments/.local/random_forest/` y reutilizan `experiments/.local/preprocessed/` por hash de datos y configuración.

### Reproducción de embeddings

Ejecuta primero las celdas de preprocesado del [cuaderno de Random Forest](../../experiments/notebooks/random_forest_training.ipynb). El banco completo lee esas particiones de `experiments/.local/preprocessed/`. El ejecutor puede usar pesos locales para MiniLM y guarda su caché y resultados en `experiments/.local/embeddings/` y `experiments/.local/random_forest/embeddings/`, respectivamente.

```sh
python -m experiments.scripts.rf_embeddings --checkpoint RUTA_AL_CHECKPOINT
```

Para el control E5, ejecuta `python -m experiments.scripts.rf_embeddings intfloat/multilingual-e5-small` con la variable de entorno `EMB_PREFIX` ajustada a `query: ` (incluido el espacio final). [Configuración archivada de E5](../../experiments/results/random_forest/embeddings/rf_embeddings_results_intfloat__multilingual-e5-small.json)

El entorno histórico usó Python 3.14, `sentence-transformers` 6.3.0, `transformers` 5.15.0, `tokenizers` 0.22.2, `huggingface-hub` 1.28.0 y `torch` 2.15.0 en CPU.

### Implementación y verificación histórica

- [`RelationFeatures`](../../src/nli_project/features/relation_features.py) añadió nombres de salida y validación de columnas. Su batería histórica cubrió 13 casos, incluidos negaciones, longitudes, números, huecos, valores no negativos, clonado y entrada de una sola columna.
- El [pipeline](../../src/nli_project/training/training.py) incorporó chi-cuadrado con `SELECT_K_FEATURES = 300` y el bloque de relación. Las ablaciones usan `select_k=None` o `use_relation_features=False`.
- La verificación histórica registró 66 pruebas en verde. Incluyó ajuste de la selección solo con train, reproducibilidad y contrato del transformador de relación.
- El cuaderno reejecutado registró 0,513303 en CV por fila, 0,5151 en CV agrupada y 0,520627 en accuracy de holdout (macro-F1 de 0,5193). La entrega de Random Forest tenía 5.195 filas válidas. [Métricas TF-IDF](../../experiments/results/random_forest/tfidf) · [CSV de Random Forest](../../experiments/submissions/random_forest/submission.csv)
