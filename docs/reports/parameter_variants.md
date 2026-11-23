# Variantes de preprocesado

[Mapa del repositorio](../../README.md) · [Guía del proyecto](../PROJECT_GUIDE.md#preprocesamiento-y-traducción) · [Comparación experimental](team_experiments.md#comparación-de-variantes-de-preprocesado)

## Índice

[Recetas del cuaderno](#recetas-del-cuaderno) · [Catálogo de variantes](#catálogo-de-variantes) · [Materialización y caché](#materialización-y-caché)

## Recetas del cuaderno

El [cuaderno de trabajo](../../experiments/notebooks/workflow_javi_aitor.ipynb) usa dos recetas de [preprocessing.py](../../src/nli_project/preprocessing/preprocessing.py).

| Receta | Filas | Tokenización y rasgos |
| --- | --- | --- |
| `MULTILINGUAL_PREPROCESSING_PARAMS` | Todos los idiomas | Espacios, sin reducción, filtrado de stopwords ni POS. |
| `ENGLISH_PREPROCESSING_PARAMS` | Solo inglés | spaCy para tokenización, lemas y POS. |

`language=None` conserva las filas de todos los idiomas. La selección de reglas lingüísticas se configura por separado. El preprocesador requiere el modelo local de spaCy incluso con la receta neutral. [Instalación de recursos lingüísticos](../reproducibility.md#instalación-y-dependencias)

## Catálogo de variantes

[parameter_variants.py](../../src/nli_project/preprocessing/parameter_variants.py) genera un baseline y cinco comparaciones pequeñas a
partir de las opciones reales de `NLITextPreprocesser`. Cada variante tiene un
ID determinista derivado de la configuración de preprocesado. Las variantes
equivalentes para las columnas del modelo se guardan una
sola vez.

El catálogo mantiene `language=None`, que procesa todos los idiomas y conserva
el mismo orden, identificadores, textos y etiquetas de entrada. Si se entrega
un DataFrame traducido, ese contenido también forma parte de la huella de
origen y se usa para materializar la variante. Por omisión se carga el train
oficial.

| Variante | Cambio respecto al baseline |
|---|---|
| `baseline` | Configuración histórica en `preprocessing.PREPROCESSING_PARAMS`. Procesa todas las filas con reglas inglesas. |
| `without_reduction` | Desactiva stemming y lematización |
| `snowball_stemming` | Usa Snowball para reducir palabras |
| `sklearn_stopwords` | Elimina stopwords conservando negaciones |
| `whitespace_no_reduction` | Tokenización whitespace sin reducción |
| `strip_accents` | Elimina marcas diacríticas |

La variante por espacios desactiva la reducción para evitar aplicar lematización de spaCy a tokens ya segmentados. `top_n` solo afecta las tablas descriptivas. `ngram_orders` genera columnas auxiliares que el vectorizador actual no consume. La rejilla cubre opciones categóricas de limpieza y reducción. Los parámetros del [vectorizador](../../src/nli_project/features/vectorizer.py) se estudian por separado.

## Materialización y caché

Cada variante se guarda de forma atómica como `dataset.pkl` y `manifest.json` en su propia carpeta bajo `experiments/.local/parameter_variants/`. `preprocessing.preprocess_partition` comparte la caché `experiments/.local/preprocessed/` con el banco de experimentos. El clasificador recibe el DataFrame procesado y ajusta vocabulario, IDF y selección dentro de cada fold.

```python
from nli_project.preprocessing.parameter_variants import grid_variants, materialize_variants, load_variant_dataset

datasets = materialize_variants()
baseline = load_variant_dataset(datasets[0])
```

Para procesar textos traducidos, pasa el DataFrame del CSV y configura
`base_params={"columns": ("premise_translated", "hypothesis_translated")}`.
La premisa original queda disponible para agrupar los folds.

Para generar una rejilla desde el cuaderno, llama a
`grid_variants({"tokenizer": ["whitespace", "spacy"], "reduction": ["none", "snowball"]})`
y pasa las cuatro variantes devueltas a `materialize_variants`. El límite por defecto es 12.
