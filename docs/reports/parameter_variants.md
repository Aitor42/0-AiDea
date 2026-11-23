# Variantes de preprocesado

> Informe histórico de la PR #14. Las rutas se han actualizado. Los estados de tareas, el vocabulario de «final» y las conclusiones corresponden a la fecha del informe. La guía describe el flujo vigente.

[Guía del proyecto](../PROJECT_GUIDE.md)

El [cuaderno de trabajo](../../experiments/notebooks/workflow_javi_aitor.ipynb) muestra dos recetas explícitas de `preprocessing.py`. `MULTILINGUAL_PREPROCESSING_PARAMS` conserva todas las filas y usa tokenización por espacios, sin lematización, stopwords ni POS ingleses. `ENGLISH_PREPROCESSING_PARAMS` filtra filas inglesas y usa el modelo de spaCy para lemas y POS. `language=None` solo conserva filas. El preprocesador de Alba requiere el modelo local de spaCy incluso con la receta neutral.

`parameter_variants.py` genera un baseline y cinco comparaciones pequeñas a
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

La tokenización whitespace se combina con reducción desactivada porque aplicar
lematización spaCy a tokens ya agrupados por espacios produciría una
comparación difícil de interpretar. `top_n` se omite porque solo controla
tablas descriptivas y no interviene en `preprocess_dataset`. `ngram_orders`
genera columnas auxiliares que el vectorizador actual no consume. Tampoco se
barre una secuencia de valores continuos. Las diferencias en limpieza y
reducción son decisiones categóricas y los parámetros de `NLITextVectorizer`
se exploran por separado.

Cada dataset se publica de forma atómica como pickle y `manifest.json` en su
propia carpeta bajo `artifacts/parameter_variants/`. La caché de cómputo sigue en
`artifacts/preprocessed/`, usando `preprocessing.preprocess_partition`, que
llama al preprocesador de Alba y se usa también desde el banco de Aitor. Así pueden
reutilizarse los resultados entre ejecuciones sin ajustar vocabulario ni IDF
fuera de cada fold. El clasificador debe recibir el DataFrame procesado y
ajustar su vectorizador dentro de la validación cruzada.

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
