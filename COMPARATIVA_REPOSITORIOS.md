# Qué añade `iteration_2` al proyecto compartido de Machine Learning

Esta guía compara [Aitor42/0-AiDea en `c18c886`](https://github.com/Aitor42/0-AiDea/tree/c18c88655f1145064c0d45fac0796be239e74463) con el directorio `0-AiDea/` de [Aitor42/training, rama `iteration_2`](https://github.com/Aitor42/training/tree/iteration_2/0-AiDea). La base publicada de la rama es `d94faf5`. La comparación incluye también el trabajo local posterior, como la prueba de MiniLM con traducción y el paso a un flujo solo desde notebooks. Los documentos, apuntes y ejercicios de clase que están en la raíz de `training` pertenecen al entorno de trabajo. Los bancos por modelo de esa raíz sí se incluyen porque evalúan este proyecto.

## 1. Qué aporta el repositorio compartido

### El problema, con un ejemplo

El objetivo es que un programa lea dos frases y decida si la segunda se deduce de la primera, la contradice o aporta algo que la primera deja sin resolver. La primera frase se llama **premisa** y la segunda **hipótesis**.

| Premisa | Hipótesis | Respuesta esperada |
| --- | --- | --- |
| Hay dos gatos en el sofá. | Hay gatos en el sofá. | **Implicación.** Si hay dos, hay gatos. |
| Hay dos gatos en el sofá. | No hay gatos en el sofá. | **Contradicción.** Las dos afirmaciones no pueden ser ciertas a la vez. |
| Hay dos gatos en el sofá. | Los gatos son negros. | **Neutralidad.** La primera frase no nos dice el color. |

Son ejemplos inventados para entender la tarea. En el proyecto, cada fila tiene un **ID**, que identifica el par, y los ejemplos de entrenamiento incluyen una **etiqueta**, que es la respuesta esperada. Entrenar significa mostrarle al modelo muchas parejas con su respuesta para que aprenda patrones. Evaluar significa pedirle respuestas para parejas reservadas y compararlas con sus etiquetas. El test de la competición se usa para generar la entrega y viene sin etiquetas públicas.

### Cómo se reparten el trabajo los dos repositorios

El repositorio del equipo ya contiene los datos oficiales de la competición, el análisis exploratorio, el preprocesador de Alba y el vectorizador y wrapper de Félix. Comprobé que los tres CSV, los tres módulos y sus notebooks compartidos tienen el mismo contenido en ambos proyectos. Dentro de los archivos comunes, `README.md` y `.gitignore` son los que difieren en `training`. El README refleja la investigación de Random Forest y las exclusiones locales cubren el entorno y los artefactos generados.

| Componente compartido | Qué hace |
| --- | --- |
| `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` | Contienen 12.120 pares etiquetados, 5.195 pares sin etiqueta y la plantilla de entrega. La tarea distingue implicación, neutralidad y contradicción en 15 idiomas. |
| `preprocesser.py` | Limpia, tokeniza, filtra palabras, reduce tokens y puede etiquetar POS. `language=None` conserva todas las filas. El idioma del procesamiento depende del resto de opciones. |
| `vectorizer.py` y `builder.py` | Representan el par y coordinan el ajuste en entrenamiento. El mismo wrapper transforma validación con el vocabulario y demás parámetros ya aprendidos. |
| `eda.ipynb`, `project.ipynb` y notebooks `*_demo.ipynb` | Explican los datos y demuestran los componentes compartidos. |

## 2. Qué añade `iteration_2`

### El recorrido de una fila por nuestros archivos

1. **Leemos las dos frases.** El notebook carga la tabla de datos. La premisa «Hay dos gatos en el sofá» entra junto a su hipótesis, su ID y su idioma. El ID nos permite seguir el mismo par hasta la predicción final.
2. **Podemos probar una copia en inglés.** [`FINAL/translator.py`](FINAL/translator.py) añade una columna para cada frase traducida. La fila conserva las frases originales, su idioma y su respuesta esperada. Así podemos comparar dos versiones del mismo par. El notebook puede usar una respuesta simulada para demostrar este paso sin Ollama.
3. **Preparamos el texto.** [`preprocessing.py`](preprocessing.py) define recetas que aplica el preprocesador de Alba. Por ejemplo, la receta neutral divide por espacios y conserva acentos. [`FINAL/permutator.py`](FINAL/permutator.py) prueba varias recetas sobre las mismas filas, como conservar las mayúsculas o pasarlas a minúsculas. Devuelve una ficha con la configuración usada y la ruta del archivo que contiene las filas procesadas. El vocabulario del modelo se aprende más adelante.
4. **Pasamos de palabras a números.** El vectorizador de Félix convierte cada par en una lista de números. Puede representar palabras o fragmentos de caracteres. `pair_diff` mantiene separadas premisa e hipótesis y añade información sobre lo que comparten y lo que cambia. MiniLM, por su parte, representa cada frase con un vector de números generado por un modelo de lenguaje ya entrenado. Son dos formas distintas de dar al clasificador información sobre el mismo par.
5. **Elegimos un modelo que aprenda a responder.** [`classifiers.py`](classifiers.py) construye el modelo pedido, como Naive Bayes o CatBoost. [`training.py`](training.py) encadena la conversión a números y el clasificador, los ajusta con las respuestas del conjunto de entrenamiento y evalúa sus predicciones en ejemplos reservados. A esta cadena de pasos se le llama *pipeline*.
6. **Comparamos las decisiones.** [`run_experiment_matrix.py`](run_experiment_matrix.py) organiza preguntas como «¿cambia el resultado si usamos el texto en inglés?» o «¿van mejor las palabras o los caracteres?». Usa el entrenamiento del paso anterior y mantiene iguales las filas, los grupos y el modelo al comparar las dos versiones de texto. Guarda métricas y, cuando se activa la evaluación reservada, predicciones vinculadas a IDs.

El **permutador** cambia la receta de preparación. El **vectorizador** cambia la forma de representar el texto con números. El **clasificador** aprende a dar una respuesta. El **comparador** repite el flujo bajo condiciones controladas para medir el efecto de cada cambio.

### Inventario de archivos

| Archivos | Qué resuelven y cómo se conectan |
| --- | --- |
| [`preprocessing.py`](preprocessing.py) | Centraliza las recetas y la caché del preprocesado fila a fila. La receta neutral conserva los 15 idiomas, los acentos y usa espacios sin lematización, stopwords ni POS ingleses. La receta inglesa filtra inglés y usa el modelo local de spaCy. El baseline histórico queda disponible para reproducir sus experimentos. |
| [`FINAL/permutator.py`](FINAL/permutator.py) | Genera variantes o rejillas de parámetros para el preprocesador de Alba. Devuelve metadatos con la configuración, número de filas y ruta del pickle. Preserva orden, IDs, textos, idiomas y etiquetas. Los pickles conservan las listas de tokens y las tuplas POS. |
| [`FINAL/translator.py`](FINAL/translator.py) | Traduce premisa e hipótesis con Ollama local, en lotes de un idioma, y valida los IDs de respuesta. Añade columnas traducidas, copia intactos los textos de origen inglés y conserva los originales y sus etiquetas e idiomas. También valida un CSV existente antes de reutilizarlo. |
| [`classifiers.py`](classifiers.py) y [`experiment_defaults.py`](experiment_defaults.py) | Una fábrica construye los modelos por nombre y configuración. Los hiperparámetros de referencia son compartidos por los bancos. Incluye Naive Bayes multinomial, Complement y Bernoulli, árbol, Random Forest, kNN, MLP, XGBoost, LightGBM y CatBoost. `FINAL/models.py` mantiene el import de compatibilidad. |
| [`pos_features.py`](pos_features.py) y [`relation_features.py`](relation_features.py) | Añaden proporciones de categorías POS y señales de relación entre frases. Las reglas de relación incluyen negaciones inglesas. En la comparación neutral ambos bloques se desactivan en los dos brazos y se pueden habilitar por separado en un experimento específico. |
| [`training.py`](training.py) | Construye el pipeline, crea particiones, hace validación cruzada y evalúa el holdout. Ajusta vocabulario, IDF y selección chi-cuadrado dentro de cada fold. Calcula accuracy, macro-F1, matriz de confusión y métricas por idioma, y prepara el formato de entrega. |
| [`run_algorithm_comparison.py`](run_algorithm_comparison.py) y [`run_experiment_matrix.py`](run_experiment_matrix.py) | La comparación de Aitor evalúa clasificadores con el mismo pipeline. La matriz cruza datasets, representaciones y modelos usando ese entrenamiento y evaluador. Verifica IDs, etiquetas, idioma y premisa original al agrupar. Publica al terminar y puede reutilizar una ejecución completa cuando coinciden configuración y huella de las entradas. |
| [`FINAL/compare_ollama.py`](FINAL/compare_ollama.py) | Compara CatBoost con y sin traducción aplicando la misma receta neutral, representación y configuración. Usa el holdout histórico. La generación de un CSV nuevo se solicita explícitamente y cada ejecución usa un directorio nuevo. |
| [`prepare_kaggle_delivery.py`](prepare_kaggle_delivery.py) | Prepara el paquete de datos y código para el cuaderno de Kaggle. Comprueba esquema, etiquetas y correspondencia de IDs con la plantilla. El cuaderno de entrega entrena y genera las predicciones en el formato oficial. |

`parameter_variants.py`, `translation.py`, `compare_ollama.py` y `FINAL/training.py` mantienen imports usados en el trabajo anterior. Los puntos de entrada de consola del flujo actual se retiraron localmente. `requirements.txt`, `pytest.ini` y [`tests/`](tests/) cubren clasificadores, características, particiones, evaluación, generación de variantes, traducción, reutilización de resultados y formato de entrega.

## 3. Bancos de Aitor y experimentos con embeddings

- **Bancos por algoritmo en la raíz de `training`.** `model_experiments.py` contiene la lógica común. Los nueve archivos `<modelo>_experiments.py` y el banco histórico `rf_experiments.py` definen las configuraciones por modelo. Los `*_experiments_results.jsonl` guardan ejecuciones. `comparison_table.py` y `run_all_experiments.py` coordinan la investigación, y `model_comparison.csv` y `model_comparison_base.csv` resumen las mejores configuraciones y la base común. `rf_confirm.py`, `rf_summary.py` y `rf_table.csv` pertenecen al estudio histórico de Random Forest.
- **Embeddings en `0-AiDea/`.** `rf_embeddings.py` y `replay_dense_rf.py` investigan y reproducen Random Forest con embeddings. Los JSON de MiniLM y E5 guardan los resultados originales. `run_dense_comparison.py` compara clasificadores sobre MiniLM, `run_dense_tuning.py` ajusta candidatos y `run_dense_confirmation.py` vuelve a evaluar configuraciones fijadas. `run_dense_generalization.py` usa folds agrupados por premisa. `dense_oof_summary.py` y `dense_vote_comparison.py` resumen predicciones fuera del fold y estudian votos. `dense_spacy_ablation.py` mide el efecto de añadir relación y POS en el subconjunto inglés.
- **MiniLM con traducción.** [`run_minilm_translation.py`](run_minilm_translation.py) es una incorporación local accesible desde el notebook. Reutiliza las traducciones y los embeddings originales validados, vuelve a codificar los textos traducidos y evalúa el mismo voto de MLP y XGBoost sobre las mismas filas y cinco folds. La representación concatena los vectores de premisa e hipótesis, su diferencia absoluta y su producto elemento a elemento.

## 4. Resultados y cómo interpretarlos

- **Accuracy** es el porcentaje de respuestas correctas. Si el modelo acierta 62 de 100 pares, su accuracy es del 62 %.
- **Macro-F1** da el mismo peso a las tres respuestas posibles. Ayuda a ver si un modelo acierta una clase con frecuencia pero se equivoca más en las otras.
- **Validación cruzada y folds.** Dividimos los datos de desarrollo en bloques. Entrenamos con varios y evaluamos con el que queda fuera, turnándonos hasta usar todos. Una división agrupada por premisa mantiene juntas todas las filas que repiten la misma primera frase.
- **Holdout** es otro conjunto de ejemplos reservado para evaluar. En este proyecto hay un holdout histórico que comparte premisas con desarrollo. El cuaderno actual crea su propia reserva agrupada por premisa.
- **Puntos porcentuales (pp).** Pasar de 52 % a 55 % equivale a subir 3 puntos porcentuales.

| Pregunta | Resultado | Evidencia |
| --- | --- | --- |
| ¿Qué clasificador lidera el banco léxico de `main`? | CatBoost, **52,77 %** de accuracy y **52,68 %** de macro-F1 en su mejor configuración. XGBoost obtuvo **52,33 %** de accuracy. | [Tabla de Aitor](../model_comparison.csv) |
| ¿Ayudó traducir con TF-IDF `pair_diff`? | En tres folds agrupados, CatBoost pasó de **52,19 % a 54,82 %**, **+2,63 pp**. Naive Bayes multinomial pasó de **48,50 % a 50,26 %**, **+1,75 pp**. | [Comparación emparejada](results_translation_cv/results.json) |
| ¿Cuál es el mejor resultado multilingüe agrupado? | MiniLM sobre texto original con voto de MLP y XGBoost, **62,14 %** de accuracy y **61,84 %** de macro-F1 en cinco folds y tres semillas. Accuracy de **63,66 %** en origen inglés y **60,12 %** en origen no inglés. | [Resultados agrupados](results_dense_generalization/grouped_summary.csv) |
| ¿Ayudó traducir con MiniLM? | En una semilla, **62,06 % → 61,73 %** global y **60,41 % → 58,71 %** en origen no inglés. Por ahora se conserva el texto original en el pipeline MiniLM. | [Resultados locales](artifacts/minilm_translation_run42/results.json) |
| ¿De dónde salen cifras aún mayores? | El voto de cuatro modelos logró **63,33 %** en el holdout histórico, que comparte premisas con desarrollo. MiniLM con relación y POS obtuvo **64,47 %** en el subconjunto de origen inglés. | [Holdout](results_dense_confirmation/holdout.csv) · [Subconjunto inglés](results_dense_generalization/dense_spacy_ablation/summary.csv) |

Los resultados de bancos con particiones distintas sirven para contextualizar. El efecto de traducir se estima comparando los dos brazos del mismo experimento. Las cifras anteriores son de evaluación local. [Informe experimental](JAVI_AITOR.md)

| Directorios de resultados | Contenido |
| --- | --- |
| `results_random_forest/`, `results_algorithms/` | Configuraciones, métricas, matrices de confusión y entregas históricas. |
| `results_javi_aitor_cv/`, `results_translation_cv/`, `results_ollama_catboost/` | Variantes léxicas, resultados agrupados con y sin traducción y predicciones del holdout histórico. |
| `results_dense/`, `results_dense_tuning/`, `results_dense_confirmation/`, `results_dense_votes/`, `results_dense_generalization/` | Predicciones fuera del fold, ajustes, votos, validación agrupada y ablación inglesa. |
| `results_kaggle_translated/` | CSV de entrega y manifiesto de ejecución local. La puntuación en Kaggle sigue sin verificar. |
| `artifacts/` | Particiones y cachés base, más material generado localmente como traducciones, pickles del permutador, resultados del notebook y paquete de Kaggle. Las traducciones y resultados locales recientes pueden faltar en una copia limpia. |

## 5. Cómo probarlo paso a paso desde el notebook

Abre [`workflow_javi_aitor.ipynb`](workflow_javi_aitor.ipynb) en VS Code y elige como kernel el Python del entorno `0-AiDea/.venv`. El kernel es el Python que ejecuta cada celda. Necesita el modelo local de spaCy `en_core_web_sm`. Puedes abrir el cuaderno desde `training` o desde `0-AiDea`.

Para la primera prueba deja `MODE = "smoke"` y mantén `TRY_OLLAMA`, `GENERATE_FULL_TRANSLATIONS` y `RUN_MINILM` en `False`. Ejecuta las celdas de preparación y datos de arriba abajo. Aparecerán **105 filas de desarrollo y 36 de validación**. Puedes ejecutar el resto en orden o elegir una demostración.

| Quiero probar | Qué celda ejecutar y qué mirar |
| --- | --- |
| El preprocesador | **Preprocesador.** Compara una fila antes y después. La demostración toma un ejemplo por idioma y muestra cuántas filas conserva cada receta. |
| El traductor | **Traducción por lotes.** Comprueba que se añaden las columnas traducidas y que los IDs y frases originales siguen ahí. La traducción de ejemplo está simulada. Con Ollama y el modelo ya disponibles, activa `TRY_OLLAMA`, vuelve a ejecutar la preparación y repite esta celda para traducir de verdad el ejemplo. |
| El permutador | **Permutador.** Verás dos recetas, sus identificadores y su número de filas. La celda carga una variante con los metadatos que devolvió el permutador y enseña los primeros tokens. |
| El vectorizador y el clasificador | **Vectorizador y modelos.** La primera celda muestra cuántas filas y columnas numéricas produce cada representación. La siguiente ajusta un modelo pequeño y muestra una tabla con el ID, la respuesta correcta y la predicción. |
| La comparación y los errores | **Comparación reproducible** y luego **Evaluación reservada**. Deja `RUN_EXPERIMENTS` y `SHOW_HOLDOUT` en `True`. Verás accuracy, macro-F1 y tiempos. En la matriz de confusión las filas muestran la respuesta correcta y las columnas la predicción. Los valores fuera de la diagonal son errores. |
| MiniLM | **MiniLM con y sin traducción.** La celda muestra la tabla si existen resultados guardados. Si faltan, puedes activar `RUN_MINILM` para ejecutar el experimento. Para hacerlo hacen falta el CSV traducido, los pesos locales de MiniLM y CUDA para XGBoost. |
| MiniLM usando el preprocesador de Alba | **Ajuste de MiniLM con el preprocesador de Alba.** Activa `TUNE_MINILM` para comparar espacios y spaCy antes de generar los vectores. Esta prueba usa toda la partición de desarrollo, incluso con el cuaderno en modo rápido. La tabla compara la profundidad de cada modelo y cuántas características se conservan. |

Si quieres probar de forma explícita solo el original, pon `USE_TRANSLATIONS = False`. Con `True`, el cuaderno usa el CSV traducido si es válido. Si falta, continúa con el texto original e indica que omitió esa parte. Después de cambiar una opción, vuelve a ejecutar la preparación y las celdas que la usan.

Cuando hayas probado el flujo rápido, `MODE = "full"` amplía la comparación a las particiones completas y añade CatBoost. Si quieres que en ese modo se genere un CSV traducido que aún no existe, activa también `GENERATE_FULL_TRANSLATIONS`. El experimento neutral completo queda por ejecutar. La última comprobación local pasó 135 pruebas y ejecutó el notebook rápido.

Consulta [`comparacion_javi_aitor.ipynb`](comparacion_javi_aitor.ipynb) para las tablas históricas, [`random_forest_training.ipynb`](random_forest_training.ipynb) para el estudio de Random Forest o [`kaggle_submission_final.ipynb`](kaggle_submission_final.ipynb) para generar la entrega con el paquete preparado.

## 6. Qué falta para integrarlo en el repositorio del equipo

- **Javi y Aitor.** Los issues de permutador, traducción, modelos y entrenamiento siguen abiertos en el repositorio compartido. La implementación experimental y el notebook están desarrollados en `training`. [#7](https://github.com/Aitor42/0-AiDea/issues/7) · [#11](https://github.com/Aitor42/0-AiDea/issues/11) · [#12](https://github.com/Aitor42/0-AiDea/issues/12) · [#13](https://github.com/Aitor42/0-AiDea/issues/13)
- **Akil.** Queda confirmar si el bucle recibe DataFrames o matrices, cómo recibe el estimador y en qué punto ajusta el vectorizador dentro de cada fold. Para el evaluador hay que acordar orden de clases, métricas y cómo devolver las predicciones vinculadas a IDs e idiomas de origen. El banco de Aitor permite probar las piezas mientras se acuerda esa conexión. [Bucle #9](https://github.com/Aitor42/0-AiDea/issues/9) · [Evaluador #10](https://github.com/Aitor42/0-AiDea/issues/10)
- **Dario.** Queda confirmar la entrada y salida del balanceo. Se aplicará dentro de cada fold de ajuste, preservando los ejemplos reservados y el mismo criterio para original y traducido. [Balanceo #8](https://github.com/Aitor42/0-AiDea/issues/8)
