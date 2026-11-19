# Contradictory, My Dear Watson

Proyecto de Machine Learning sobre inferencia de lenguaje natural multilingüe. Clasificamos cada par premisa–hipótesis como implicación (0), neutralidad (1) o contradicción (2).

## Qué probamos

| Modelo | Evaluación local | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Random Forest con TF-IDF, POS y rasgos de relación | Validación agrupada por premisa | 51,51 % | 51,40 % |
| CatBoost con texto original → traducido | Validación agrupada, semilla 42 | 52,19 % → 54,82 % | 52,08 % → 54,82 % |
| MiniLM multilingüe con voto MLP + XGBoost | Validación agrupada, media de tres semillas | 62,14 % | 61,84 % |

[Random Forest](results_random_forest/grouped_cv.json) · [Comparación de traducción](results_translation_cv/results.json) · [Comparación de votos](results_dense_generalization/grouped_summary.csv)

El [flujo de experimentación](workflow_javi_aitor.ipynb) reúne las variantes y la traducción. El [notebook histórico de Random Forest](random_forest_training.ipynb) conserva la ejecución con los componentes de preprocesamiento y representación del equipo. Las traducciones guardadas permiten volver a comparar con el mismo texto sin pedir nuevas respuestas a Ollama.

## Predicciones para Kaggle

[kaggle_minilm.ipynb](kaggle_minilm.ipynb) usa el texto original en los 15 idiomas. Verifica los datos oficiales y los archivos del checkpoint público MiniLM, corrige la clase del tokenizador y ajusta MLP y XGBoost con semilla 42. El voto promedia las probabilidades de los dos modelos. El test solo se usa para calcular predicciones. El notebook guarda `submission.csv` tras verificar las 5.195 filas, las tres etiquetas posibles y el orden exacto de `sample_submission.csv`.

En la versión 6 de Kaggle, Transformers cargó WordPiece porque faltaba `tokenizer_class`. Con `PreTrainedTokenizerFast` se recuperó Unigram y se reprodujeron los IDs de tokens y los embeddings locales sobre 128 pares. La coincidencia de etiquetas del test con la referencia local subió de **2.211/5.195 (42,56 %)** a **4.885/5.195 (94,03 %)**. Quedan 310 discrepancias. Kaggle usó XGBoost en CPU, frente a CUDA en el protocolo local, y el entorno de librerías difiere. Las probabilidades históricas del test local no se conservaron, por lo que no podemos atribuir esas discrepancias a un clasificador. El 94,03 % mide coincidencia entre ejecuciones. La puntuación de competición se obtiene cuando Kaggle procesa un envío.

Para desarrollo local, instala las dependencias con `python -m pip install -r requirements.txt`. El notebook MiniLM requiere los tres CSV oficiales en `data/` y el [checkpoint usado en Kaggle](https://www.kaggle.com/datasets/atsunorifujita/paraphrasemultilingualminilml12v2) descomprimido en `.local/minilm-checkpoint/`. Los parámetros están en el [protocolo](results_dense_generalization/protocol.json) y las versiones efectivas se registran en `minilm_run.json`. Para comprobar un CSV existente, ejecuta `python minilm_submission.py ruta/submission.csv data/sample_submission.csv`.
