# Guía del proyecto

[Mapa del repositorio](../README.md) · [Instalación y reproducción](reproducibility.md)

## Índice

- [Problema y datos](#problema-y-datos)
- [Cómo se relacionan los componentes](#cómo-se-relacionan-los-componentes)
- [Entrenamiento, validación y métricas](#entrenamiento-validación-y-métricas)
- [Experimentos y resultados](#experimentos-y-resultados)
- [Configuración seleccionada para Kaggle](#configuración-seleccionada-para-kaggle)

## Problema y datos

Dada una **premisa** y una **hipótesis**, clasificamos su relación como implicación (0, *entailment*), neutralidad (1) o contradicción (2).

| Conjunto | Pares | Contenido |
| --- | ---: | --- |
| Train oficial | 12.120 | Pares etiquetados en 15 idiomas. |
| Test oficial | 5.195 | Pares para inferencia. La plantilla fija el orden de entrega. |

Los dos CSV contienen `id`, `premise`, `hypothesis`, `lang_abv` y `language`. El train añade `label`. [Datos oficiales de la competición](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data)

El cargador valida las columnas y las etiquetas. El split histórico estratificado de desarrollo (9.696) y holdout (2.424), con semilla 42, sigue disponible con sus IDs y orden. Comparte 1.092 premisas distintas. Se conserva para replicar las comparaciones originales y se etiqueta como holdout histórico. [Carga y particiones](../src/nli_project/data/dataset.py) · [Resumen del split histórico](../experiments/results/validation/historical_split/split_summary.json)

## Cómo se relacionan los componentes

1. **Carga y preprocesado.** El cargador lee los CSV y puede anexar la traducción por ID. El preprocesador genera columnas nuevas y conserva los textos originales para revisar ejemplos y agrupar premisas.
2. **Representación.** El vectorizador y los bloques de POS y relación convierten las columnas procesadas en una matriz de características. `NLIDatasetBuilder` reúne preprocesado y vectorización para obtener los datos transformados directamente. [Constructor](../src/nli_project/data/builder.py)
3. **Ajuste y evaluación.** El pipeline ajusta las transformaciones aprendidas y el clasificador dentro de cada fold. Los ejecutores guardan configuraciones, métricas y predicciones por ID. [Pipeline y CV](../src/nli_project/training/training.py)

El flujo de Kaggle codifica el texto original con MiniLM congelado. MLP y XGBoost reciben la misma representación y ordenan sus probabilidades por clase. El voto promedia esas probabilidades y el exportador valida las predicciones con la plantilla oficial. [Cuaderno de Kaggle](../experiments/notebooks/kaggle_minilm.ipynb)

### Preprocesamiento y traducción

El preprocesador configura minúsculas, acentos y filtrado por idioma. Puedes elegir tokenización por espacios, NLTK o spaCy, aplicar stemming o lematización (WordNet o spaCy), filtrar stopwords con listas de scikit-learn, NLTK o propias, conservar negaciones y guardar las etiquetas POS junto a los tokens.

- **Receta histórica.** Conversión a minúsculas y conservación de acentos, con tokenización, lematización y POS de spaCy, y todas las stopwords. Aplicar el modelo inglés a otros idiomas limita la interpretación de esos rasgos.
- **Receta multilingüe neutral.** Conserva las filas de todos los idiomas y tokeniza por espacios sin reducción, filtrado de stopwords ni POS. [Recetas y catálogo de variantes](reports/parameter_variants.md)

El traductor envía lotes del mismo idioma a Ollama y recibe JSON asociado a los IDs. Añade `premise_translated` y `hypothesis_translated` y conserva el orden, los originales, idiomas y etiquetas. Copia los pares de origen inglés. Los experimentos emparejados cambian las columnas que recibe el preprocesador y siguen agrupando por la premisa original. Cambios de negación, cuantificador o entidad pueden alterar la clase del par. [Traductor](../src/nli_project/translation/ollama.py) · [Procedencia y regeneración](reproducibility.md#traducción) · [Comparaciones con traducción](reports/team_experiments.md#comparación-agrupada-con-y-sin-traducción)

### Representaciones y clasificadores

La representación clásica usa conteos, presencia, TF-IDF de palabras o caracteres y variantes de n-gramas. `pair_separate` mantiene bloques separados de premisa e hipótesis. `pair_diff` añade diferencias absolutas y solapamiento. Las características auxiliares resumen proporciones POS y señales de relación, entre ellas negaciones inglesas. Chi-cuadrado selecciona señales no negativas cuando la receta lo activa. El vectorizador también implementa SVD y embeddings. [Vectorizador](../src/nli_project/features/vectorizer.py) · [POS](../src/nli_project/features/pos_features.py) · [Relación](../src/nli_project/features/relation_features.py)

MiniLM representa cada frase con 384 componentes, pooling por media y normalización L2. La representación densa del voto concatena la premisa, la hipótesis, la diferencia absoluta y el producto por componentes. Resultan 1.536 rasgos por par. El estudio de Random Forest compara también embeddings multilingües E5, otras combinaciones de los vectores y el baseline léxico. [Características MiniLM](../src/nli_project/features/minilm.py) · [Estudio de Random Forest](reports/random_forest_research.md)

La fábrica entrega estimadores sin ajustar. El banco clásico compara Multinomial, Complement y Bernoulli Naive Bayes, kNN, árbol de decisión, Random Forest, CatBoost, LightGBM, XGBoost y MLP. La comparación léxica controla la representación y el protocolo para estudiar el efecto del clasificador. Las pruebas densas evalúan también el voto como configuración propia. [Fábrica](../src/nli_project/models/classifiers.py) · [Configuraciones del catálogo clásico](../experiments/configs/experiment_defaults.py)

## Entrenamiento, validación y métricas

| Protocolo | Uso |
| --- | --- |
| CV por fila | Comparaciones clásicas históricas. Distintas hipótesis de la misma premisa pueden quedar en ajuste y validación. |
| CV agrupada por premisa original | `StratifiedGroupKFold` mantiene juntas las filas de cada premisa e intenta conservar las proporciones de clase. MiniLM usó 5 folds por semilla (42, 7 y 2024), con cero premisas compartidas en las 15 particiones. Las comparaciones léxicas de traducción usaron 3 folds con semilla 42. |
| Holdout histórico | Evaluación sobre la partición reservada del split histórico, con el solapamiento descrito en [Problema y datos](#problema-y-datos). |

La normalización fila a fila se puede guardar antes de dividir. El pipeline se clona por fold y aprende vocabulario, IDF, selección y parámetros del modelo solo con el bloque de ajuste. Los experimentos de traducción comprueban los IDs y mantienen los grupos y etiquetas. En MiniLM se reutiliza el encoder congelado y se ajustan los clasificadores por fold. Los archivos OOF (*out of fold*) guardan las probabilidades en las filas reservadas de cada fold. [Protocolo MiniLM](../experiments/results/validation/minilm_grouped/protocol.json) · [Auditoría de splits](../experiments/results/validation/minilm_grouped/split_audit.csv)

- **Accuracy**. Proporción de aciertos.
- **Macro-F1**. Promedio del F1 de las clases con el mismo peso. El F1 combina precisión y exhaustividad.
- **Matriz de confusión**. Cruza clases reales y predichas.
- **Métricas por idioma**. Se calculan por lengua de origen y se interpretan junto al número de ejemplos. Se registran también tiempos y, según el protocolo, dispersión entre folds o semillas. [Implementación de métricas](../src/nli_project/evaluation/metrics.py)

## Experimentos y resultados

| Familia | Pregunta | Evidencia |
| --- | --- | --- |
| Bancos clásicos | Rendimiento por clasificador y representación léxica | [Comparativas y bancos](../experiments/results/classical_models) |
| Random Forest | Efecto de representación, POS, rasgos de relación, hiperparámetros y embeddings | [Informe de Random Forest](reports/random_forest_research.md) · [Resultados de RF](../experiments/results/random_forest) |
| MiniLM exploratorio | Comparación densa, ajustes, confirmación y ablación con spaCy | [Resultados MiniLM](../experiments/results/minilm) |
| Traducción | Original frente a inglés en protocolos emparejados léxicos y MiniLM | [Informe de preprocesado y traducción](reports/team_experiments.md) · [Traducciones, métricas y predicciones](../experiments/results/translation) |
| Ensembles y validación | Votos, auditoría de particiones y pruebas agrupadas | [Votos](../experiments/results/ensembles) · [Validación](../experiments/results/validation) |

| Resultado archivado | Accuracy | Protocolo |
| --- | ---: | --- |
| CatBoost TF-IDF, original → traducido | 52,19 % → 54,82 % | [CV agrupada léxica](reports/team_experiments.md#comparación-agrupada-con-y-sin-traducción) |
| Naive Bayes, original → traducido | 48,50 % → 50,26 % | [CV agrupada léxica](reports/team_experiments.md#comparación-agrupada-con-y-sin-traducción) |
| Voto MiniLM, original → traducido | 62,06 % → 61,73 % | [CV agrupada con semilla 42](reports/team_experiments.md#traducción-con-minilm) |
| Voto MiniLM de MLP y XGBoost | 62,14 % | CV agrupada con 3 semillas. Macro-F1 de 61,84 %. Accuracy de 63,66 % en inglés y 60,12 % en el resto. [Resumen de la validación](../experiments/results/validation/minilm_grouped/grouped_summary.csv) |
| Voto MiniLM de cuatro modelos | 63,33 % | [Holdout histórico](../experiments/results/minilm/confirmation/holdout.csv) |
| MiniLM con POS y relación | 64,47 % | [Ablación sobre el subconjunto inglés](../experiments/results/minilm/spacy_ablation/summary.csv) |

## Configuración seleccionada para Kaggle

| Componente | Configuración |
| --- | --- |
| Entrada y encoder | Texto original. SHA-256 de datos y pesos, tokenizador Unigram con longitud máxima 128, codificación única por frase distinta. |
| MLP | 256 unidades y parada temprana. |
| XGBoost | 500 estimadores, profundidad 3, tasa 0,05. CPU en el cuaderno de entrega y CUDA en la prueba agrupada histórica. |
| Ajuste y voto | Todo el train oficial, semilla 42. Se elige la clase con mayor probabilidad media tras dar el mismo peso a ambos modelos. |

El cuaderno escribe `id,prediction` y valida desde disco los 5.195 IDs únicos en el orden oficial, con etiquetas 0, 1 o 2. El manifiesto registra versiones, hashes y parámetros. El SHA-256 del CSV corregido coincide con el [manifiesto de la ejecución archivada](../experiments/submissions/minilm_tokenizer_corrected/execution.json). Comparte 4.885 etiquetas (94,03 %) con la ejecución local de referencia. Esa cifra mide la coincidencia entre ejecuciones. La puntuación de competición de ese CSV quedó pendiente en la PR. [Entrega MiniLM](../experiments/submissions/minilm_tokenizer_corrected) · [Entrega histórica de CatBoost traducido](../experiments/submissions/catboost_translated) · [PR #14](https://github.com/Aitor42/0-AiDea/pull/14)

Los pasos y comandos están en la [guía de generación de entregas](reproducibility.md#generación-de-entregas).
