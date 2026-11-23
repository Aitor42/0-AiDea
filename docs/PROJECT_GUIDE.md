# Guía del proyecto

## Problema y datos

Clasificamos el significado de una **hipótesis** dada una **premisa**. La etiqueta 0 expresa implicación (*entailment*), la 1 neutralidad y la 2 contradicción. Es una clasificación nominal supervisada sobre pares. El conjunto etiquetado tiene 12.120 pares en 15 idiomas. El test oficial tiene 5.195 pares y su plantilla establece el orden de entrega. Las columnas son `id`, `premise`, `hypothesis`, `lang_abv` y `language`, además de `label` en entrenamiento. El test aporta textos e IDs para inferencia. [Datos oficiales de la competición](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data)

El cargador valida las columnas y las etiquetas. El split histórico estratificado de desarrollo (9.696) y holdout (2.424), con semilla 42, sigue disponible con sus IDs y orden. Comparte 1.092 premisas distintas. Se conserva para replicar las comparaciones originales y se etiqueta como holdout histórico. [Carga y particiones](../src/nli_project/data/dataset.py) · [Resumen del split histórico](../experiments/results/validation/historical_split/split_summary.json)

## Cómo se relacionan los componentes

El flujo léxico pasa de CSV a un DataFrame por pares. Puede anexar una traducción alineada por ID. El preprocesador genera columnas procesadas y preserva los textos originales para auditoría y agrupamiento. El vectorizador y los bloques POS y de relación convierten esas columnas en una matriz. El pipeline aprende las transformaciones supervisadas y ajusta el clasificador dentro de cada fold. El evaluador resume el resultado y los ejecutores guardan configuraciones, métricas y predicciones por ID. El constructor `NLIDatasetBuilder` ofrece el flujo de preprocesado y vectorización para quien necesita los datos transformados directamente. [Constructor](../src/nli_project/data/builder.py) · [Pipeline y CV](../src/nli_project/training/training.py)

El flujo MiniLM de Kaggle usa los textos originales y un encoder congelado. La representación densa alimenta los dos clasificadores, que comparten el orden de las clases. El voto combina probabilidades y el exportador valida las predicciones frente a la plantilla oficial. [Notebook de Kaggle](../experiments/notebooks/kaggle_minilm.ipynb)

### Preprocesamiento y traducción

El preprocesador de Alba permite configurar minúsculas, acentos, idioma de entrada, tokenización por espacios, NLTK o spaCy, y selección de stopwords de scikit-learn, NLTK o una lista propia. Puede conservar negaciones al filtrar. Dispone de técnicas de stemming y de lematización con WordNet o spaCy. Las etiquetas POS se conservan junto a los tokens cuando se solicitan. La configuración histórica usa minúsculas, conserva acentos, tokeniza y lematiza con spaCy y mantiene POS sin quitar stopwords. Aplicar el modelo inglés a otros idiomas limita la interpretación. La receta neutral mantiene las filas de todos los idiomas y usa espacios sin reducción, stopwords ni POS. El permutador materializa variantes identificables y permite controlar que cada una conserva las filas esperadas. [Recetas](../src/nli_project/preprocessing/preprocessing.py) · [Variantes](reports/parameter_variants.md)

El brazo de traducción envía lotes de un mismo idioma a Ollama y espera JSON estructurado asociado a los IDs. Añade `premise_translated` y `hypothesis_translated` conservando textos originales, idiomas, etiquetas y orden. Los pares de origen inglés se copian. Las comparaciones emparejadas cambian las columnas de entrada al preprocesador y mantienen como grupo la premisa original. En NLI un cambio de negación, cuantificador o entidad puede cambiar la relación. [Traductor](../src/nli_project/translation/ollama.py) · [Procedencia y regeneración](reproducibility.md#traducción)

### Representaciones y clasificadores

La representación clásica usa conteos, presencia, TF-IDF de palabras o caracteres y variantes de n-gramas. `pair_separate` mantiene bloques separados de premisa e hipótesis. `pair_diff` añade diferencias absolutas y solapamiento. Las características auxiliares resumen proporciones POS y señales de relación, entre ellas negaciones inglesas. Chi-cuadrado selecciona señales no negativas cuando la receta lo activa. El vectorizador también implementa SVD y embeddings. [Vectorizador](../src/nli_project/features/vectorizer.py) · [POS](../src/nli_project/features/pos_features.py) · [Relación](../src/nli_project/features/relation_features.py)

MiniLM representa cada frase con 384 componentes, pooling por media y normalización L2. La representación densa del voto concatena la premisa, la hipótesis, la diferencia absoluta y el producto por componentes. Resultan 1.536 rasgos por par. El estudio de Random Forest compara también embeddings multilingües E5, otras combinaciones de los vectores y el baseline léxico. [Características MiniLM](../src/nli_project/features/minilm.py) · [Estudio de Random Forest](reports/random_forest_research.md)

La fábrica genera estimadores sin ajustar. Multinomial, Complement y Bernoulli Naive Bayes ofrecen referencias para rasgos de frecuencia o presencia. kNN compara ejemplos cercanos en la representación y el árbol aporta una referencia de particiones no lineales. Random Forest promedia árboles, mientras que CatBoost, LightGBM y XGBoost entrenan árboles de forma aditiva. MLP aprende interacciones no lineales entre los rasgos densos. La comparación léxica controla la representación y el protocolo para estudiar el efecto del clasificador. El voto final combina dos familias con predicciones distintas y se evalúa como configuración propia. [Fábrica](../src/nli_project/models/classifiers.py) · [Configuraciones del catálogo clásico](../experiments/configs/experiment_defaults.py)

## Entrenamiento, validación y métricas

El banco clásico ajusta y compara sobre desarrollo y reserva el holdout para la evaluación correspondiente. En la validación por fila, distintas hipótesis de la misma premisa pueden caer a ambos lados. `StratifiedGroupKFold` mantiene juntas las filas con la misma premisa original e intenta conservar las proporciones de las clases. El protocolo MiniLM agrupado usa cinco folds en cada semilla (42, 7 y 2024), con cero premisas compartidas en las 15 particiones auditadas. Las comparaciones léxicas emparejadas de traducción citadas aquí usan tres folds agrupados en la semilla 42. [Protocolo MiniLM](../experiments/results/validation/minilm_grouped/protocol.json) · [Auditoría de splits](../experiments/results/validation/minilm_grouped/split_audit.csv)

La normalización por fila puede almacenarse antes de dividir. El pipeline se clona por fold para aprender vocabulario, IDF, selección de características y parámetros del modelo solo a partir del bloque de ajuste. Los experimentos que comparan original y traducción validan la correspondencia por ID y usan los mismos grupos y etiquetas. En MiniLM se puede reutilizar el encoder congelado. Los clasificadores se ajustan por fold, dejando las probabilidades de los ejemplos reservados en ficheros OOF (*out of fold*).

Accuracy es la proporción de aciertos. Macro-F1 da el mismo peso al F1 de cada clase, combinando precisión y exhaustividad. La matriz de confusión muestra qué clases se confunden. Las métricas por idioma deben leerse junto al número de ejemplos y la lengua original. Los informes registran también tiempos y, según el protocolo, dispersión entre folds o semillas. [Implementación de métricas](../src/nli_project/evaluation/metrics.py)

## Catálogo y lecturas recomendadas

| Familia | Pregunta | Evidencia |
| --- | --- | --- |
| Bancos clásicos | Rendimiento por clasificador y representación léxica | [Comparativas y bancos](../experiments/results/classical_models) |
| Random Forest | Efecto de representación, POS, rasgos de relación, hiperparámetros y embeddings | [Resultados de RF](../experiments/results/random_forest) |
| MiniLM exploratorio | Comparación densa, ajustes, confirmación y ablación con spaCy | [Resultados MiniLM](../experiments/results/minilm) |
| Traducción | Original frente a inglés en protocolos emparejados léxicos y MiniLM | [Traducciones, métricas y predicciones](../experiments/results/translation) |
| Ensembles y validación | Votos, auditoría de particiones y pruebas agrupadas | [Votos](../experiments/results/ensembles) · [Validación](../experiments/results/validation) |

CatBoost con TF-IDF agrupado pasó de 52,19 % a 54,82 % de accuracy al traducir, y Naive Bayes de 48,50 % a 50,26 %. MiniLM traducido bajó de 62,06 % a 61,73 % en el contraste de una semilla. El mejor voto registrado en la comparación MiniLM agrupada alcanzó 62,14 % de accuracy y 61,84 % de macro-F1, con 63,66 % de accuracy en inglés y 60,12 % en el resto. El 63,33 % del voto de cuatro modelos procede del holdout histórico. La ablación que añade POS y relación al MiniLM reporta 64,47 % en el subconjunto inglés, con su propio ámbito. [Traducción léxica](../experiments/results/translation/grouped_cv/results.json) · [Traducción MiniLM](../experiments/results/translation/minilm_seed42/results.json) · [Validación agrupada](../experiments/results/validation/minilm_grouped/grouped_summary.csv) · [Holdout histórico](../experiments/results/minilm/confirmation/holdout.csv) · [Ablación](../experiments/results/minilm/spacy_ablation/summary.csv)

## Configuración seleccionada para Kaggle

El notebook usa MiniLM sobre texto original. Verifica SHA-256 de los datos y pesos, valida la configuración del tokenizador Unigram con longitud máxima 128 y codifica cada frase distinta una vez. Entrena MLP con 256 unidades y parada temprana y XGBoost con 500 estimadores, profundidad 3 y tasa 0,05. Usa semilla 42 y XGBoost en CPU en el notebook. La prueba agrupada histórica usó CUDA para XGBoost. La decisión es el máximo del promedio a partes iguales de las tres probabilidades de ambos modelos. La ejecución de entrega usa todo el train oficial.

El CSV se escribe con columnas `id,prediction` y se valida desde disco contra la plantilla. Exige 5.195 IDs únicos en el orden oficial y etiquetas 0, 1 o 2. El manifiesto registra versiones, hashes y parámetros. El CSV archivado de la ejecución corregida tiene el mismo SHA-256 que el consignado en la PR. La coincidencia histórica de 4.885 etiquetas (94,03 %) con la referencia local es una medida de reproducibilidad. La PR deja pendiente la puntuación de competición de ese archivo. El cuaderno de CatBoost traducido conserva una receta y una entrega anteriores. [Entrega MiniLM](../experiments/submissions/minilm_tokenizer_corrected) · [Entrega CatBoost](../experiments/submissions/catboost_translated) · [PR #14](https://github.com/Aitor42/0-AiDea/pull/14)

## Instalación, ejecución y localización

La [guía de reproducción](reproducibility.md) explica instalación, recursos, comandos, ejecución de los cuadernos y generación y validación de la entrega. El [mapa del repositorio](../README.md) enlaza el código, los experimentos y la documentación.
