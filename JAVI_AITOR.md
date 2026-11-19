# Experimentos de Javi y Aitor · Bloque 1

La [guía de revisión](FINAL/README.md) recoge los comandos y los puntos de conexión con el proyecto compartido. Los resultados de este informe proceden de las ejecuciones guardadas. Los experimentos nuevos usan el [cuaderno de trabajo](workflow_javi_aitor.ipynb), con una receta neutral y una validación reservada agrupada por premisa.

La pregunta experimental es si traducir los dos textos al inglés o cambiar su preprocesamiento ayuda a reconocer implicación, neutralidad y contradicción. La unidad de evaluación es el par original, identificado por `id`. El [enunciado, pp. 2 y 4–5](../Integrated_Midterm_NLI_2026_2027.pdf) pide comparar algoritmos en los mismos ejemplos, ajustar las transformaciones aprendidas solo en entrenamiento y justificar las decisiones en la defensa.

## Auditoría y reparto propuesto

Los issues pertenecen al repositorio del equipo. El directorio local incluye además resultados y herramientas de entrenamiento que ya estaban en `Aitor42/training`. El 18 de noviembre los cuatro issues seguían abiertos. La rama principal del equipo incluía el preprocesador y el wrapper, y los issues del bucle genérico y el evaluador seguían abiertos. Las funciones equivalentes disponibles en esta copia están en `training.py`.

| Issue | Responsable propuesto | Entrega y dependencia |
|---|---|---|
| [#7 · Permutador](https://github.com/Aitor42/0-AiDea/issues/7) | Javi | Generar variantes identificables con el preprocesador de Alba. Las variantes conservan el par y pueden alimentar distintas representaciones de Félix. |
| [#11 · Traducción](https://github.com/Aitor42/0-AiDea/issues/11) | Javi | Traducir y guardar una copia antes del wrapper. Conservar originales para auditar cambios de sentido y agrupar por la premisa de origen. |
| [#12 · Modelos](https://github.com/Aitor42/0-AiDea/issues/12) | Javi y Aitor | Javi se encarga de CatBoost. Propongo que Aitor mantenga el clasificador de referencia y compruebe que ambos reciben la misma representación. La fábrica local ya contiene los algoritmos. |
| [#13 · Entrenamiento](https://github.com/Aitor42/0-AiDea/issues/13) | Javi y Aitor | Javi ejecuta las variantes y documenta CatBoost. Propongo que Aitor revise los errores y reproduzca la comparación desde el cuaderno. La elección para Kaggle queda para el equipo. |

El orden de ejecución es fijar los IDs de entrenamiento y validación, preparar el original y su traducción, materializar el preprocesado y ajustar cada modelo en los mismos folds. La tokenización fila a fila se puede guardar. Aprender vocabulario, IDF o seleccionar palabras con chi-cuadrado requiere ajustar de nuevo dentro de cada fold. La evaluación por lengua debe utilizar la lengua de origen incluso cuando el modelo recibe inglés.

Alba conserva el EDA y el preprocesamiento. Félix conserva vectorizador, POS y wrapper. El balanceo corresponde a Dario y se mantiene la distribución observada en esta comparación. Para integrar futuras versiones de Akil, el punto de unión son los datasets preprocesados con IDs y los estimadores de la fábrica, junto con la partición guardada. [#8](https://github.com/Aitor42/0-AiDea/issues/8) · [#9](https://github.com/Aitor42/0-AiDea/issues/9) · [#10](https://github.com/Aitor42/0-AiDea/issues/10)

## Qué explica cada comparación

- **Representación.** En `pair_diff`, las palabras de la premisa y de la hipótesis ocupan bloques distintos. La diferencia absoluta y las medidas de solapamiento aportan señales sobre lo que cambia entre ambas. Los ratios POS resumen la composición gramatical. La etiqueta de clase sigue siendo nominal. El etiquetador inglés aplicado al texto multilingüe original es una limitación para interpretar los resultados por lengua.
- **CatBoost.** Se parte de puntuaciones para las tres clases y se añaden árboles sucesivamente para reducir la pérdida multiclase. La profundidad regula las interacciones que puede aprender cada árbol. La tasa de aprendizaje escala su contribución y el número de iteraciones fija cuántos árboles se añaden. Aquí recibe características numéricas calculadas por el pipeline. Una palabra compartida, una negación y un cambio de longitud pueden interactuar en un mismo árbol. [Algoritmo de CatBoost](https://catboost.ai/docs/en/concepts/algorithm-main-stages)
- **Traducción.** Un vocabulario común puede compartir evidencia entre idiomas y permite aplicar el etiquetador inglés. También puede cambiar una negación, un cuantificador, una entidad o una referencia de género. En NLI basta uno de esos cambios para alterar la relación entre las frases. Conviene leer los originales y sus traducciones en ejemplos donde cambie el acierto.
- **Métricas.** Accuracy es la fracción de pares correctamente clasificados. Macro-F1 promedia los F1 de las tres clases con igual peso. Para interpretar una diferencia entre ambos hay que mirar precisión, exhaustividad y soporte por clase. Una diferencia en un idioma con pocos ejemplos puede depender de muy pocos aciertos.

## Referencias experimentales anteriores

El train oficial tiene 12 120 pares y 15 lenguas. De ellos, 5250 pares requieren traducción al inglés. Al contar por lengua y texto exacto hay 10 225 frases no inglesas distintas entre premisas e hipótesis. El split histórico conserva 9696 pares para desarrollo y 2424 para validación. Comparte 1092 premisas distintas entre ambos lados, por lo que la métrica histórica incluye premisas vistas. La validación agrupada por la premisa original permite estudiar la generalización a premisas nuevas. [Partición guardada](artifacts/splits/split_summary.json)

| Ejecución histórica de CatBoost | Accuracy | Macro-F1 | Interpretación |
|---|---:|---:|---|
| TF-IDF `pair_diff`, POS y relación, CV de 5 folds | 0,5264 | 0,5250 | Selección en entrenamiento. 300 características, profundidad 6 y 300 árboles. |
| Misma configuración, validación histórica | 0,5206 | 0,5195 | 2424 pares. Detectó 380 de 813 contradicciones y confundió 237 con neutralidad. |
| MiniLM multilingüe, validación histórica | 0,6188 | 0,6183 | Profundidad 6 y 500 árboles. También cambian la representación y los hiperparámetros. |

Estos resultados preceden a las nuevas pruebas. El experimento MiniLM por sí solo no cubre el requisito de usar n-gramas y POS o NER como características predictivas. Ninguno de estos valores es una puntuación del leaderboard ni mide el efecto de traducir. [Configuración TF-IDF](results_algorithms/final_config.json) · [Matriz histórica](results_algorithms/holdout_metrics.json) · [Validación MiniLM](results_dense_confirmation/holdout.csv)

## Comparación de variantes de preprocesado

El [cuaderno ejecutado](comparacion_javi_aitor.ipynb) reproduce la comparación y conserva sus tablas por clase e idioma. Desde `0-AiDea`, ábrelo con el intérprete de `.venv`. La matriz de desarrollo cruza dos decisiones de preprocesamiento (lematizar o conservar los tokens), dos representaciones (bloques TF-IDF separados, con o sin diferencias y solapamiento) y dos clasificadores. Los tres folds agrupan por premisa original, con semilla 42. Se seleccionó la mayor accuracy media por algoritmo antes de evaluar el holdout.

| Finalista con lematización, POS y `pair_diff` | Accuracy CV | Macro-F1 CV | Accuracy validación | Macro-F1 validación |
|---|---:|---:|---:|---:|
| Naive Bayes multinomial | 0,4850 | 0,4828 | 0,4847 | 0,4828 |
| CatBoost | 0,5219 | 0,5208 | 0,5206 | 0,5195 |

Con CatBoost y lematización, añadir diferencias y solapamiento mejora la accuracy de desarrollo en 4,53 puntos porcentuales. Manteniendo `pair_diff`, lematizar aporta 0,37 puntos. Este último margen es pequeño y corresponde a esta partición. Naive Bayes suma la evidencia de las palabras bajo una hipótesis de independencia condicional que simplifica el entrenamiento. CatBoost aprovecha interacciones entre variables, aunque también comete errores de sentido. Por ejemplo, predijo implicación para «Per week?» / «Every day.», etiquetado como contradicción. Los dos textos expresan periodicidad, pero la frecuencia es distinta. [Ocho configuraciones, matrices, métricas y tiempos por fold](results_javi_aitor_cv/results.json)

## Comparación histórica con traducción

**Resultado del experimento completado.** Gemma 3 tradujo los 5250 pares no ingleses.
Sobre los mismos 2424 IDs de validación, CatBoost pasó de 1262 a 1371 aciertos
(accuracy de 0,5206 a 0,5656 y macro-F1 de 0,5195 a 0,5655). El modelo entrenado
con traducción acertó 313 ejemplos que antes fallaba y falló 204 que antes acertaba.
En los 1070 pares de origen no inglés hubo 77 aciertos netos adicionales
(483 frente a 560). En los 1354 ingleses hubo 32 (779 frente a 811):
sus textos siguieron iguales, pero cambió el corpus de entrenamiento.
Esta comparación mide el efecto sobre el clasificador en el holdout histórico.
La fidelidad lingüística de cada traducción requiere revisión por separado.
[Resultados por clase e idioma](results_ollama_catboost/metrics.json) · [Predicciones emparejadas](results_ollama_catboost/predictions.csv)

El 18 de noviembre se verificó el servidor local con `gemma3:4b` (Q4_K_M,
digest `68ceab7e468a31c988b44d5a4984a0e21f58c4f7158c6ce29c1804babac38df7`).
El piloto incluyó dos pares españoles con negación. En el ID `2a1d1ecdb0`,
«Vale, ¿puedes oírme?» y «Sé que no puedes oírme» se tradujeron como
«Okay, can you hear me?» y «I know you cannot hear me.». Este control solo
comprueba esos ejemplos. Se usa JSON estructurado para vincular los textos
de cada lote a su ID. [API de Ollama](https://docs.ollama.com/capabilities/structured-outputs)

La ejecución histórica sobre el corpus usó una versión anterior de [compare_ollama.py](compare_ollama.py), con lematización, POS y relación ingleses en ambos brazos. El comando actual usa la receta neutral en ambos brazos y publica en un directorio nuevo.
Primero guarda todos los pares en `artifacts/translations/train_gemma3_4b.csv`.
Después entrena el CatBoost ya seleccionado con texto original y con texto
traducido, manteniendo los mismos IDs, semilla, características e hiperparámetros.
Al terminar publica `results_ollama_catboost/metrics.json` y `predictions.csv`.
Para regenerar la traducción desde el [cuaderno de trabajo](workflow_javi_aitor.ipynb), selecciona `MODE = "full"` y activa `GENERATE_FULL_TRANSLATIONS`. La celda usa el CSV guardado si está disponible.

El traductor escribe el CSV al terminar todos los lotes. Si se interrumpe, vuelve a ejecutar la traducción desde el principio. El script de comparación reutiliza el CSV ya generado para el modelo tras comprobar que conserva las filas originales.

Para reproducir los resultados agrupados desde una copia limpia, se recupera cada partición por los IDs guardados en el orden original. En esta copia el directorio de resultados ya existe. Usa otro nombre de salida si quieres conservar la ejecución anterior.

```python
from pathlib import Path
import pandas as pd
from parameter_variants import materialize_variants
from run_experiment_matrix import DEFAULT_REPRESENTATIONS, run_matrix

options = {"dtype": {"id": str}, "keep_default_na": False}
english = pd.read_csv("artifacts/translations/train_gemma3_4b.csv", **options).set_index("id")
baseline = [{"name": "baseline", "params": {}}]
datasets = {}
for split in ("train", "val"):
    original = pd.read_csv(f"artifacts/splits/{split}.csv", **options)
    translated = english.loc[original.id].reset_index()
    pd.testing.assert_frame_equal(original, translated[original.columns])
    raw, = materialize_variants(original, variants=baseline)
    eng, = materialize_variants(translated, variants=baseline, base_params={
        "columns": ["premise_translated", "hypothesis_translated"]
    })
    datasets[split] = [("original", Path(raw["dataset_path"])),
                       ("translated", Path(eng["dataset_path"]))]
run_matrix(datasets["train"], {"pair_diff": DEFAULT_REPRESENTATIONS["pair_diff"]},
           ["naive_bayes", "catboost"], Path("results_translation_cv"),
           n_splits=3, seed=42, group_by_premise=True,
           use_pos_features=True, use_relation_features=True,
           holdout_datasets=datasets["val"], holdout_classifiers=["naive_bayes"])
```

La evaluación final empareja del mismo modo los IDs de `artifacts/splits/val.csv`. Conviene examinar por separado las filas originalmente inglesas y las traducidas, junto con los cambios en negaciones y cantidades. El holdout histórico ya se había consultado en el proyecto y se utiliza como referencia compartida.

## Comparación agrupada con y sin traducción

Se amplió el experimento a CatBoost y Naive Bayes con idéntico preprocesamiento, representación `pair_diff`, selección chi-cuadrado de 300 características, semilla 42 y tres folds agrupados por la premisa original. Cada par conserva su ID, etiqueta, idioma y premisa originales en ambas condiciones. La traducción cambia las columnas que consume el preprocesador, y el agrupamiento usa la premisa de origen.

| Modelo | Accuracy CV original | Accuracy CV traducido | Diferencia | Macro-F1 CV original | Macro-F1 CV traducido | Accuracy holdout original | Accuracy holdout traducido |
|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost | 0,5219 | 0,5482 | +2,63 pp | 0,5208 | 0,5482 | 0,5206 | 0,5656 |
| Naive Bayes | 0,4850 | 0,5026 | +1,75 pp | 0,4828 | 0,4989 | 0,4847 | 0,4930 |

CatBoost mejora en cada uno de los tres folds agrupados. Sus accuracies pasan de 0,5198, 0,5192 y 0,5266 a 0,5424, 0,5523 y 0,5498. Naive Bayes también sube en los tres folds. En el holdout, el macro-F1 de CatBoost cambia de 0,5195 a 0,5655 y el de Naive Bayes de 0,4828 a 0,4900.

La revisión de folds confirma que los dos brazos contienen los mismos 9696 IDs de desarrollo y 2424 IDs de holdout, en el mismo orden, y que cada par mantiene etiqueta y lengua original. Los tres folds contienen 6464 ejemplos de ajuste y 3232 de evaluación. No comparten ninguna premisa entre ambos lados de cada fold. El split de holdout se mantiene como el artefacto histórico y tiene solapamiento de premisas con desarrollo. El cuaderno reproduce la auditoría agrupada y las comparaciones desde los artefactos guardados. [Resultados agrupados](results_translation_cv/results.json) · [Métricas de CatBoost en holdout](results_ollama_catboost/metrics.json)

Los pares `3931fbe82a` (francés) y `b962358e67` (vietnamita) se incluyeron como ejemplos de efectos opuestos. En el primero la traducción coincide con la etiqueta real y cambia la predicción de contradiction a entailment. En el segundo, el texto traducido mantiene la afirmación de que el hablante está contento usando la expresión citada, pero CatBoost cambia una predicción correcta de entailment a contradiction. Son ejemplos inspeccionables, no explicación causal del modelo. [Predicciones por ID](results_ollama_catboost/predictions.csv) · [Cuaderno ejecutado](comparacion_javi_aitor.ipynb)
