# Experimentos de preprocesado y traducción

[Mapa del repositorio](../../README.md) · [Guía del proyecto](../PROJECT_GUIDE.md) · [Guía de reproducción](../reproducibility.md)

Comparamos cómo el preprocesado y la traducción al inglés afectan a la clasificación de pares multilingües. El informe recoge ejecuciones archivadas de la PR #14 y un contraste con MiniLM. Cada comparación mantiene los pares identificados por `id` y la lengua de origen para evaluar los cambios por idioma.

## Índice

- [Diseño de las comparaciones](#diseño-de-las-comparaciones)
- [Referencias experimentales anteriores](#referencias-experimentales-anteriores)
- [Comparación de variantes de preprocesado](#comparación-de-variantes-de-preprocesado)
- [Comparación histórica con traducción](#comparación-histórica-con-traducción)
- [Comparación agrupada con y sin traducción](#comparación-agrupada-con-y-sin-traducción)
- [Traducción con MiniLM](#traducción-con-minilm)
- [Reproducción de la comparación agrupada](#reproducción-de-la-comparación-agrupada)

## Diseño de las comparaciones

- **Particiones**. Fijamos los IDs antes de preparar las variantes y comparamos ambos brazos sobre los mismos folds. Conservamos la distribución de clases observada.
- **Representación**. Comparamos bloques TF-IDF separados con `pair_diff`, que añade diferencia absoluta y solapamiento. El POS inglés aplicado a textos de otros idiomas limita su interpretación. [Representaciones y clasificadores](../PROJECT_GUIDE.md#representaciones-y-clasificadores)
- **Ajuste**. Podemos guardar el preprocesado fila a fila. Ajustamos vocabulario, IDF, chi-cuadrado y modelo dentro de cada fold, tal como pide el enunciado (pp. 2 y 4–5). [Variantes de preprocesado](parameter_variants.md) · [Validación y métricas](../PROJECT_GUIDE.md#entrenamiento-validación-y-métricas)

## Referencias experimentales anteriores

De los 12.120 pares del train oficial, 5.250 requieren traducción al inglés. Hay 10.225 frases no inglesas distintas al contar por lengua y texto exacto entre premisas e hipótesis. El holdout histórico incluye premisas vistas en desarrollo. [Datos y partición histórica](../PROJECT_GUIDE.md#problema-y-datos)

| Ejecución histórica de CatBoost | Accuracy | Macro-F1 | Interpretación |
|---|---:|---:|---|
| TF-IDF `pair_diff`, POS y relación, CV de 5 folds | 0,5264 | 0,5250 | Selección en entrenamiento. 300 características, profundidad 6 y 300 árboles. |
| Misma configuración, validación histórica | 0,5206 | 0,5195 | 2.424 pares. Detectó 380 de 813 contradicciones y confundió 237 con neutralidad. |
| MiniLM multilingüe, validación histórica | 0,6188 | 0,6183 | Profundidad 6 y 500 árboles. También cambian la representación y los hiperparámetros. |

Las cifras son referencias locales previas a las comparaciones de traducción. Los experimentos léxicos cubren el requisito académico de usar n-gramas y POS o NER como características predictivas. [Configuración TF-IDF](../../experiments/results/classical_models/algorithm_comparison/final_config.json) · [Matriz histórica](../../experiments/results/classical_models/algorithm_comparison/holdout_metrics.json) · [Validación MiniLM](../../experiments/results/minilm/confirmation/holdout.csv)

## Comparación de variantes de preprocesado

El [cuaderno ejecutado](../../experiments/notebooks/archive/comparacion_javi_aitor.ipynb) reproduce la comparación y conserva sus tablas por clase e idioma. Ábrelo desde la raíz con el [kernel del entorno local](../reproducibility.md#preparación-de-los-datos). La matriz cruza lematización frente a tokens conservados, TF-IDF con bloques separados frente a `pair_diff`, y Naive Bayes multinomial frente a CatBoost. Los tres folds agrupan por premisa original, con semilla 42. Seleccionamos la configuración con mayor accuracy media por algoritmo antes de evaluar el holdout.

| Finalista con lematización, POS y `pair_diff` | Accuracy CV | Macro-F1 CV | Accuracy validación | Macro-F1 validación |
|---|---:|---:|---:|---:|
| Naive Bayes multinomial | 0,4850 | 0,4828 | 0,4847 | 0,4828 |
| CatBoost | 0,5219 | 0,5208 | 0,5206 | 0,5195 |

Con CatBoost y lematización, añadir diferencias y solapamiento mejora la accuracy de desarrollo en 4,53 puntos porcentuales. Manteniendo `pair_diff`, lematizar aporta 0,37 puntos. Este último margen es pequeño y corresponde a esta partición. En un error de sentido, CatBoost predijo implicación para «Per week?» / «Every day.», etiquetado como contradicción. Los dos textos expresan periodicidad, pero la frecuencia es distinta. [Ocho configuraciones, matrices, métricas y tiempos por fold](../../experiments/results/validation/javi_aitor/results.json)

### Espacios frente a spaCy

El cuaderno de trabajo archivó otra comparación sobre 9.694 filas de desarrollo, con cinco folds agrupados y semilla 42. Ambas variantes usan texto original, unigramas con `min_df=2`, `pair_diff` y chi-cuadrado con 300 rasgos. Se desactivaron la lematización, el POS y los rasgos de relación para comparar los tokenizadores. [Configuraciones, tiempos y métricas por fold](../../experiments/results/validation/workflow_javi_aitor/tokenizers/full/f160a561fdbc141a/results.json)

| Modelo | Accuracy espacios | Accuracy spaCy | Macro-F1 espacios | Macro-F1 spaCy |
| --- | ---: | ---: | ---: | ---: |
| Naive Bayes multinomial | 0,4403 | 0,4425 | 0,4080 | 0,4110 |
| CatBoost | 0,4958 | 0,4994 | 0,4935 | 0,4961 |

## Comparación histórica con traducción

CatBoost pasó de 1.262 a 1.371 aciertos sobre los mismos 2.424 IDs. La accuracy subió de 0,5206 a 0,5656 y el macro-F1 de 0,5195 a 0,5655. La traducción recuperó 313 aciertos y perdió 204.

| Lengua de origen | Pares | Aciertos con original | Aciertos con traducción | Cambio neto |
| --- | ---: | ---: | ---: | ---: |
| Inglés | 1.354 | 779 | 811 | +32 |
| Resto | 1.070 | 483 | 560 | +77 |
| Total | 2.424 | 1.262 | 1.371 | +109 |

Los textos en inglés se mantuvieron. Su mejora coincide con el cambio del corpus de entrenamiento. Las métricas miden el efecto en el clasificador. La fidelidad de cada traducción requiere revisar los textos. [Métricas por clase e idioma](../../experiments/results/translation/catboost_holdout/metrics.json) · [Predicciones emparejadas](../../experiments/results/translation/catboost_holdout/predictions.csv)

<details>
<summary>CatBoost por lengua de origen en el holdout</summary>

| Lengua | Pares | Accuracy original | Accuracy traducida | Macro-F1 original | Macro-F1 traducido |
| --- | ---: | ---: | ---: | ---: | ---: |
| Inglés | 1.354 | 0,5753 | 0,5990 | 0,5715 | 0,5975 |
| Chino | 103 | 0,3301 | 0,4563 | 0,2769 | 0,4507 |
| Ruso | 94 | 0,4787 | 0,4362 | 0,4760 | 0,4349 |
| Hindi | 86 | 0,4535 | 0,5233 | 0,4472 | 0,5207 |
| Español | 83 | 0,5542 | 0,4940 | 0,5488 | 0,4614 |
| Árabe | 79 | 0,3418 | 0,5316 | 0,3348 | 0,5257 |
| Griego | 77 | 0,4675 | 0,6234 | 0,4614 | 0,6135 |
| Tailandés | 77 | 0,3766 | 0,5844 | 0,3704 | 0,5685 |
| Suajili | 72 | 0,4167 | 0,5000 | 0,4173 | 0,4911 |
| Búlgaro | 71 | 0,5634 | 0,6901 | 0,5630 | 0,6907 |
| Urdu | 70 | 0,5571 | 0,4571 | 0,5493 | 0,4471 |
| Alemán | 67 | 0,3731 | 0,4776 | 0,3505 | 0,4666 |
| Vietnamita | 67 | 0,5970 | 0,5821 | 0,5857 | 0,5779 |
| Turco | 65 | 0,3846 | 0,5077 | 0,3799 | 0,5040 |
| Francés | 59 | 0,4746 | 0,5085 | 0,4686 | 0,5093 |

</details>

El piloto con Gemma 3 incluyó dos pares españoles con negación. En `2a1d1ecdb0`, «Vale, ¿puedes oírme?» y «Sé que no puedes oírme» se tradujeron como «Okay, can you hear me?» y «I know you cannot hear me.». El corpus completo cubrió los 5.250 pares no ingleses. [Modelo, manifiestos y regeneración](../reproducibility.md#traducción) · [JSON estructurado de Ollama](https://docs.ollama.com/capabilities/structured-outputs)

Esta ejecución usó una versión anterior de [compare_ollama.py](../../experiments/scripts/compare_ollama.py), con lematización, POS y rasgos de relación en ambos brazos. El ejecutor actual aplica la receta neutral sin POS ni relación, valida el CSV reutilizado y guarda `metrics.json` y `predictions.csv` en una carpeta nueva de `experiments/.local/ollama_comparisons/` por defecto.

## Comparación agrupada con y sin traducción

Se amplió el experimento a CatBoost y Naive Bayes con idéntico preprocesamiento, representación `pair_diff`, selección chi-cuadrado de 300 características, semilla 42 y tres folds agrupados por la premisa original. Cada par conserva su ID, etiqueta, idioma y premisa originales en ambas condiciones. La traducción cambia las columnas que consume el preprocesador, y el agrupamiento usa la premisa de origen.

| Modelo | Accuracy CV original | Accuracy CV traducido | Diferencia | Macro-F1 CV original | Macro-F1 CV traducido | Accuracy holdout original | Accuracy holdout traducido |
|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost | 0,5219 | 0,5482 | +2,63 pp | 0,5208 | 0,5482 | 0,5206 | 0,5656 |
| Naive Bayes | 0,4850 | 0,5026 | +1,75 pp | 0,4828 | 0,4989 | 0,4847 | 0,4930 |

CatBoost mejora en cada uno de los tres folds agrupados. Sus accuracies pasan de 0,5198, 0,5192 y 0,5266 a 0,5424, 0,5523 y 0,5498. Naive Bayes también sube en los tres folds. En el holdout, el macro-F1 de CatBoost cambia de 0,5195 a 0,5655 y el de Naive Bayes de 0,4828 a 0,4900.

El holdout ya se había consultado previamente y sirve como referencia común. La revisión de folds confirma que los dos brazos contienen los mismos 9.696 IDs de desarrollo y 2.424 IDs de holdout, en el mismo orden, y que cada par mantiene etiqueta y lengua original. Cada fold contiene 6.464 ejemplos de ajuste y 3.232 de evaluación, con cero premisas compartidas. El split de holdout se mantiene como el artefacto histórico y tiene solapamiento de premisas con desarrollo. El cuaderno reproduce la auditoría agrupada y las comparaciones desde los artefactos guardados. [Resultados agrupados](../../experiments/results/translation/grouped_cv/results.json) · [Métricas de CatBoost en holdout](../../experiments/results/translation/catboost_holdout/metrics.json)

Los pares `3931fbe82a` (francés) y `b962358e67` (vietnamita) se incluyeron como ejemplos de efectos opuestos. En el primero, con el texto traducido el modelo cambia su predicción de `contradiction` a `entailment` y acierta. En el segundo, el texto traducido mantiene la afirmación de que el hablante está contento usando la expresión citada, pero CatBoost cambia una predicción correcta de `entailment` a `contradiction`. [Predicciones por ID](../../experiments/results/translation/catboost_holdout/predictions.csv) · [Cuaderno ejecutado](../../experiments/notebooks/archive/comparacion_javi_aitor.ipynb)

## Traducción con MiniLM

El voto de MLP y XGBoost se probó con cinco folds agrupados por la premisa original y semilla 42. La accuracy global con traducción bajó 0,33 puntos porcentuales. La caída se concentró en los pares de origen no inglés. [Configuración y métricas](../../experiments/results/translation/minilm_seed42/results.json) · [Predicciones](../../experiments/results/translation/minilm_seed42/predictions.csv)

| Entrada | Accuracy | Macro-F1 | Accuracy en inglés | Accuracy en el resto |
| --- | ---: | ---: | ---: | ---: |
| Original | 0,6206 | 0,6169 | 0,6331 | 0,6041 |
| Traducida | 0,6173 | 0,6145 | 0,6401 | 0,5871 |

## Reproducción de la comparación agrupada

Ejecuta la celda desde la raíz, con las [dependencias instaladas](../reproducibility.md#instalación-y-dependencias). Usa los CSV archivados para recuperar cada partición en el orden original. Elige una salida nueva si ya existe una ejecución.

```python
from pathlib import Path
import pandas as pd
from nli_project.preprocessing.parameter_variants import materialize_variants
from experiments.scripts.run_experiment_matrix import DEFAULT_REPRESENTATIONS, run_matrix

options = {"dtype": {"id": str}, "keep_default_na": False}
english = pd.read_csv("experiments/results/translation/corpus/train_gemma3_4b.csv", **options).set_index("id")
baseline = [{"name": "baseline", "params": {}}]
datasets = {}
for split in ("train", "val"):
    original = pd.read_csv(f"experiments/results/validation/historical_split/{split}.csv", **options)
    translated = english.loc[original.id].reset_index()
    pd.testing.assert_frame_equal(original, translated[original.columns])
    raw, = materialize_variants(original, variants=baseline)
    eng, = materialize_variants(translated, variants=baseline, base_params={
        "columns": ["premise_translated", "hypothesis_translated"]
    })
    datasets[split] = [("original", Path(raw["dataset_path"])),
                       ("translated", Path(eng["dataset_path"]))]
run_matrix(datasets["train"], {"pair_diff": DEFAULT_REPRESENTATIONS["pair_diff"]},
           ["naive_bayes", "catboost"], Path("experiments/.local/translation/grouped_cv"),
           n_splits=3, seed=42, group_by_premise=True,
           use_pos_features=True, use_relation_features=True,
           holdout_datasets=datasets["val"], holdout_classifiers=["naive_bayes"])
```

La celda repite la CV de ambos modelos y el holdout de Naive Bayes. El holdout de CatBoost procede de la [comparación histórica](#comparación-histórica-con-traducción). El [cuaderno de comparación histórica](../../experiments/notebooks/archive/comparacion_javi_aitor.ipynb) permite consultar las tablas por clase e idioma. Para nuevas comparaciones con la receta neutral, usa el [cuaderno de trabajo](../../experiments/notebooks/workflow_javi_aitor.ipynb).
