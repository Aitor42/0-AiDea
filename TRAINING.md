# Random Forest — informe de entrenamiento

**Proyecto:** *Contradictory, My Dear Watson* (Kaggle) — Machine Learning
2026/27, ESI-UCLM.

Este documento resume el entrenamiento y la evaluación del **Random Forest**,
el único modelo que se entrena en esta entrega. La implementación está en
`training.py` y el flujo completo, con salidas ejecutadas, en
`random_forest_training.ipynb`.

## 1. Datos y partición

- `data/train.csv`: 12.120 pares premisa–hipótesis en 15 idiomas.
- Etiquetas: `entailment` (4.176), `neutral` (3.880) y `contradiction` (4.064);
  las clases están casi equilibradas y no se remuestrean.
- Partición estratificada 80/20 con `seed=42`: 9.696 pares de entrenamiento y
  2.424 de validación reservada.
- El holdout no participa en ninguna decisión de diseño.
- El split se verifica contra `artifacts/splits/train.csv` y `val.csv`.
- Hay 1.092 premisas compartidas entre train y validación (48,5% de la
  partición pequeña), una propiedad conocida del dataset.

## 2. Preprocesado y representación

- **Preprocesado:** tokenización y lematización de spaCy
  (`tokenizer="spacy"`, `reduction="spacy_lemma"`, `en_core_web_sm` con parser
  y NER desactivados). Minúsculas, sin eliminación de stopwords (se conservan
  las negaciones) y `use_pos=True`, de modo que cada token es `(palabra, POS)`.
- **Representación:** `pair_diff`, es decir, TF-IDF de unigramas de premisa,
  hipótesis y su diferencia absoluta, más cuatro features de solape léxico
  (Jaccard, cobertura direccional y similitud coseno), con `min_df=5`. El
  vectorizador usa `use_pos=False`: el sufijo `_POS` fragmenta el vocabulario y
  cuesta 0,65 puntos de accuracy en validación.
- **Features POS:** 24 ratios de `POSRatioFeatures` (8 etiquetas Universal ×
  premisa, hipótesis y diferencia absoluta). El POS tagging se sigue aplicando
  aunque no contamine los tokens del TF-IDF.
- **Features de relación:** 14 columnas de `RelationFeatures`, calculadas fila a
  fila: negaciones (recuento en premisa e hipótesis, diferencia absoluta y XOR),
  longitudes (cada frase, diferencia, ratio y qué frase es más larga) y números
  (recuentos, diferencia y Jaccard de números compartidos). Todas son no
  negativas, como exige chi². El transformador admite también una sola columna
  (solo premisas) y emite entonces las 3 señales propias de la frase; con
  `pair_diff` el pipeline necesita premisa e hipótesis porque calcula `|Xp - Xh|`.
- **Selección:** `SelectKBest(chi2, k=300)` como paso del pipeline. Con
  `min_df=5` el pipeline genera **19.107 features**; chi² conserva 300: 287 de
  texto, 1 de ratios POS y 12 de relación.
- El preprocesado es una transformación fila a fila y se cachea por hash de
  configuración y datos; no aprende del corpus y no introduce fugas.

## 3. Modelo

`RandomForestClassifier` con:

| Hiperparámetro | Valor |
|---|---|
| `n_estimators` | 1.000 |
| `max_depth` | 50 |
| `min_samples_leaf` | 5 |
| `min_samples_split` | 5 |
| `max_features` | `"sqrt"` |
| `criterion` | `"gini"` |
| `n_jobs` | -1 |

La poda a `max_depth=50` con hojas de al menos 5 ejemplos reduce el tiempo de
ajuste y mantiene la accuracy de validación cruzada. Una rejilla sobre
`n_estimators`, `min_samples_leaf`, `max_depth`, `max_features` y `criterion`
no mejoró esos valores: el bosque ya estaba en su meseta.

El pipeline es `features -> select -> classifier`. La selección chi² aportó
+1,3 puntos de accuracy y redujo el ajuste de 5,1 s a 2,6 s por fold (7.756
pares). Al ser un paso del pipeline, se reajusta dentro de cada fold con el
bloque de entrenamiento y nunca ve el fold de test.

## 4. Protocolo de evaluación

- Validación cruzada estratificada de 5 folds con `seed=42`.
- En cada fold se clona el pipeline y se ajusta solo con el bloque de
  entrenamiento: vocabulario, IDF, features auxiliares y selección chi² nunca
  ven el fold de test.
- Holdout final: ajuste con los 9.696 pares de train y evaluación sobre los
  2.424 pares reservados.
- Comprobación adicional con `StratifiedGroupKFold` por premisa: ninguna
  premisa se comparte entre train y test dentro de un fold.

## 5. Resultados

### Validación cruzada

| Métrica | Media ± desviación |
|---|---|
| Accuracy | **0,5133 ± 0,0095** |
| Macro-F1 | **0,5123 ± 0,0100** |

### Holdout

| Métrica | Valor |
|---|---|
| Accuracy (métrica Kaggle) | **0,520627** |
| Macro-F1 | **0,519292** |

| Clase | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| entailment | 0,5423 | 0,5760 | 0,5587 | 835 |
| neutral | 0,4932 | 0,5155 | 0,5041 | 776 |
| contradiction | 0,5248 | 0,4686 | 0,4951 | 813 |

La configuración anterior (tokenización por espacios, bigramas, `min_df=50`,
sin selección ni features de relación) obtenía 0,4525 en CV y 0,4814 en el
holdout: la representación actual suma **+6,1 puntos en CV y +3,9 en el
holdout**. La descomposición de la mejora, con las ablaciones y las
comparaciones pareadas por fold, está en el informe de investigación.

### Validación agrupada por premisa

| Protocolo | Accuracy | Macro-F1 |
|---|---:|---:|
| Estándar | 0,513303 | 0,512295 |
| Agrupada | 0,515057 | 0,514006 |

La CV agrupada no degrada la estimación, así que las premisas repetidas no
inflan la accuracy reportada.

### Errores y lenguas

- 1.162 errores de 2.424 pares (47,9%).
- Direcciones más frecuentes: `contradiction → neutral` (230),
  `neutral → entailment` (204), `contradiction → entailment` (202),
  `entailment → neutral` (181), `entailment → contradiction` (173) y
  `neutral → contradiction` (172). Los seis sentidos siguen poblados: ninguna
  clase colapsa.
- Accuracy por lengua: inglés 0,5790 (1.354 pares), vietnamita 0,5672, búlgaro
  0,5493, español 0,5301, hindi 0,5000; alemán 0,3433, chino 0,3398,
  tailandés 0,3506 y turco 0,3538.
- La mejora se concentra en inglés (+6,6 puntos); en el resto de lenguas hay
  60–100 pares por idioma (error estándar ≈5–6 puntos), así que las diferencias
  individuales no son concluyentes.

## 6. Submission

- El pipeline se reajusta con los 12.120 pares etiquetados.
- Predice los 5.195 pares de `data/test.csv`.
- Salida: `results_random_forest/submission.csv` con esquema `id,prediction`.
- Distribución: 1.899 `entailment`, 1.872 `neutral` y 1.424 `contradiction`.
- Coincide al 65,29% con la submission de la configuración anterior; el cambio
  es esperado porque cambian el vocabulario y las features que ve el bosque.

## 7. Limitaciones y siguientes pasos

- Herramientas NLP de inglés (tokenizador, lematizador y POS) aplicadas a 15
  idiomas: es la limitación principal y la razón de que la mejora se concentre
  en inglés.
- La configuración se eligió con validación cruzada sobre el mismo split de
  entrenamiento (~45 variantes probadas); el holdout independiente confirma
  +3,9 puntos, pero el máximo puede estar ligeramente inflado.
- Parte de la señal de las features de relación son longitudes, un patrón del
  dataset más que comprensión semántica; conviene declararlo.
- Representación de bolsa de palabras: no modela composicionalidad ni el orden
  de las palabras.
- Próximos pasos: traducción previa al inglés, modelos contextuales y la
  comparación con un segundo algoritmo supervisado que pide la rúbrica (fuera
  del alcance de esta rama, centrada solo en Random Forest).

## 8. Reproducibilidad

```bash
cd 0-AiDea
.venv/bin/python -m pytest -q
.venv/bin/jupyter execute random_forest_training.ipynb --inplace
```

La ejecución completa del notebook tarda unos 20 minutos, dominados por el POS
tagging fila a fila del preprocesado (el ajuste del bosque son ~2,6 s por
fold). Las particiones ya preprocesadas se reutilizan desde
`artifacts/preprocessed/` por hash de configuración y datos.

Artefactos generados en `results_random_forest/`:

- `final_config.json`
- `cv_results.csv`
- `holdout_metrics.json`
- `per_language.csv`
- `grouped_cv.json`
- `confusion_matrix.png`
- `submission.csv`

Entorno verificado: Python 3.12.14, scikit-learn 1.9.1, spaCy 3.8.18 y
NLTK 3.10.4.
