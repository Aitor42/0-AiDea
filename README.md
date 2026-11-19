# Contradictory, My Dear Watson — Random Forest

Proyecto de la asignatura **Machine Learning** (2026/27, ESI-UCLM) para la
competición de Kaggle *Contradictory, My Dear Watson*: inferencia de lenguaje
natural (NLI) multilingüe sobre pares premisa–hipótesis.

Esta entrega entrena **únicamente un Random Forest** sobre una representación
TF-IDF `pair_diff` (unigramas, `min_df=5`) enriquecida con ratios POS y
features de relación, con selección chi² dentro del pipeline. El flujo completo está en
`random_forest_training.ipynb` y la implementación reutilizable en
`training.py`.

## Estructura

| Fichero | Contenido |
|---|---|
| `training.py` | Carga, split, preprocesado cacheado, pipeline RF, CV, holdout, métricas y submission |
| `random_forest_training.ipynb` | Entrenamiento y evaluación del RF con salidas ejecutadas |
| `pos_features.py` | Ratios de etiquetas POS |
| `relation_features.py` | Features de relación: negaciones, longitudes y números |
| `preprocesser.py` | Preprocesado de texto (`NLITextPreprocesser`) |
| `vectorizer.py` | Vectorizadores NLI (`NLITextVectorizer`) |
| `builder.py` | Wrapper de preprocesado + vectorización (PR 6) |
| `data/` | `train.csv`, `test.csv` y `sample_submission.csv` |
| `tests/` | Suite de tests del pipeline RF |
| `TRAINING.md` | Informe de entrenamiento y resultados |

Los notebooks `eda.ipynb`, `project.ipynb`, `builder_demo.ipynb`,
`preprocesser_demo.ipynb` y `vectorizer_demo.ipynb` documentan el análisis
exploratorio y los componentes de preprocesado/vectorización.

## Instalación

```bash
cd 0-AiDea
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

El preprocesado con spaCy requiere el modelo `en_core_web_sm` (ver
`requirements.txt`). `vectorizer.py` importa `sentence-transformers` al
cargarse, de modo que también es necesaria para los tests y el notebook; el
paquete arrastra `torch` (en CPU conviene instalarlo antes desde el índice de
PyTorch, ver `requirements.txt`).

## Uso

```bash
# Tests
.venv/bin/python -m pytest -q

# Notebook completo
.venv/bin/jupyter execute random_forest_training.ipynb --inplace
```

El notebook escribe en `results_random_forest/`:

- `final_config.json`
- `cv_results.csv`
- `holdout_metrics.json`
- `per_language.csv`
- `grouped_cv.json`
- `confusion_matrix.png`
- `submission.csv`

## Resultados

| Protocolo | Accuracy | Macro-F1 |
|---|---:|---:|
| CV 5-fold | 0,5133 | 0,5123 |
| Holdout (Kaggle) | **0,5206** | **0,5193** |
| CV agrupada por premisa | 0,5151 | 0,5140 |

La submission final contiene 5.195 predicciones con distribución
1.899 / 1.872 / 1.424. Los detalles y las limitaciones están en
`TRAINING.md`.
