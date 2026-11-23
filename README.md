# 0-AiDea · Inferencia textual multilingüe

Proyecto de [Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson). A partir de una premisa y una hipótesis, el sistema predice implicación, neutralidad o contradicción. El trabajo compara representaciones léxicas y densas, mide el efecto de traducir ambos textos al inglés y prepara una entrega reproducible para Kaggle.

El voto MiniLM de MLP y XGBoost obtuvo **62,14 % de accuracy y 61,84 % de macro-F1** en validación local agrupada por premisa. En una comparación léxica emparejada, la accuracy agrupada de CatBoost pasó del **52,19 % al 54,82 %** tras traducir con Gemma 3. Cada cifra corresponde a su propio protocolo. [Resultados de MiniLM](experiments/results/validation/minilm_grouped/grouped_summary.csv) · [Comparación con traducción](experiments/results/translation/grouped_cv/results.json)

## Índice

[Mapa del repositorio](#mapa-del-repositorio) · [Primer uso](#primer-uso) · [Cuadernos y resultados](#cuadernos-y-resultados)

## Mapa del repositorio

| Ruta | Qué encontrarás |
| --- | --- |
| [`src/`](src) | [Paquete Python](src/nli_project) para cargar datos, preprocesar, representar los pares, entrenar, evaluar y generar entregas. Incluye las [pruebas](src/tests). |
| [`experiments/`](experiments) | [Cuadernos](experiments/notebooks), [ejecutores](experiments/scripts) y [configuraciones](experiments/configs). Contiene los [resultados archivados](experiments/results) y los [CSV de entrega](experiments/submissions). |
| [`docs/`](docs) | [Guía del proyecto](docs/PROJECT_GUIDE.md), [instalación y reproducción](docs/reproducibility.md) e [informes experimentales](docs/reports). |

## Primer uso

Desde la raíz del proyecto, los pasos son activar el entorno virtual, instalar el paquete y los recursos lingüísticos, y colocar los tres CSV oficiales en `experiments/data/`. Los comandos y dependencias están en la [guía de instalación](docs/reproducibility.md#instalación-y-dependencias).

Abre el [cuaderno de trabajo](experiments/notebooks/workflow_javi_aitor.ipynb) en JupyterLab con el kernel de ese entorno. Arranca en modo reducido (`smoke`).

## Cuadernos y resultados

| Para consultar | Cuaderno | Informe o resultados |
| --- | --- | --- |
| Preprocesamiento y traducción | [Cuaderno de trabajo](experiments/notebooks/workflow_javi_aitor.ipynb) · [Comparación histórica](experiments/notebooks/archive/comparacion_javi_aitor.ipynb) | [CatBoost, Naive Bayes y traducción](docs/reports/team_experiments.md) · [Variantes de preprocesado](docs/reports/parameter_variants.md) |
| Random Forest | [Cuaderno de entrenamiento](experiments/notebooks/random_forest_training.ipynb) | [Estudio de Random Forest](docs/reports/random_forest_research.md) · [Resultados archivados](experiments/results/random_forest) |
| MiniLM y entrega | [Cuaderno de Kaggle](experiments/notebooks/kaggle_minilm.ipynb) | [Validación agrupada](experiments/results/validation/minilm_grouped) · [Entrega archivada](experiments/submissions/minilm_tokenizer_corrected) |
