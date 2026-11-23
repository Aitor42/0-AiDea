# 0-AiDea · Inferencia textual multilingüe

Proyecto de [Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson). A partir de una premisa y una hipótesis, el sistema predice implicación, neutralidad o contradicción. El trabajo compara representaciones léxicas y densas, mide el efecto de traducir ambos textos al inglés y prepara una entrega reproducible para Kaggle.

El voto MiniLM de MLP y XGBoost obtuvo **62,14 % de accuracy y 61,84 % de macro-F1** en validación local agrupada por premisa. En una comparación léxica emparejada, la accuracy agrupada de CatBoost pasó del **52,19 % al 54,82 %** tras traducir con Gemma 3. Cada cifra corresponde a su propio protocolo. [Resultados de MiniLM](experiments/results/validation/minilm_grouped/grouped_summary.csv) · [Comparación con traducción](experiments/results/translation/grouped_cv/results.json)

- [src](src) contiene el paquete Python y las pruebas. Usa pandas, scikit-learn, spaCy, NLTK, Sentence Transformers y las librerías de boosting.
- [experiments](experiments) contiene los cuadernos, ejecutores, configuraciones y resultados. La traducción opcional usa Ollama.
- [docs](docs) contiene la guía, los informes históricos y el inventario de la reorganización.

## Primer uso

Ejecuta estos comandos desde la raíz del proyecto con el entorno virtual activo.

```sh
python -m pip install -e .
python -m pip install jupyterlab
python -m spacy download en_core_web_sm
```

Coloca los tres CSV de la competición en `experiments/data/` y abre JupyterLab en la raíz con el kernel del entorno instalado. El cuaderno de trabajo arranca en modo reducido. [Preparación del entorno y reproducción](docs/reproducibility.md)

- [Guía completa del proyecto](docs/PROJECT_GUIDE.md)
- [Flujo de trabajo de Javi y Aitor](experiments/notebooks/workflow_javi_aitor.ipynb)
- [Investigación de Random Forest](experiments/notebooks/random_forest_training.ipynb)
- [Entrega MiniLM para Kaggle](experiments/notebooks/kaggle_minilm.ipynb)
- [Informe de experimentos del equipo](docs/reports/team_experiments.md) · [Estudio de Random Forest](docs/reports/random_forest_research.md)
