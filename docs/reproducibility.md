# Entorno y reproducción

[Mapa del repositorio](../README.md) · [Guía del proyecto](PROJECT_GUIDE.md)

## Índice

- [Instalación y dependencias](#instalación-y-dependencias)
- [Preparación de los datos](#preparación-de-los-datos)
- [Ejecución de experimentos](#ejecución-de-experimentos)
- [Reproducción de resultados](#reproducción-de-resultados)
- [Traducción](#traducción)
- [Generación de entregas](#generación-de-entregas)

## Instalación y dependencias

El paquete requiere Python 3.12 o posterior. La reorganización se verificó con Python 3.14.7. Crea y activa un entorno virtual. La instalación editable hace importables `nli_project` y `experiments.scripts`. Usa el mismo Python para Jupyter y pytest. [Requisitos del paquete](../pyproject.toml) · [Dependencias Python](../requirements.txt)

```sh
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m pip install jupyterlab seaborn
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

Instala los recursos NLTK para las variantes que usan su tokenizador, stopwords o WordNet. En CPU puedes usar el [índice de PyTorch](https://download.pytorch.org/whl/cpu/) con `--extra-index-url https://download.pytorch.org/whl/cpu`.

## Preparación de los datos

1. Crea `experiments/data/` y guarda ahí `train.csv`, `test.csv` y `sample_submission.csv`. Los tests de particiones usan el train oficial y el cuaderno de MiniLM comprueba los hashes de los CSV. [Datos de Kaggle](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data) · [Hashes de referencia](../src/nli_project/submission/minilm_submission.py)
2. Ejecuta `python -m pytest` para comprobar el entorno y abre `python -m jupyter lab` desde la raíz. Selecciona el kernel del entorno instalado.

Los resultados nuevos y las cachés se guardan por defecto en `experiments/.local/`. Usa un directorio por ejecución si el programa solicita `--output-dir`.

## Ejecución de experimentos

| Flujo | Punto de entrada y recursos |
| --- | --- |
| Cuaderno de trabajo | [Cuaderno de trabajo](../experiments/notebooks/workflow_javi_aitor.ipynb). El modo `smoke` usa 105 filas de desarrollo y 36 de validación. Demuestra el traductor con una respuesta simulada y reutiliza la traducción archivada si está disponible. Activa `COMPARE_TOKENIZERS`, `RUN_MINILM` o `TUNE_MINILM` según la prueba. |
| Comparación clásica | `python -m experiments.scripts.run_algorithm_comparison --group-by-premise`. `--classifiers`, `--n-splits`, `--skip-tune`, `--skip-holdout` y `--skip-submission` permiten una ejecución parcial. El protocolo por fila histórico se obtiene sin `--group-by-premise`. |
| Bancos por modelo | `python -m experiments.scripts.naive_bayes_experiments --list` (análogamente para los demás ejecutores por modelo). `run_all_experiments` recorre el catálogo y registra los resultados locales. El límite por señal funciona en plataformas con `SIGALRM`. `comparison_table` resume los bancos archivados y los locales. |
| Random Forest | [Cuaderno de Random Forest](../experiments/notebooks/random_forest_training.ipynb) para particiones, caché y entrenamiento. [Comandos del banco y confirmación](reports/random_forest_research.md#reproducción-de-tf-idf). |
| Embeddings históricos | `python -m experiments.scripts.rf_embeddings --cache-only --checkpoint RUTA_AL_CHECKPOINT` genera los cuatro arrays `train/val` × `p/h` de MiniLM. [Ejecución del banco completo y control E5](reports/random_forest_research.md#reproducción-de-embeddings). |
| Experimentos MiniLM | Ejecuta `run_dense_comparison`, `run_dense_tuning`, `run_dense_confirmation` y `run_dense_generalization` como módulos bajo `experiments.scripts`. Las fases leen los manifiestos, métricas, OOF y configuraciones archivados en el repositorio. `dense_vote_comparison` recalcula el análisis a partir de las probabilidades archivadas. |
| Comparación histórica de preprocesado y traducción | [Cuaderno de análisis](../experiments/notebooks/archive/comparacion_javi_aitor.ipynb). Sus tablas provienen de los resultados guardados por ID. |

Los ajustes densos históricos pueden requerir GPU y versiones concretas de CUDA y estimadores según el manifiesto. Los datos o checkpoints deben estar accesibles para las pruebas que los usan. La comparación exhaustiva consume más memoria y tiempo que las pruebas de integración.

El archivo de cuadernos incluye el [EDA](../experiments/notebooks/archive/eda.ipynb) de distribución, longitud y lengua, la [inspección inicial del esquema](../experiments/notebooks/archive/project.ipynb) y ejemplos del [preprocesador](../experiments/notebooks/archive/preprocesser_demo.ipynb), el [vectorizador](../experiments/notebooks/archive/vectorizer_demo.ipynb) y el [constructor de datasets](../experiments/notebooks/archive/builder_demo.ipynb).

## Reproducción de resultados

- **Preprocesado y traducción**. El informe recoge los protocolos históricos y la celda para repetir la comparación agrupada con los CSV archivados. [Reproducción de la comparación agrupada](reports/team_experiments.md#reproducción-de-la-comparación-agrupada) · [Materialización de variantes](reports/parameter_variants.md#materialización-y-caché)
- **Random Forest**. El informe indica cómo repetir la confirmación léxica y el experimento con embeddings. [TF-IDF](reports/random_forest_research.md#reproducción-de-tf-idf) · [Embeddings](reports/random_forest_research.md#reproducción-de-embeddings)
- **MiniLM**. Los manifiestos registran las particiones, configuraciones y resultados de la validación agrupada. [Protocolo, auditoría y métricas](../experiments/results/validation/minilm_grouped)

## Traducción

El corpus archivado se generó con Ollama, `gemma3:4b` Q4_K_M, digest de modelo `68ceab7e468a31c988b44d5a4984a0e21f58c4f7158c6ce29c1804babac38df7`. La implementación actual usa temperatura 0, lotes de seis por defecto y el servidor local `http://127.0.0.1:11434`. Los manifiestos conservan hashes SHA-256 de entrada y salida con los saltos originales CRLF. Usa los CSV archivados si quieres evaluar las mismas traducciones. [Configuración del traductor](../src/nli_project/translation/ollama.py) · [Manifiesto de train](../experiments/results/translation/corpus/train_gemma3_4b.csv.manifest.json) · [Manifiesto de test](../experiments/results/translation/corpus/test_gemma3_4b.csv.manifest.json)

Con el modelo disponible en Ollama, selecciona `MODE = "full"` y activa `GENERATE_FULL_TRANSLATIONS` en el [cuaderno de trabajo](../experiments/notebooks/workflow_javi_aitor.ipynb). El cuaderno reutiliza el CSV si ya está disponible. Para generar las traducciones de train y test, ejecuta esta celda desde la raíz.

```python
from nli_project.data.paths import repository_root
from nli_project.translation.ollama import translate_file

root = repository_root()
for split in ("train", "test"):
    translate_file(
        root / "experiments/data" / f"{split}.csv",
        root / "experiments/.local/translations" / f"{split}_gemma3_4b.csv",
        model="gemma3:4b",
        batch_size=6,
    )
```

El traductor escribe el CSV al completar los lotes. Si se interrumpe, vuelve a ejecutar la traducción desde el principio. Compara los hashes para comprobar equivalencia byte a byte. Para usar traducciones nuevas en la matriz histórica, cambia el archivo de entrada y conserva el mismo conjunto y orden de IDs. [Comparación agrupada](reports/team_experiments.md#reproducción-de-la-comparación-agrupada)

## Generación de entregas

### MiniLM en local y Kaggle

En local descomprime el checkpoint en `experiments/.local/minilm-checkpoint/` y ejecuta [el notebook MiniLM](../experiments/notebooks/kaggle_minilm.ipynb). El CSV y `minilm_run.json` quedan en `experiments/.local/minilm_submission/`. El siguiente comando comprueba el CSV generado.

```sh
python -m nli_project.submission.minilm_submission experiments/.local/minilm_submission/submission.csv experiments/data/sample_submission.csv
```

Para Kaggle genera el ZIP mediante `python -m experiments.scripts.bundle`. Adjunta los datos oficiales, el ZIP de código `experiments/.local/minilm-source.zip` y el checkpoint **descomprimido** como entradas distintas. El notebook encuentra el paquete de código comprimido o extraído y valida los archivos del checkpoint mediante SHA-256. El archivo se genera en `/kaggle/working/submission.csv`. Guarda la versión ejecutada y comprueba el CSV antes de elegir esa versión para la competición. [Checkpoint de referencia](https://www.kaggle.com/datasets/atsunorifujita/paraphrasemultilingualminilml12v2) · [CSV de la ejecución archivada](../experiments/submissions/minilm_tokenizer_corrected/submission.csv)

### CatBoost traducido

El [cuaderno histórico de CatBoost](../experiments/notebooks/archive/kaggle_submission_final.ipynb) usa el paquete de datos procesados y código que genera `python -m experiments.scripts.prepare_kaggle_delivery` en `experiments/.local/kaggle_delivery/`. Adjunta ese directorio al cuaderno. [Configuración archivada](../experiments/results/translation/kaggle_catboost/run_manifest.json) · [CSV de entrega](../experiments/submissions/catboost_translated/submission.csv)
