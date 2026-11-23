# Entorno y reproducción

## Instalación local

Crea y activa un entorno virtual (Python 3.14.7 en la verificación de esta reorganización). La instalación editable hace importables `nli_project` y los ejecutores `experiments.scripts`. Usa el mismo Python para Jupyter y pytest.

```sh
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python -m pip install jupyterlab seaborn
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt punkt_tab stopwords wordnet omw-1.4
```

Los recursos NLTK se usan en las variantes que seleccionan ese tokenizador, stopwords o WordNet. En CPU puedes usar el índice oficial de ruedas de PyTorch mediante `--extra-index-url https://download.pytorch.org/whl/cpu`. Descarga `train.csv`, `test.csv` y `sample_submission.csv` y colócalos en `experiments/data/`. El cuaderno de MiniLM comprueba sus hashes. Los tests de particiones también usan el train oficial. [Datos de Kaggle](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data) · [Hashes de referencia](../src/nli_project/submission/minilm_submission.py)

Ejecuta `python -m pytest` y abre `python -m jupyter lab` desde la raíz. Los nuevos resultados y cachés predeterminados se guardan en `experiments/.local/`. Mantén un directorio de salida por corrida, especialmente cuando un ejecutor solicita `--output-dir`.

## Puntos de entrada

| Flujo | Punto de entrada y recursos |
| --- | --- |
| Matriz Javi y Aitor | [Cuaderno de trabajo](../experiments/notebooks/workflow_javi_aitor.ipynb). El modo `smoke` usa 105 filas de desarrollo y 36 de validación. Demuestra el traductor con una respuesta simulada y reutiliza la traducción archivada si está disponible. Activa `COMPARE_TOKENIZERS`, `RUN_MINILM` o `TUNE_MINILM` según la prueba. |
| Comparación clásica | `python -m experiments.scripts.run_algorithm_comparison --group-by-premise`. `--classifiers`, `--n-splits`, `--skip-tune`, `--skip-holdout` y `--skip-submission` permiten una ejecución parcial. El protocolo por fila histórico se obtiene sin `--group-by-premise`. |
| Bancos por modelo | `python -m experiments.scripts.naive_bayes_experiments --list` (análogamente para los demás ejecutores por modelo). `run_all_experiments` recorre el catálogo y registra los resultados locales. El límite por señal funciona en plataformas con `SIGALRM`. `comparison_table` resume los bancos archivados y los locales. |
| Random Forest | [Cuaderno RF](../experiments/notebooks/random_forest_training.ipynb) para particiones, caché y entrenamiento. `python -m experiments.scripts.rf_experiments --list` lista variantes. `rf_confirm` repite el protocolo de confirmación y `rf_summary` vuelve a calcular la tabla. |
| Embeddings históricos | `python -m experiments.scripts.rf_embeddings --cache-only --checkpoint RUTA_AL_CHECKPOINT` genera los cuatro arrays `train/val` × `p/h` de MiniLM. Para E5, pasa el modelo como argumento posicional y replica el prefijo de su [manifiesto](../experiments/results/random_forest/embeddings/rf_embeddings_results_intfloat__multilingual-e5-small.json) mediante `EMB_PREFIX`. Para el banco RF completo ejecuta antes las celdas de preprocesado del cuaderno RF. |
| Experimentos MiniLM | Ejecuta `run_dense_comparison`, `run_dense_tuning`, `run_dense_confirmation` y `run_dense_generalization` como módulos bajo `experiments.scripts`. Las fases leen los manifiestos, métricas, OOF y configuraciones archivados en el repositorio. `dense_vote_comparison` recalcula el análisis a partir de las probabilidades archivadas. |
| Comparación histórica del equipo | [Cuaderno de análisis](../experiments/notebooks/archive/comparacion_javi_aitor.ipynb). Sus tablas provienen de los resultados guardados por ID. |
| CatBoost traducido para Kaggle | `python -m experiments.scripts.prepare_kaggle_delivery` empaqueta los CSV traducidos como datos procesados y código. Adjunta el directorio `experiments/.local/kaggle_delivery/` a [su cuaderno histórico](../experiments/notebooks/archive/kaggle_submission_final.ipynb). |

Los ajustes densos históricos pueden requerir GPU y versiones concretas de CUDA y estimadores según el manifiesto. Los datos o checkpoints deben estar accesibles para las pruebas que los usan. La comparación exhaustiva consume más memoria y tiempo que las pruebas de integración.

Los cuadernos de archivo cumplen funciones diferentes. EDA contiene los análisis de distribución, longitud y lengua. El cuaderno `project.ipynb` conserva la inspección inicial del esquema. Los otros tres demuestran el preprocesador, el vectorizador y el constructor de datasets por separado.

## Traducción

El corpus archivado se generó con Ollama, `gemma3:4b` Q4_K_M, digest de modelo `68ceab7e468a31c988b44d5a4984a0e21f58c4f7158c6ce29c1804babac38df7`. La implementación actual usa temperatura 0, lotes de seis por defecto y el servidor local `http://127.0.0.1:11434`. Los manifiestos conservan hashes SHA-256 de entrada y salida con los saltos originales CRLF. La historia conserva los CSV completos porque permiten repetir las comparaciones con exactamente las traducciones medidas. [Configuración del traductor](../src/nli_project/translation/ollama.py) · [Manifiesto de train](../experiments/results/translation/corpus/train_gemma3_4b.csv.manifest.json) · [Manifiesto de test](../experiments/results/translation/corpus/test_gemma3_4b.csv.manifest.json)

Con el modelo ya disponible en Ollama, puedes activar las opciones explícitas en el cuaderno o ejecutar el siguiente bloque en una celda.

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

Compara la huella de una regeneración si necesitas comprobar equivalencia byte a byte. Para utilizar las traducciones nuevas en la matriz histórica, pásalas explícitamente y conserva el mismo conjunto y orden de IDs. [Comparación del equipo](reports/team_experiments.md)

## Entrega MiniLM

En local descomprime el checkpoint en `experiments/.local/minilm-checkpoint/` y ejecuta [el notebook MiniLM](../experiments/notebooks/kaggle_minilm.ipynb). El CSV y `minilm_run.json` quedan en `experiments/.local/minilm_submission/`. El siguiente comando comprueba el CSV generado.

```sh
python -m nli_project.submission.minilm_submission experiments/.local/minilm_submission/submission.csv experiments/data/sample_submission.csv
```

Para Kaggle genera el ZIP mediante `python -m experiments.scripts.bundle`. Adjunta los datos oficiales, el ZIP de código `experiments/.local/minilm-source.zip` y el checkpoint **descomprimido** como entradas distintas. El notebook encuentra el paquete de código comprimido o extraído y valida los archivos del checkpoint mediante SHA-256. El archivo se genera en `/kaggle/working/submission.csv`. Guarda la versión ejecutada y comprueba el CSV antes de elegir esa versión para la competición. [Checkpoint de referencia](https://www.kaggle.com/datasets/atsunorifujita/paraphrasemultilingualminilml12v2) · [CSV de la ejecución archivada](../experiments/submissions/minilm_tokenizer_corrected/submission.csv)
