# Javi y Aitor · Revisión del 19 de noviembre

[workflow_javi_aitor.ipynb](../workflow_javi_aitor.ipynb) es el punto de entrada para ejecutar el flujo. Selecciona el kernel de `0-AiDea/.venv` con `ipykernel` y el modelo local de spaCy `en_core_web_sm`. El notebook admite kernels iniciados desde `training` y desde `0-AiDea`.

- `MODE = "smoke"` usa 105 filas de desarrollo y 36 de validación. El traductor se demuestra con una respuesta simulada. El brazo traducido entra en la comparación si existe un CSV compatible.
- `MODE = "full"` usa las particiones completas y añade CatBoost. Activa `GENERATE_FULL_TRANSLATIONS` para solicitar a Ollama local que genere el CSV si falta. `TRY_OLLAMA` solo prueba el ejemplo de traducción.
- La comparación usa preprocesado neutral en ambos brazos y deja fuera POS y las reglas de relación inglesas. La matriz permite activarlas para experimentos específicos mediante `use_pos_features` y `use_relation_features`.
- El notebook guarda los resultados nuevos bajo `artifacts/workflow_javi_aitor/`. La matriz reutiliza ejecuciones completas por configuración y huella de los datos, y publica las salidas al terminar. Un intento interrumpido puede repetirse desde el inicio. El [cuaderno histórico](../comparacion_javi_aitor.ipynb) lee los resultados archivados.

| Tarea | Qué se puede revisar hoy |
| --- | --- |
| [#7 · Permutador](https://github.com/Aitor42/0-AiDea/issues/7) | [`permutator.py`](permutator.py) aplica las opciones del preprocesador de Alba. Genera un pickle y un manifiesto por variante. Conserva el orden, los IDs, el idioma, la etiqueta si existe y las columnas de entrada. |
| [#11 · Traducción](https://github.com/Aitor42/0-AiDea/issues/11) | [`translator.py`](translator.py) traduce premisa e hipótesis al inglés por lotes de un mismo idioma con Ollama. Asocia la respuesta al ID de origen y añade dos columnas. Los pares originalmente ingleses se copian tal cual. |
| [#12 · Modelos](https://github.com/Aitor42/0-AiDea/issues/12) | [`classifiers.build_classifier`](../classifiers.py) es la fábrica de Aitor que usa esta rama. Construye un estimador sin ajustar a partir del nombre, la semilla y los hiperparámetros. |
| [#13 · Comparación](https://github.com/Aitor42/0-AiDea/issues/13) | Los [bancos por modelo](../../model_experiments.py) de Aitor usan el mismo pipeline y evaluador que [la matriz de variantes](../run_experiment_matrix.py). La matriz organiza datasets y guarda resultados. Con agrupamiento, comprueba que las variantes conservan la misma premisa original para formar los folds. |

## Uso desde notebooks

Las dos primeras celdas de `workflow_javi_aitor.ipynb` preparan el entorno y cargan los datos. Después puedes ejecutar cada componente por separado. La celda de MiniLM muestra los resultados guardados y solo entrena si activas `RUN_MINILM` y aún no hay una ejecución guardada. Necesita pesos locales, el CSV traducido y CUDA para XGBoost. MiniLM puede codificar en CPU.

La celda **Ajuste de MiniLM con el preprocesador de Alba** se activa con `TUNE_MINILM` y usa toda la partición de desarrollo. Compara las recetas por espacios y con spaCy, profundidades de CatBoost y XGBoost, y 300, 768 o todas las características. Mantiene el texto sin preprocesar como referencia histórica. Requiere pesos locales de MiniLM y CUDA para los clasificadores. La tabla es de selección por validación cruzada agrupada.

Si necesitas regenerar el paquete para Kaggle, importa `prepare_delivery` de `prepare_kaggle_delivery` y llama a esa función en una celda después de la preparación. La función requiere los CSV traducidos de train y test.

## Conexión con el proyecto de equipo

La revisión toma como referencia `Aitor42/training@a5b6809` y `Aitor42/0-AiDea@c18c886`. El preprocesador de Alba, el vectorizador y el wrapper de Félix coinciden con los de ese estado del repositorio compartido.

- Para usar el wrapper de Félix, pásale el DataFrame de entrada y fija `language=None` en `preprocessor_params` para conservar todas las filas. Si contiene traducciones, configura también `columns=("premise_translated", "hypothesis_translated")`. Ajusta el wrapper con `mode="train"` y reutiliza ese objeto con `mode="val"` para transformar validación. El pickle del permutador ya contiene las columnas procesadas y puede alimentar el vectorizador directamente.
- Cuando Akil entregue el [bucle](https://github.com/Aitor42/0-AiDea/issues/9) y el [evaluador](https://github.com/Aitor42/0-AiDea/issues/10), habrá que confirmar cómo reciben las particiones y los estimadores, dónde se ajusta el vectorizador y cómo se devuelven las predicciones por ID. Con esos contratos se conectarán las variantes y la fábrica. El balanceo sigue pendiente de la [entrega de Dario](https://github.com/Aitor42/0-AiDea/issues/8).

## Guion breve para la revisión

Queríamos medir qué aporta cambiar el preprocesado y traducir al inglés para clasificar relaciones entre premisa e hipótesis. El permutador prepara las variantes con el código de Alba y el traductor añade una versión inglesa de las dos frases sin perder el par de origen. Comparamos sobre los mismos ejemplos con la infraestructura de Aitor. En el experimento histórico de tres folds agrupados por premisa original, la accuracy de CatBoost pasó de 52,19 % a 54,82 % al usar la traducción. Falta conectar estos componentes al bucle y al evaluador de Akil cuando se conozcan sus interfaces. [Cuaderno ejecutado](../comparacion_javi_aitor.ipynb) · [Resultados guardados](../results_translation_cv/results.json)
