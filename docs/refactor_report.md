# Reorganización de la PR #14

La revisión parte de `3bdb09a1ade215aeca7566606e0aeac789558e50` sobre `c18c88655f1145064c0d45fac0796be239e74463`. Se hizo en un checkout aislado de `review/proyecto-completo`. La publicación se destina a la rama homónima de `Aitor42/training`. [PR de origen](https://github.com/Aitor42/0-AiDea/pull/14)

## Árbol versionado

```text
0-AiDea/
├── docs/
│   ├── PROJECT_GUIDE.md
│   ├── reproducibility.md
│   ├── refactor_inventory.tsv
│   ├── refactor_report.md
│   └── reports/
├── src/
│   ├── nli_project/
│   │   ├── data/
│   │   ├── preprocessing/
│   │   ├── features/
│   │   ├── models/
│   │   ├── training/
│   │   ├── evaluation/
│   │   ├── translation/
│   │   └── submission/
│   └── tests/
├── experiments/
│   ├── configs/
│   ├── scripts/
│   ├── notebooks/
│   │   └── archive/
│   ├── results/
│   │   ├── classical_models/
│   │   ├── random_forest/
│   │   ├── minilm/
│   │   ├── translation/
│   │   ├── ensembles/
│   │   └── validation/
│   └── submissions/
├── .gitattributes
├── .gitignore
├── README.md
├── pyproject.toml
├── pytest.ini
└── requirements.txt
```

`experiments/data/` y `experiments/.local/` son directorios de trabajo ignorados. El [inventario por archivo](refactor_inventory.tsv) recoge las rutas de origen incluidas en la revisión y documenta su destino o exclusión.

| Origen | Destino |
| --- | --- |
| Componentes de `FINAL/`, scripts reutilizables de raíz | `src/nli_project/` por responsabilidad |
| Comparadores, ejecutores, bancos y parámetros compartidos | `experiments/scripts/` y `experiments/configs/` |
| Notebooks activos y cuadernos históricos | `experiments/notebooks/` y `experiments/notebooks/archive/` |
| `results_*`, bancos JSONL y evidencias de `artifacts/` | `experiments/results/` por familia |
| CSV candidatos emitidos por experimentos | `experiments/submissions/` |
| Tests | `src/tests/` |
| Informes y documentación anterior | `docs/reports/` |

## Conservación y exclusiones

Se retiró un documento interno de la rama de revisión.

- **101 evidencias experimentales de la PR** conservan exactamente el mismo blob Git, incluidos los CSV de traducción, manifiestos, métricas, configuraciones, tablas de referencia, figuras, predicciones y los dos CSV del split histórico. Se conservaron **21 CSV comprimidos** de probabilidades y predicciones, porque sirven para reproducir votos, confirmar configuraciones y comprobar comparaciones por ID. Los CSV traducidos se escriben en checkout con CRLF para preservar los hashes de sus manifiestos.
- La entrega MiniLM corregida añade el [CSV archivado y su manifiesto](../experiments/submissions/minilm_tokenizer_corrected) que documentaba la PR. El CSV coincide por SHA-256 con la ejecución citada. Su manifiesto histórico también referencia las probabilidades y un muestreo de embeddings de aquella ejecución. El CSV y el manifiesto se incluyen como evidencia de la entrega.
- Se retiraron del árbol nuevo los tres CSV oficiales, descargables desde Kaggle y verificables con el validador MiniLM. Se conservan los splits históricos, pues son referencia para comprobar los IDs y el orden exactos de los experimentos.
- Se retiraron el log y el registro de avisos del proceso de traducción, junto con tres PID transitorios. El log registraba dos métricas ya conservadas con toda su precisión en el [holdout de CatBoost](../experiments/results/translation/catboost_holdout/metrics.json).
- Se retiraron seis archivos de compatibilidad o de inicialización del antiguo espacio `FINAL`. Los destinos de la implementación que antes exponían figuran en el inventario. Los notebooks retienen sus celdas de código y explicaciones. Se quitaron gráficos, tablas grandes, avisos del entorno y otras salidas regenerables. Las métricas experimentales se conservan en registros estructurados. Las representaciones originales de los cuadernos siguen disponibles en el commit de origen.
- Las rutas de seis ficheros experimentales históricos se conservan como metadatos originales de las corridas. Los ejecutores activos y notebooks localizan sus entradas usando rutas del repositorio y el inventario documenta las equivalencias. Los modelos, pesos, cachés, logs nuevos y recursos privados se excluyen selectivamente con `.gitignore`.

## Imports, resultados y documentación

El código reutilizable usa `nli_project`. La instalación editable también expone `experiments.scripts` para ejecutar módulos con `python -m`. Pytest descubre las pruebas en `src/tests`. Los notebooks locales arrancan con el kernel instalado. Los dos notebooks de Kaggle añaden el directorio `src` del paquete de código al arrancar en el entorno remoto. Los resultados nuevos van a `experiments/.local/` o al destino explícito de la ejecución. Las rutas históricas se mantienen como archivos de entrada en las comparaciones correspondientes.

La [guía central](PROJECT_GUIDE.md) integra datos, metodología, catálogo, métricas y receta elegida. La [guía de reproducción](reproducibility.md) reúne dependencias, comandos, traducción y pasos para preparar las entregas.

## Verificación

- `python -m pytest -q` en Windows, en un entorno limpio con Python 3.14.7 y los datos oficiales. Resultado de **142 pruebas aprobadas** y un aviso de convergencia del MLP sintético.
- Se importaron 59 submódulos desde fuera del checkout en un entorno virtual nuevo, con instalación editable y sin `PYTHONPATH`. `pip check` confirmó dependencias consistentes.
- Se ejecutaron el cuaderno de trabajo en modo reducido y el análisis histórico de Javi y Aitor, las celdas de arranque y partición de Random Forest y celdas de los cinco cuadernos de demostración. Se probó la comparación clásica agrupada con 120 pares y se recalculó el voto MiniLM a partir de los OOF archivados. La carga de caché alineó 9.696 filas con sus 1.536 rasgos.
- Se ejecutó la preparación de CatBoost con los corpus traducidos completos y se comprobó el notebook con un montaje equivalente al de Kaggle, 135 pares de entrenamiento y 12 predicciones.
- Se comprobó el notebook MiniLM con un ZIP nuevo, el checkpoint real verificado, 60 pares de ajuste y 12 de inferencia en un montaje local equivalente. Se validaron los cuatro CSV archivados, cada uno con 5.195 predicciones en el orden oficial.
- Se comprobó la resolución de enlaces internos, los tres directorios versionados de la raíz, la ausencia de rutas de máquinas personales en código y cuadernos activos, la identidad de los 101 blobs y los hashes de traducción. Los entrenamientos densos completos, la regeneración con Gemma 3 y una ejecución nueva en Kaggle quedan fuera de estas comprobaciones.

## Tamaño y número de archivos

| Medida | PR #14 original | Reorganización | Diferencia |
| --- | ---: | ---: | ---: |
| Archivos versionados en el árbol | 189 | 201 | +12 |
| Bytes del contenido versionado | 34.246.333 | 29.616.231 | −4.630.102 |
| Entradas del diff respecto a `main@c18c886` | 178 | 206 | +28 |

El tamaño suma los blobs del árbol, sin incluir el historial Git ni los recursos locales ignorados. Los movimientos y la separación de módulos generan más entradas de diff contra la base.
