# Bloque 2: modelado de tópicos con BERTopic (resumen)

Semilla 42 en todo. Ejecutado el 19 de noviembre de 2026.

## 1. Qué se hizo

| Notebook | Contenido |
|---|---|
| [`topic_eda.ipynb`](../topic_eda.ipynb) | EDA del corpus de tópicos y auditoría de la partición del grupo |
| [`topic_modeling.ipynb`](../topic_modeling.ipynb) | Embeddings en caché, BERTopic con HDBSCAN y con K-means, interpretación de los tópicos, tópicos frente a clases y frente a idioma |
| [`topic_experiment.ipynb`](../topic_experiment.ipynb) | CatBoost y Naive Bayes del Bloque 1, con y sin tópico como feature |

- **Corpus:** todas las filas del train oficial (12.120 pares, 15 idiomas), con el texto traducido al inglés (`premise_translated`, gemma3:4b), como el mejor modelo con n-gramas del Bloque 1.
- **Documento = premisa sin repetir**, deduplicada por el texto original. Son 8.209 documentos: 7.051 de train para ajustar. Las 2.250 premisas de val (incluidas las 1.092 que también están en train) reciben su tópico con `transform`. Cada fila hereda el tópico de su premisa. En las premisas compartidas, `transform` coincide con el `fit` en el 94,1 % (HDBSCAN) y el 96,7 % (K-means) de los casos.
- **Embeddings:** `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensiones, normalizados), guardados en `artifacts/embeddings/topics/`.
- **Pipelines:** mismo UMAP (`n_neighbors=15, n_components=5, min_dist=0, cosine, seed 42`) y mismo c-TF-IDF (stopwords inglesas + *uh, um, yeah, oh*; 1–2-gramas; `min_df=2`). Solo cambia el clustering.
- **Partición:** la del grupo (`artifacts/splits/`), sin crear otra.
- **Qué entrenamos:** UMAP, HDBSCAN, K-means, el vocabulario del c-TF-IDF, el one-hot del tópico y los clasificadores (CatBoost y Naive Bayes, junto con el TF-IDF y la selección chi² del Bloque 1), siempre solo con train. El modelo de embeddings se usa preentrenado, sin ajustarlo.

## 2. Resultados clave

### 2.1. Los dos pipelines (ajustados con 7.051 premisas de train)

| | A · HDBSCAN (`min_cluster_size=30`) | B · K-means (k=20) |
|---|---:|---:|
| Tópicos | 20 | 20 |
| % de outliers (−1) | 30,9 % | 0 % |
| Tamaño máximo / mediano / mínimo | 1.762 / 107 / 34 | 701 / 350 / 171 |
| Diversidad (palabras distintas en las top-10) | 0,895 | 0,820 |
| NMI con el idioma | 0,026 | 0,021 |
| Tiempo de ajuste (UMAP + clustering + c-TF-IDF) | 8,9 s | 8,9 s |

- **Por qué `min_cluster_size=30`:** con 15 salen 55 tópicos y un 38 % de outliers; con 50 salen 19 tópicos, pero un 45 % de outliers. Con 30 hay 20 tópicos interpretables y el menor % de outliers.
- **Por qué k=20:** la silhouette es casi plana entre k=10 y k=40 (0,34–0,37), así que se iguala al número de tópicos de A para que la comparación sea justa.
- **Los tópicos son los géneros de MultiNLI:** gobierno, viajes, ficción, conversaciones telefónicas, Slate y el informe del 11-S. Esto confirma el origen XNLI/MultiNLI de los datos.
- **A** separa muy bien los temas pequeños (11-S, ficción histórica, Clinton, música). Su defecto es un tópico "gobierno" que reúne el 25 % de los documentos.
- **B** parte ese bloque del gobierno en 4 temas, pero crea 2–3 clústeres "cajón de sastre" con los documentos que HDBSCAN habría marcado como outliers.
- **Comunes a los dos:** el informe del 11-S, la ficción histórica, "lengua y palabras" y "texto sin traducir". Que aparezcan con los dos algoritmos indica que son temas reales.

### 2.2. Tópicos frente a clases NLI y frente a idioma (filas de train)

| | χ² (g.l.) | p | V de Cramér |
|---|---:|---:|---:|
| A · HDBSCAN | 40,0 (40) | 0,47 | 0,045 |
| B · K-means | 30,3 (38) | 0,81 | 0,040 |

- **El tópico no está asociado con la clase.** Para cada premisa hay hipótesis de las tres clases.
- **Los tópicos agrupan por tema, no por idioma:** NMI de 0,02–0,03. Con el texto original (control), la NMI sube a 0,05–0,07, pero sigue siendo baja, porque MiniLM es multilingüe.

### 2.3. Experimento controlado (val, 2.424 filas; mismos IDs, semilla e hiperparámetros)

| Clasificador | Brazo | Accuracy | Macro-F1 | AUROC macro | Δacc frente a A [IC 95 %] | McNemar p |
|---|---|---:|---:|---:|---|---:|
| CatBoost | A (X del Bloque 1) | 0,5656 | 0,5655 | 0,7406 | — | — |
| CatBoost | + one-hot del tópico HDBSCAN | 0,5639 | 0,5638 | 0,7407 | −0,17 pp [−1,20; +0,74] | 0,80 |
| CatBoost | + one-hot del tópico K-means | 0,5639 | 0,5637 | 0,7413 | −0,17 pp [−1,11; +0,83] | 0,80 |
| Naive Bayes | A (X del Bloque 1) | 0,4930 | 0,4900 | 0,6673 | — | — |
| Naive Bayes | + one-hot del tópico HDBSCAN | 0,4905 | 0,4871 | 0,6662 | −0,25 pp [−0,78; +0,29] | 0,45 |
| Naive Bayes | + one-hot del tópico K-means | 0,4880 | 0,4848 | 0,6652 | −0,50 pp [−1,07; +0,17] | 0,16 |

- **El brazo A reproduce exactamente el Bloque 1:** 0,565594 en CatBoost (`results_translation_cv/catboost_holdout.json`) y 0,492987 en Naive Bayes.
- **Resultado negativo:** los tópicos no mejoran ningún clasificador y ninguna diferencia es significativa.
- **CatBoost apenas los usa:** las columnas de tópico reciben un 1–1,6 % de la importancia total.
- **Premisa vista frente a nueva:** CatBoost acierta un 59,5 % con premisa vista y un 53,7 % con premisa nueva. Los tópicos no cambian nada en ninguno de los dos subconjuntos.
- **Errores por tópico (CatBoost):** con K-means, el error va del 35 % en "gestión pública" (n=272) al 55 % en "negación: no sé / no creo" (n=127). El tópico HDBSCAN "texto sin traducir" tiene un 67 % de error (n=27). Aquí está la utilidad real de los tópicos: diagnosticar dónde falla el modelo, no usarlos como feature.

## 3. Conclusiones

1. HDBSCAN es mejor para interpretar, porque deja fuera lo que no encaja. K-means es mejor para asignar un tópico a todo, a costa de crear clústeres sin sentido.
2. Los tópicos describen el contenido (género) de la premisa, que en un dataset NLI bien construido es independiente de la clase (V ≈ 0,04).
3. Por eso, añadirlos al clasificador no aporta (Δ ≈ 0 y p ≥ 0,15). Es un resultado negativo, pero está explicado. Para que ayudaran harían falta features que relacionen premisa e hipótesis, por ejemplo si caen en el mismo tópico.
4. Como herramienta de análisis de errores sí son útiles: muestran que fallan más las premisas con negaciones, el informe del 11-S y el texto que no se tradujo.

## 4. Limitaciones

- **Partición no agrupada por premisa:** 1.092 premisas compartidas, así que el 49 % de val (1.188 filas) tiene una premisa ya vista en train. Además, "vista" es casi siempre "inglesa" (1.099 de 1.188). Se mitiga repitiendo las métricas por `premise_seen`.
- **Traducción automática incompleta:** el 33 % de las filas no inglesas (1.717 de 5.250) tiene la premisa idéntica al original, es decir, sin traducir; en chino son 312 de 411. Esto produce un tópico de "texto sin traducir" y más errores del clasificador en él. Además, la traducción se hizo por par, así que 198 premisas tienen varias traducciones distintas; se usa la primera.
- **Posible solapamiento del modelo de embeddings:** los modelos `paraphrase-*` se entrenaron con AllNLI (SNLI + MultiNLI), y esta competición viene muy probablemente de XNLI (MultiNLI traducido). El modelo puede haber visto algunas premisas inglesas. Para los tópicos importa poco (no se usan etiquetas); para el MiniLM del Bloque 1 podría inflar el resultado.
- **Entorno:** Python 3.11.9 en lugar de 3.12 (en el PC no hay 3.12). Por eso numpy 2.4.6, scipy 1.17.1 y xgboost 3.2.0 están por debajo de los mínimos de `requirements.txt`, que exigen Python ≥ 3.12. Los 142 tests del repo pasan y el Bloque 1 se reproduce al dígito.
- **El p-valor del χ² es algo optimista:** las filas de una misma premisa comparten tópico y no son independientes. La conclusión se apoya en la V de Cramér.
- **Una sola partición y una sola semilla:** los IC bootstrap cuantifican la variabilidad del conjunto de val, pero no la del entrenamiento.

## 5. Problemas encontrados en código o artefactos ajenos (no se han modificado)

1. **`FINAL/translator.py` no detecta traducciones fallidas.** Valida IDs y textos no vacíos, pero acepta como "traducción" un texto idéntico al original. En `artifacts/translations/train_gemma3_4b.csv`, 1.717 premisas y 804 hipótesis no inglesas están sin traducir. Propuesta: reintentar los textos cuya salida coincida con la entrada o no esté en alfabeto latino.
2. **`requirements.txt` exige Python ≥ 3.12** (`numpy>=2.5`, `scipy>=1.18`, `xgboost>=3.4`). En 3.11 la instalación falla. No es un bug, pero conviene indicarlo en el README.
3. **`FINAL/compare_ollama.py` ya no reproduce `results_ollama_catboost/metrics.json`.** La versión actual usa la receta neutral sin POS ni relación, mientras que el resultado guardado (0,5656) se obtuvo con lemas + POS + relación. `JAVI_AITOR.md` lo explica, pero puede confundir. La receta que sí lo reproduce es la del fragmento de código de `JAVI_AITOR.md` (`PREPROCESSING_PARAMS` + `build_pipeline(..., use_pos_features=True, use_relation_features=True)`), que es la usada aquí.
4. **`results_translation_cv/results.json` no guarda la clave `spec`.** Los flags de POS y relación no quedan registrados en el propio resultado.
5. **Nota para el grupo:** la `X` de referencia del Bloque 1 (CatBoost con texto traducido, 0,5656) la construye `training.build_pipeline` sobre los datasets de `parameter_variants`, no `builder.py`. Hay 8.404 premisas traducidas distintas para 8.209 premisas originales, por las traducciones por par.

## 6. Archivos creados

Notebooks (raíz del repo, ejecutados con sus salidas): `topic_eda.ipynb`, `topic_modeling.ipynb`, `topic_experiment.ipynb`.

`results_topics/` (se entrega):
- Tablas:
  - `comparacion_pipelines.csv`;
  - `topicos_hdbscan.csv` y `topicos_kmeans.csv` (nombre, top-10 palabras y 3 documentos representativos);
  - `topicos_vs_clase_*.csv` y `topicos_vs_idioma_nmi.csv`;
  - `topics_rows.csv` (`id, part, topic_hdb, topic_km`) y `val_premise_seen.csv`;
  - `experimento.csv`, `experimento_pruebas.csv` y `errores_por_topico_{hdb,km}.csv`.
- Figuras:
  - `eda_longitudes.png`, `fig_tamanos_topicos.png`;
  - `fig_palabras_{hdbscan,kmeans}.png`, `fig_mapa_2d.png`;
  - `fig_clases_por_topico_{hdbscan,kmeans}.png`, `fig_idioma_por_topico.png`;
  - `fig_confusion_{catboost,naive_bayes}.png`, `fig_roc_{catboost,naive_bayes}.png`, `fig_errores_por_topico_catboost.png`.
- Este resumen.

Caché (ignorada por git, se regenera sola):
- `artifacts/embeddings/topics/` (`emb_premises.npy`, `emb_premises_original.npy`, `premises.csv`; 27 MB);
- `artifacts/parameter_variants/preprocess_e328b9b92d60_*` y `artifacts/preprocessed/*.pkl`, creados por las funciones del grupo.

Paquetes instalados en `.venv`: catboost 1.2.10, spacy 3.8.18 + en_core_web_sm 3.8.0, nltk 3.10.4, xgboost 3.2.0, lightgbm 4.7.0 y pytest 9.2.0. Ya estaban: bertopic 0.17.4, umap-learn 0.6.0, hdbscan 0.8.44, kaleido 1.5.2, sentence-transformers 6.3.0, torch 2.15.0 (CPU), scikit-learn 1.9.1 y pandas 3.1.1.

## 7. Cómo reproducirlo todo, en orden

```
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace topic_eda.ipynb
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace topic_modeling.ipynb
.venv\Scripts\python -m jupyter nbconvert --to notebook --execute --inplace topic_experiment.ipynb
```

Tiempos en CPU:
- `topic_eda.ipynb`: ~10 s.
- `topic_modeling.ipynb`: ~3,5 min con la caché de embeddings. Sin caché, +~3,5 min (~105 s por cada conjunto de embeddings) y la descarga de MiniLM de HuggingFace.
- `topic_experiment.ipynb`: ~2 min. Sin la caché de preprocesado de spaCy, +~2,5 min.

Los resultados son deterministas: dos ejecuciones completas dieron los mismos tópicos y las mismas métricas.

## 8. Chuleta para la defensa

- **BERTopic:** técnica que encuentra temas en textos en cuatro pasos: frase → vector (embeddings) → reducir dimensiones (UMAP) → agrupar (clustering) → describir cada grupo con sus palabras más características (c-TF-IDF). No usa etiquetas.
- **Embeddings:** un vector de números por frase; frases con significado parecido tienen vectores cercanos. Usamos MiniLM multilingüe, que coloca una frase y su traducción casi en el mismo punto.
- **UMAP:** comprime 384 dimensiones a 5 conservando quién es vecino de quién, porque agrupar en dimensiones muy altas funciona mal. Es igual en los dos pipelines para que solo cambie el clustering.
- **HDBSCAN:** agrupa por densidad (zonas con muchos puntos juntos). Decide cuántos grupos hay y deja como outlier (−1) lo que no está en ninguna zona densa. `min_cluster_size` es el tamaño mínimo de un grupo.
- **K-means:** hay que decirle k. Pone k centros y asigna cada punto al más cercano, así que no hay outliers aunque un punto no encaje. Tampoco da probabilidades de pertenencia.
- **c-TF-IDF:** junta todos los documentos de un tópico en uno y busca las palabras frecuentes en ese tópico y raras en los demás. Solo describe los tópicos; no los forma.
- **Por qué premisas sin repetir:** el tema lo da la premisa, y cada premisa aparece 2–3 veces con hipótesis distintas. Repetirla crearía grupos de copias y contaría varias veces sus palabras. Cada fila hereda el tópico de su premisa.
- **Por qué one-hot:** el número de tópico es una etiqueta, no una cantidad ("tópico 7" no es mayor que "tópico 3"). El one-hot crea una columna 0/1 por tópico, y el −1 de HDBSCAN es una categoría más.
- **Fuga de datos:** usar información de validación al entrenar. Lo evitamos ajustando BERTopic, el one-hot, el vocabulario/TF-IDF y la selección chi² solo con train. A todo val, incluidas las premisas repetidas de train, solo se le aplica `transform`. Calcular embeddings de val no es fuga, porque el modelo ya viene entrenado.
- **ROC / AUROC:** para cada clase frente a las otras dos se mueve el umbral de probabilidad y se dibuja aciertos frente a falsas alarmas. El área bajo la curva vale 0,5 si el modelo es aleatorio y 1 si es perfecto. Necesita probabilidades; el macro es la media de las 3 clases.
- **V de Cramér:** fuerza de la relación entre dos variables categóricas (tópico y clase), de 0 a 1. Aquí sale ≈ 0,04, prácticamente ninguna relación.
- **NMI (información mutua normalizada):** cuánto dice una agrupación sobre otra (tópico ↔ idioma), de 0 a 1. Aquí ≈ 0,03: los tópicos no son idiomas.
- **McNemar:** compara dos modelos sobre los mismos ejemplos fijándose solo en los casos en que discrepan. Con 74 frente a 70 desacuerdos, p = 0,8: no hay diferencia.
- **Cómo interpretar el resultado (negativo):** los tópicos dicen de qué habla la premisa, pero la clase NLI depende de cómo se relaciona la hipótesis con ella, y para cada premisa hay hipótesis de las tres clases. Por eso el tópico no puede predecir la clase, y el experimento lo confirma (Δ ≈ 0, V ≈ 0,04). Aun así, los tópicos sirven para explicar errores: el modelo falla más en negaciones, en el 11-S y en el texto sin traducir.
- **Si hubiera salido positivo:** habría que comprobar que la mejora se mantiene con premisas nuevas (`premise_seen=False`) y que no se debe al solapamiento de premisas entre train y val.
