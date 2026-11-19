# Investigación de mejoras — Random Forest (NLI)

**Proyecto:** *Contradictory, My Dear Watson* (Kaggle) — Machine Learning 2026/27, ESI-UCLM.

Este documento recoge la investigación experimental sobre el pipeline de Random Forest
que ya está en `0-AiDea/training.py`. Todo lo que sigue está medido con el código del
repositorio y es reproducible con `rf_experiments.py`, `rf_confirm.py` y `rf_summary.py`.

---

## 1. Resumen ejecutivo

| Configuración | CV 5-fold (3 semillas) | Holdout reservado (accuracy / macro-F1) |
|---|---:|---:|
| Base actual | 0,4527 | 0,4814 / 0,4806 |
| **Propuesta** | **0,5146** | **0,5206 / 0,5193** |
| Diferencia | **+6,19 pts** | **+3,92 / +3,87 pts** |

La mejora se apoya en cuatro cambios, todos compatibles con «solo Random Forest»:

1. **Tokenización real de spaCy** en lugar de partir por espacios (mayor efecto).
2. **Selección de características chi²** dentro del pipeline, con `k = 300` (fit por fold).
3. **Unigramas** y `min_df = 5` en la representación `pair_diff`.
4. **Features de relación** calculadas fila a fila: negaciones, longitudes y números.

El bosque no se toca: los hiperparámetros actuales (`n_estimators=1000`,
`max_depth=50`, `min_samples_leaf=5`, `min_samples_split=5`, `max_features="sqrt"`,
`criterion="gini"`) siguen siendo la mejor zona encontrada.

---

## 2. Protocolo

- **Selección:** solo el split de entrenamiento (9.696 pares), CV estratificada de 5 folds.
- **Holdout:** los 2.424 pares reservados no se usan para elegir nada; solo confirman.
- **Sin fugas:** `SelectKBest` vive dentro del `Pipeline`, así que se ajusta en cada fold
  con el bloque de entrenamiento y nunca ve el fold de test. El preprocesado es fila a
  fila (no aprende del corpus) y el vectorizador se ajusta por fold con `clone()`.
  Es exactamente lo que exige el enunciado para selección de características y vocabulario.
- **Comparaciones pareadas:** mismas particiones para todas las configuraciones; se
  reportan los 5 folds y un t-test pareado, además de repetir la CV con 3 semillas.
- **Holdout:** 2.424 ejemplos implican un error estándar de ≈1,0 punto porcentual; solo
  diferencias mayores que ~2 puntos son concluyentes por sí solas.

---

## 3. Descomposición de la mejora

Cada fila añade **un** cambio sobre la anterior (CV 5-fold, semilla 42):

| Paso | Configuración | CV accuracy | Δ vs base |
|---|---|---:|---:|
| 0 | Base: whitespace, (1,2), `min_df=50` | 0,4525 | — |
| 1 | + chi² `k=500` | 0,4652 | +1,28 |
| 2 | + unigramas y `k=300` | 0,4668 | +1,43 |
| 3 | + tokenizador spaCy | 0,4868 | +3,43 |
| 4 | + features de negación | 0,4893 | +3,68 |
| 5 | + features de longitud | 0,4966 | +4,41 |
| 6 | + números (bloque de relación completo) | 0,5028 | +5,03 |
| 7 | + `min_df=5` | 0,5068 | +5,44 |
| 8 | + tokens sin sufijo `_POS` | **0,5133** | **+6,09** |

Atribución de los bloques de features de relación (sobre el paso 3):

| Bloque | CV accuracy | Δ sobre paso 3 |
|---|---:|---:|
| Negaciones | 0,4893 | +0,25 |
| Números | 0,4896 | +0,28 |
| Longitudes | 0,4966 | +0,98 |
| Relación completa (14 features) | 0,5028 | +1,60 |

Con la configuración final (`min_df=5` y tokens sin `_POS`) el pipeline genera
**19.107 features** y chi² elige 300: 287 de texto, **1** de ratios POS y 12 de
relación. Los ratios POS aportan +0,12 pts (dentro del ruido); se mantienen
porque son baratos y porque el enunciado pide POS tagging, que además se aplica
en el preprocesado.

---

## 4. Qué **no** funciona (o no aporta)

| Cambio | CV accuracy | Δ vs base | Lectura |
|---|---:|---:|---|
| `tfidf_svd` (LSA, 300 componentes) | 0,3719 | −8,05 | pierde la señal léxica exacta |
| `word_char` (palabra + carácter) | 0,4068 | −4,57 | mucho ruido y 16 s por fit |
| `pair_separate` (sin `|Xp−Xh|` ni solape) | 0,4141 | −3,84 | las features de diferencia son clave |
| Quitar stopwords | 0,4341 | −1,84 | destruye palabras función y negaciones |
| Bigramas/trigramas | 0,4558–0,4566 | +0,3 | sin mejora real |
| `class_weight="balanced"` | 0,4635 | −0,17 | las clases ya están casi balanceadas |
| `f_classif` en vez de chi² | 0,4645 | −0,22 | chi² encaja mejor con TF-IDF no negativo |
| One-hot de idioma | 0,4660 | −0,08 | el idioma no aporta tras el texto |
| `min_df` 20 / 100 | 0,4513 / 0,4506 | ≈0 | insensible en ese rango |
| `min_samples_leaf` 1 / 10, `max_depth=None`, `entropy` | 0,4628–0,4648 | ≤0 | la poda actual ya es adecuada |
| `max_features=0.33` | 0,4596 | −0,56 | 18 s por fit, peor |

Conclusión: el techo no estaba en el bosque ni en reducir dimensionalidad, sino en
**cómo se tokeniza y qué señales lógicas se le dan al modelo**.

---

## 5. Por qué funciona (teoría y práctica del curso)

- **Tokenización (L02-1).** Partir por espacios deja la puntuación pegada a las palabras
  (`movie.`, `don't`) y crea un vocabulario ruidoso. El tokenizador de spaCy separa
  puntuación y contracciones, lo que mejora todos los n-gramas posteriores. Es el mayor
  salto individual (+3,4 pts).
- **Stopwords (L02-1) y negaciones.** El enunciado avisa de no eliminar negaciones
  mecánicamente. Empíricamente, quitarlas cuesta 1,8–2,0 pts: en NLI las palabras función
  codifican la relación lógica, no solo el tema.
- **Stemming vs lematización (L02-2).** Porter, Snowball, WordNet y spaCy quedan a ±0,4 pts;
  no es un eje relevante aquí. Se mantiene `spacy_lemma` porque es el más rápido y ya está.
- **POS tagging (L03).** Etiquetar los tokens como `palabra_POS` fragmenta el vocabulario y
  cuesta 0,65 pts; los ratios POS como bloque apenas aportan tras las features de relación.
  Se conserva el POS tagging en el preprocesado (requisito y documentación), pero sin
  contaminar los tokens del TF-IDF.
- **Selección de características (Plan de trabajo).** Con 2.041 features, 311 tienen
  importancia casi nula y las 200 primeras concentran el 76 %. chi² con `k≈300` actúa como
  filtro de ruido: sube 1,3 pts y reduce el fit de 5,1 s a 2,6 s por fold.
- **Ensembles y poda (S05/S06).** `n_estimators`, `max_depth`, `min_samples_leaf` y
  `max_features` se probaron uno a uno: la configuración actual está en la meseta. Más
  regularización no ayuda porque el bosque ya está limitado por profundidad y hoja mínima.
- **Ingeniería de features.** La señal de negación es fuerte y verificable: cuando solo la
  hipótesis lleva negación, el 71 % de los pares son contradicción (frente al 29 % base).
  Las longitudes separan neutral de entailment/contradiction; los números añaden una pista
  de contradicción. Todo se calcula fila a fila, sin aprender del corpus, así que no hay fuga.

---

## 6. Configuración propuesta

```python
PREPROCESSING_PARAMS = {
    ...,
    "tokenizer": "spacy",          # antes: "whitespace"
    "reduction": "spacy_lemma",    # sin cambios
    "stopword_method": "none",     # sin cambios
    "use_pos": True,               # sigue alimentando los ratios POS
}

VECTORIZER_PARAMS = {
    "method": "pair_diff",
    "columns": ("premise_processed", "hypothesis_processed"),
    "min_df": 5,                   # antes: 50
    "word_ngram_range": (1, 1),    # antes: (1, 2)
    "use_pos": False,              # antes: True (tokens sin sufijo _POS)
}

# Pipeline: features -> SelectKBest(chi2, k=300) -> RandomForestClassifier
# features = FeatureUnion([NLITextVectorizer, POSRatioFeatures, RelationFeatures])
```

`RelationFeatures` (14 columnas, no negativas para ser compatible con chi²):

- negaciones en premisa, en hipótesis, diferencia absoluta y XOR;
- longitudes de premisa e hipótesis, diferencia absoluta, ratio y qué frase es más larga;
- números en cada frase, diferencia y Jaccard de números compartidos.

---

## 7. Confirmación

### CV repetida (3 semillas × 5 folds)

| Configuración | Media | Desv. entre semillas |
|---|---:|---:|
| Base | 0,4527 | 0,0006 |
| Unigramas + spaCy + relación + `min_df=5` | 0,5071 | 0,0002 |
| **+ tokens sin `_POS`** | **0,5146** | 0,0016 |

Comparación pareada de los 15 folds (3 semillas × 5): **+6,19 pts, 15/15 folds ganados,
p ≈ 1,2·10⁻¹¹**.

### Holdout (2.424 pares, nunca usado para elegir)

| Clase | Precisión | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
| entailment | 0,5423 | 0,5760 | 0,5587 | 835 |
| neutral | 0,4932 | 0,5155 | 0,5041 | 776 |
| contradiction | 0,5248 | 0,4686 | 0,4951 | 813 |

- Accuracy **0,5206** (base 0,4814), macro-F1 **0,5193** (base 0,4806).
- 1.262 aciertos frente a 1.167: +95 pares, sobre todo en `entailment` y `neutral`.
- La matriz de confusión sigue repartida en las seis direcciones; ya no hay una clase
  colapsada.

### Por lengua (holdout)

| Lengua | n | Base | Propuesta | Δ |
|---|---:|---:|---:|---:|
| English | 1.354 | 0,513 | 0,579 | +6,6 |
| Russian | 94 | 0,394 | 0,468 | +7,4 |
| Hindi | 86 | 0,616 | 0,500 | −11,6 |
| Spanish | 83 | 0,530 | 0,530 | 0,0 |
| Arabic | 79 | 0,367 | 0,418 | +5,1 |
| Thai | 77 | 0,286 | 0,351 | +6,5 |
| Bulgarian | 71 | 0,465 | 0,549 | +8,5 |
| French | 59 | 0,407 | 0,458 | +5,1 |

La mejora está concentrada en inglés (57 % del corpus); en el resto las muestras son de
60–100 pares (error estándar ≈5–6 pts), así que las diferencias individuales no son
concluyentes. Es coherente con usar herramientas de inglés sobre 15 idiomas: sigue siendo
la limitación principal.

---

## 8. Limitaciones y riesgos

- **Selección sobre CV.** Se han probado ~45 configuraciones sobre el mismo CV; el máximo
  puede estar inflado ~0,3 pts. La CV repetida y, sobre todo, el holdout independiente
  (+3,9 pts) reducen ese riesgo, pero no lo eliminan.
- **Features de longitud.** Parte de la ganancia viene de longitudes, un patrón del dataset
  más que comprensión semántica. Train y test comparten distribución (mismas lenguas y
  proporciones), así que debería transferir a Kaggle, pero conviene declararlo.
- **Inglés sobre 15 idiomas.** Sin traducción previa ni modelos multilingües, el techo de los
  idiomas no ingleses sigue bajo.
- **chi² y escalas.** chi² favorece features de mayor magnitud (las de relación) frente al
  TF-IDF; con `k=300` el efecto es limitado, pero conviene tenerlo presente.
- **Coste.** El pipeline nuevo es más barato: 300 features, ~2,6 s por fit (antes 5,1 s).

---

## 9. Reproducibilidad

Ficheros (en la raíz del proyecto, fuera del repositorio `0-AiDea`):

| Fichero | Contenido |
|---|---|
| `rf_experiments.py` | banco de configuraciones; `--list` para verlas |
| `rf_confirm.py` | CV repetida + holdout de las finalistas |
| `rf_summary.py` | tabla resumen y comparaciones pareadas |
| `rf_experiments_results.jsonl` | resultados por fold de cada ejecución |
| `rf_table.csv` | tabla consolidada |
| `rf_confirm.json` | confirmación de las configuraciones finales |

```bash
# Una configuración concreta
0-AiDea/.venv/bin/python rf_experiments.py sel300_uni_spacy_rel_mindf5_noposvec

# Confirmación (3 semillas + holdout)
0-AiDea/.venv/bin/python rf_confirm.py base sel300_uni_spacy_rel_mindf5_noposvec

# Tabla resumen
0-AiDea/.venv/bin/python rf_summary.py
```

La caché de preprocesado vive en `0-AiDea/artifacts/preprocessed` y se reutiliza por hash
de configuración y datos.

---

## 10. Embeddings: experimento controlado

**Pregunta.** La brecha por lengua (inglés 0,579 frente a no inglés 0,447, 13,2 pts)
apunta a que el TF-IDF con `min_df=5` sobre 15 idiomas se queda sin vocabulario fuera
del inglés. ¿Un encoder multilingüe recupera parte de esa brecha sin dejar de usar
Random Forest?

**Protocolo.** Mismo split holdout (9.696 / 2.424), mismo Random Forest y mismos
hiperparámetros. Se codifican `premise` y `hypothesis` **en crudo** (sin lematizar ni
quitar puntuación), normalizados en L2, y se prueban cuatro representaciones:

- `concat`: `[ep, eh]` (768 dims).
- `full`: `[ep, eh, |ep−eh|, ep·eh]` (1.536 dims).
- `diff+product`: solo los dos bloques de interacción (768 dims).
- `concat + cosine`: concat más la similitud coseno como columna (769 dims).

Además: PCA a 200 dims, un diagnóstico con `max_features=0.1` y un híbrido
TF-IDF + POS + relación + embeddings con selección `f_classif` (chi² no admite
valores negativos). Todo se mide global y por lengua.

### Modelo por defecto: `paraphrase-multilingual-MiniLM-L12-v2`

| Variante | Dims | Acc | F1 | Inglés | No inglés | Fit |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF `pair_diff` (referencia) | 300 | 0,5206 | 0,5193 | 0,5790 | 0,4467 | 2 s |
| `concat` | 768 | 0,4769 | 0,4615 | 0,4609 | 0,4972 | 21 s |
| `diff+product` | 768 | 0,5837 | 0,5804 | 0,6108 | 0,5495 | 22 s |
| `concat + cosine` | 769 | 0,5763 | 0,5725 | 0,5835 | 0,5673 | 23 s |
| **`full`** | **1.536** | **0,6052** | **0,6062** | **0,6337** | **0,5692** | **32 s** |
| `full` + `max_features=0.1` | 1.536 | 0,5986 | 0,6001 | 0,6248 | 0,5654 | 132 s |
| PCA 200 | 200 | 0,5268 | 0,5186 | 0,5140 | 0,5430 | 13 s |
| Híbrido TF-IDF + embeddings | 400 | 0,5771 | 0,5780 | 0,5975 | 0,5514 | 28 s |

Dos observaciones importantes:

1. **La interacción es lo que aporta.** Concatenar los dos vectores sin más es *peor*
   que el TF-IDF (0,477): el bosque no puede calcular la relación entre ambas mitades.
   Al añadir `|ep−eh|` y `ep·eh` sube 12,8 pts. Es el mismo principio que `pair_diff`
   en el vectorizador léxico: NLI es una tarea de relación, no de bolsas de palabras.
2. **PCA destruye la señal.** Con 200 dims se queda en 0,527: las direcciones de mayor
   varianza no son las que separan relación lógica.

### Por lengua (MiniLM `full` frente a TF-IDF)

| Lengua | n | TF-IDF | Embeddings | Δ pts |
|---|---:|---:|---:|---:|
| English | 1.354 | 0,579 | 0,634 | +5,5 |
| Chinese | 103 | 0,340 | 0,670 | **+33,0** |
| Russian | 94 | 0,468 | 0,532 | +6,4 |
| Hindi | 86 | 0,500 | 0,547 | +4,7 |
| Spanish | 83 | 0,530 | 0,554 | +2,4 |
| Arabic | 79 | 0,418 | 0,506 | +8,9 |
| Greek | 77 | 0,455 | 0,649 | +19,5 |
| Thai | 77 | 0,351 | 0,584 | +23,4 |
| Swahili | 72 | 0,444 | 0,417 | −2,8 |
| Bulgarian | 71 | 0,549 | 0,507 | −4,2 |
| Urdu | 70 | 0,500 | 0,543 | +4,3 |
| Vietnamese | 67 | 0,567 | 0,672 | +10,4 |
| German | 67 | 0,343 | 0,612 | **+26,9** |
| Turkish | 65 | 0,354 | 0,569 | +21,5 |
| French | 59 | 0,458 | 0,593 | +13,6 |

Mejoran 13 de 15 lenguas; la media ponderada por `n` es +8,5 pts. La brecha
inglés/no inglés se reduce de 13,2 a 6,5 pts.

### Control de contaminación

`paraphrase-multilingual-MiniLM-L12-v2` pertenece a una familia entrenada con datos de
paráfrasis y NLI; como este dataset es XNLI (traducción de MNLI), parte de las frases
puede haber pasado por su entrenamiento. Para separar «mejor representación» de
«memorización» se repite el experimento con un encoder **sin entrenamiento NLI**,
`intfloat/multilingual-e5-small` (prefijo `query: ` en ambos lados):

| Variante | Dims | Acc | F1 | Inglés | No inglés |
|---|---:|---:|---:|---:|---:|
| TF-IDF (referencia) | 300 | 0,5206 | 0,5193 | 0,5790 | 0,4467 |
| E5 `full` | 1.536 | 0,5602 | 0,5589 | 0,5731 | 0,5439 |
| E5 `diff+product` | 768 | 0,5503 | 0,5456 | 0,5731 | 0,5215 |
| E5 híbrido | 400 | 0,5660 | 0,5662 | 0,5753 | 0,5542 |

El control limpio sigue mejorando el global (+4,0 pts) y sobre todo el no inglés
(+9,7 pts), mientras que en inglés se queda igual (0,573 frente a 0,579). Es el patrón
esperado de una mejora de transferencia cross-lingual, no de memorización. Que MiniLM
gane 5,5 pts en inglés y E5 ninguno sugiere que **parte** del extra de MiniLM sí puede
venir de familiaridad con NLI; el uso de MiniLM en la entrega habría que declararlo y,
si es posible, confirmarlo con la leaderboard de Kaggle (el test es otro corte de XNLI).

### Robustez y coste

- **CV agrupada por premisa** (5 folds, sin premisas compartidas): 0,5997 ± 0,0082
  frente a 0,5151 del TF-IDF. La ganancia no depende de ver la misma premisa en train.
- **Coste**: codificar 24.240 frases tarda ~8 min en CPU (una vez, cacheado en
  `artifacts/embeddings`); el fit pasa de 2 s a 32 s y el modelo ocupa 1.536 dims.
- **Híbrido**: añadir TF-IDF a los embeddings y seleccionar 400 features con
  `f_classif` empeora frente a embeddings solos (0,577 frente a 0,605). Con este
  presupuesto de features, la señal densa y la léxica se estorban.

### Conclusión

Los embeddings **sí mejoran** el modelo, y lo hacen donde el TF-IDF falla: las lenguas
no inglesas. La mejora es grande (+8,5 pts de holdout con MiniLM, +4,0 con el control
limpio) y sobrevive a la CV agrupada. Antes de adoptarla hay que decidir dos cosas:
qué encoder se declara (MiniLM es mejor pero tiene riesgo de familiaridad con NLI; E5
es el control conservador) y si se acepta el coste de dependencia y de tiempo. No se ha
cambiado el modelo final: esta sección documenta el experimento, no una sustitución.

Reproducibilidad (requiere `sentence-transformers` instalado; ver versiones al final):

```bash
# MiniLM (por defecto)
0-AiDea/.venv/bin/python 0-AiDea/rf_embeddings.py

# Control sin NLI
EMB_PREFIX="query: " 0-AiDea/.venv/bin/python 0-AiDea/rf_embeddings.py intfloat/multilingual-e5-small
```

Los resultados quedan en `0-AiDea/rf_embeddings_results_<modelo>.json`. El script cachea
los embeddings por modelo en `0-AiDea/artifacts/embeddings` y lee las particiones
preprocesadas de `0-AiDea/artifacts/preprocessed`; no necesita spaCy. Entorno usado:
Python 3.14, `sentence-transformers` 6.3.0, `transformers` 5.15.0, `tokenizers` 0.22.2,
`huggingface-hub` 1.28.0, `torch` 2.15.0 (CPU).

---

## 11. Nota sobre la rúbrica

El enunciado del trabajo pide **comparar al menos dos algoritmos supervisados** distintos
en los mismos ejemplos de validación. Esta rama entrena solo Random Forest por decisión
explícita, así que esa parte de la rúbrica («justified classifier comparison and tuning»,
8 puntos de 30) quedaría sin cubrir si esta es la entrega final. Las mejoras de este
documento son independientes de ese punto y se pueden trasladar a la comparación con los
demás algoritmos.

---

## 12. Implementación (completada)

1. `relation_features.py`: `RelationFeatures` con `get_feature_names_out`, validación de
   columnas y tests unitarios (13 casos: negaciones, longitudes, números, cadenas planas,
   huecos, no negatividad, clonado, ausencia de ajuste y entrada de una sola columna).
2. `training.py`: `PREPROCESSING_PARAMS` con `tokenizer="spacy"`; `VECTORIZER_PARAMS` con
   `min_df=5`, unigramas y `use_pos=False`; nueva constante `SELECT_K_FEATURES = 300`;
   `build_pipeline` añade `SelectKBest(chi2)` y `RelationFeatures`, con
   `select_k=None`/`use_relation_features=False` para ablaciones.
3. Tests: 66 en verde, incluidos los nuevos de selección ajustada solo con train,
   reproducibilidad con selección y contrato del transformador de relación.
4. `random_forest_training.ipynb`: re-ejecutado con la nueva configuración; añade la
   atribución de la selección por bloque y actualiza conclusiones.
5. `TRAINING.md` y `README.md`: métricas nuevas (CV 0,5133; holdout 0,5206/0,5193;
   agrupada 0,5151), distribución de la submission y limitaciones revisadas.
6. Verificación: CV y holdout reproducen exactamente el banco de experimentos
   (0,513303 y 0,520627), `pytest` en verde y submission con 5.195 filas válidas.

Pendiente y fuera de esta rama: comparar con un segundo algoritmo supervisado (la
rúbrica lo pide), decidir si se adopta la representación con embeddings de la sección 10
y actualizar la submission de Kaggle con la configuración elegida.

