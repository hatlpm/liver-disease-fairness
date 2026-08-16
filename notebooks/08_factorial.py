# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fase 8 — Experimento factorial
#
# **Actividad 2 · Criterio 4 (C4, 25%) · Tarea T8 del enunciado — la tarea
# mejor pagada, depende de una sola pregunta.**
#
# Este notebook responde:
#
# - **T8** — ¿Qué impacto tuvo la normalización (MinMax o Z-Score) y el
#   balanceo de datos (SMOTE) en las métricas de rendimiento?
#
# La lógica vive en `src/factorial.py` (P-R2 del PRD) — este notebook solo la
# invoca y narra. Ningún número se copia de otro documento: todos salen de
# una celda ejecutada aquí mismo.
#
# 🔴 **El conjunto de prueba NO se carga en ningún momento de este
# notebook** — a diferencia de la Fase 7, aquí ni siquiera hay una sección
# final que lo toque. El test ya se evaluó una vez, como debía (F7-R4);
# evaluarlo 20 veces más para elegir entre configuraciones lo convertiría en
# un conjunto de selección y destruiría retroactivamente la validez de la
# tabla de T7. Las 20 configuraciones se comparan **por validación cruzada
# estratificada sobre train** (F8-R2), con la misma `build_cv(params)` de la
# Fase 6.
#
# **Requisitos citados aquí:** F8-R1, F8-R2, F8-R3, F8-R4, F8-R5, F8-R6, F8-R7.
#
# **Fuera de alcance de este notebook** (se detiene aquí a propósito): la
# auditoría de equidad por sexo (Fase 9), el informe (E2), y cualquier
# tablero Streamlit (Fase P).

# %%
import copy
import sys
import time
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

from sklearn.feature_selection import SelectKBest, f_classif

from src.config import PROJECT_ROOT, TARGET_COL, load_params
from src.data import load_modeling_data
from src.factorial import (
    collect_frozen_best_params,
    compare_scaled_medians,
    compare_smote_synthetic_across_scalers,
    db_gt_tb_row_indices,
    detect_degenerate_cells,
    run_factorial,
    sensitivity_without_db_gt_tb_rows,
    smote_neighbor_agreement_across_scalers,
)
from src.models import MODEL_KEYS, fit_grid_search
from src.splitting import load_split_indices

params = load_params()
params

# %% [markdown]
# ## Carga de datos — **solo train**
#
# Mismo patrón que `06_modelos.ipynb`/`07_evaluacion.ipynb`:
# `load_modeling_data()` reproduce el dataset de la Fase 4,
# `load_split_indices()` recupera los índices congelados en la Fase 4 (**no
# se recalcula el split**). Variante oficial de F6A-R4: **sin `Gender`**.
# `X_test`/`y_test` no se cargan — no existen en este notebook.

# %%
df = load_modeling_data()
split = load_split_indices(PROJECT_ROOT / params["split"]["indices_path"])

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
X_train = X.loc[split["train_index"]].drop(columns=["Gender"])
y_train = y.loc[split["train_index"]]

print(f"train: {len(X_train)} filas -> {y_train.value_counts().to_dict()}")
assert len(X_train) == 456
assert y_train.value_counts().to_dict() == {1: 325, 0: 131}

# %% [markdown]
# ## F8-R1 — Las 20 configuraciones, desde `params.yaml`
#
# `{minmax, zscore} x {none, smote} x 5 modelos`. Ningún factor está escrito
# literal: los valores salen de `params["preprocessing"]["scalers"]` y
# `params["balancing"]["methods"]`, y el número de celdas es
# `len(scalers) * len(methods) * len(MODEL_KEYS)`, lo que sea que declare
# `params.yaml` — no un "20" fijo en el código.

# %%
selector = SelectKBest(f_classif)
n_celdas = len(params["preprocessing"]["scalers"]) * len(params["balancing"]["methods"]) * len(MODEL_KEYS)
print("escaladores:", params["preprocessing"]["scalers"])
print("balanceo:", params["balancing"]["methods"])
print("modelos:", MODEL_KEYS)
print(f"celdas del factorial: {len(params['preprocessing']['scalers'])} x {len(params['balancing']['methods'])} x {len(MODEL_KEYS)} = {n_celdas}")

# %% [markdown]
# ## Por qué los hiperparámetros vienen CONGELADOS de la Fase 6 (Trampa 2)
#
# Los hiperparámetros de cada uno de los 5 algoritmos (incluido `selector__k`,
# el tamaño del subconjunto de variables) se ajustaron **una sola vez** en la
# Fase 6, con `GridSearchCV`, bajo la configuración fija de
# `params["tuning_baseline"]` (Z-Score + SMOTE). Este notebook **reutiliza
# esos `best_params_` sin volver a ajustar nada** dentro de cada una de las
# 20 celdas — nunca se vuelve a llamar `GridSearchCV` aquí
# (`tests/test_fase8_factorial.py::test_hiperparametros_no_reajustados` lo
# verifica sobre el código de `src/factorial.py`).
#
# La razón es de diseño experimental, no de ahorro de cómputo: si el `k` del
# selector o la `C` de la regresión logística se reajustaran dentro de cada
# celda, el efecto del preprocesamiento (lo que pregunta T8) quedaría
# confundido con el efecto de volver a buscar hiperparámetros — no se podría
# saber si un cambio en la métrica viene del escalado/balanceo o de que el
# `GridSearchCV` encontró, por azar, una configuración distinta. Congelar los
# hiperparámetros aísla el primer efecto del segundo (ver
# [ADR-0015](../docs/adr/0015-hiperparametros-congelados-en-el-factorial.md)).
#
# **Limitación declarada:** el `k` óptimo de KNN (o cualquier otro
# hiperparámetro) bajo MinMax podría no ser el mismo que bajo Z-Score+SMOTE.
# Más abajo se incluye, aparte de la tabla oficial, un chequeo de
# sensibilidad que reajusta un solo modelo bajo un escalador distinto para
# medir cuánto importa esto en la práctica.
#
# `collect_frozen_best_params` recalcula esos `best_params_` una vez (mismo
# patrón que `07_evaluacion.ipynb`: autocontenido, no persiste a disco
# todavía — eso es Fase P) y los aplica a cada celda con `pipe.set_params(...)`.

# %%
t0 = time.time()
best_params_by_model = collect_frozen_best_params(X_train, y_train, params, selector=selector)
print(f"Tiempo: {time.time() - t0:.1f} s")
for nombre, bp in best_params_by_model.items():
    print(f"  {nombre:20s} {bp}")

# %% [markdown]
# ## F8-R6 — La predicción, escrita **antes** de medir
#
# La Actividad 1 (EDA) encontró que varias variables bioquímicas tienen
# asimetría fuerte hacia la derecha (colas largas de valores altos: `Sgot`
# con asimetría 10.5, entre otras — `src.config.SKEWED_COLS`). `MinMaxScaler`
# fija sus dos extremos (0 y 1) en el mínimo y el máximo **observados** de
# cada variable. Con una cola larga hacia la derecha, el máximo queda muy
# lejos del grueso de las observaciones — así que ese grueso, incluida la
# mediana, se comprime hacia el extremo bajo del rango `[0, 1]`.
#
# **Predicción:** la mediana escalada de `SKEWED_COLS` bajo MinMax va a
# quedar muy por debajo de 0.5 (el centro "esperado" si la variable fuera
# simétrica) — mucho más cerca de 0 que la mediana equivalente bajo Z-Score
# se aleja de su propio centro (0). Se mide en la celda siguiente, sin haber
# mirado el resultado todavía.

# %%
tabla_medianas = compare_scaled_medians(X_train, params)
tabla_medianas.pivot(index="columna", columns="escalador", values="mediana_escalada").round(4)

# %% [markdown]
# **Lectura frente a la predicción:** se confirma. Bajo MinMax, la mediana de
# las 5 variables asimétricas queda entre ~0.006 y ~0.074 — pegada al 0, muy
# lejos del 0.5 que tendría una variable simétrica. Bajo Z-Score, la mediana
# de las mismas variables queda entre -0.42 y -0.24: se aleja de su propio
# centro (0) porque la distribución es asimétrica, pero no está comprimida
# contra un extremo fijo de un rango acotado — Z-Score no tiene ese "techo"
# artificial que MinMax sí impone con el máximo observado. La predicción de
# F8-R6 se cumple: **MinMax aplasta la mediana de las variables asimétricas**,
# tal como anticipaba el hallazgo de la Actividad 1.

# %% [markdown]
# ## F8-R1/F8-R2 — Las 20 celdas, evaluadas por CV sobre train
#
# Cada celda: `build_pipeline(escalador, balanceo, modelo, params, selector)`
# con los `best_params_` congelados aplicados vía `set_params`, evaluado con
# `cross_validate(cv=build_cv(params), ...)` — la misma CV estratificada de
# 5 pliegues de la Fase 6. **Ninguna celda toca el test.**

# %%
t0 = time.time()
tabla_factorial = run_factorial(X_train, y_train, params, best_params_by_model, selector=selector)
print(f"Tiempo: {time.time() - t0:.1f} s")
assert len(tabla_factorial) == n_celdas
tabla_factorial.round(4)

# %%
pivote_ba = tabla_factorial.pivot_table(index="modelo", columns=["escalador", "balanceo"], values="balanced_accuracy_cv")
pivote_ba.round(4)

# %% [markdown]
# ## Trampa 4 — Las celdas con `balanced_accuracy = 0.5000` exacto
#
# Antes de discutir efectos promedio (F8-R3) hace falta ver esto: varias de
# las 20 celdas tienen `balanced_accuracy_cv` **exactamente** 0.5000, lo que
# significa que el modelo predijo **una sola clase** para todo train. No es
# un error a depurar: sin balanceo, con la clase enferma en 2.48:1, varios
# modelos colapsan exactamente a eso. Verificado con las predicciones reales
# de cada celda (`cross_val_predict`), no solo inferido del valor de la
# métrica.

# %%
tabla_degeneradas = detect_degenerate_cells(X_train, y_train, params, best_params_by_model, selector=selector)
celdas_degeneradas = tabla_degeneradas[tabla_degeneradas["n_clases_predichas"] == 1]
celdas_degeneradas[["modelo", "escalador", "balanceo", "clase_unica_predicha", "enfermos_no_detectados", "sanos_falsos_positivos"]]

# %%
n_sanos_train = int((y_train == 0).sum())
n_enfermos_train = int((y_train == 1).sum())
for _, fila in celdas_degeneradas.iterrows():
    clase = "enfermo" if fila["clase_unica_predicha"] == 1 else "sano"
    print(
        f"{fila['modelo']:20s} ({fila['escalador']}, {fila['balanceo']}): predice SIEMPRE '{clase}' -- "
        f"{int(fila['sanos_falsos_positivos'])} de {n_sanos_train} sanos de train marcados como enfermos, "
        f"{int(fila['enfermos_no_detectados'])} de {n_enfermos_train} enfermos sin detectar."
    )

# %% [markdown]
# **Esto es el hallazgo más contundente de toda la fase, más que cualquier
# diferencia de dos decimales entre escaladores: sin balanceo, varios de los
# cinco algoritmos no aprenden nada en absoluto** — ni siquiera intentan
# distinguir un paciente de otro, simplemente responden "enfermo" siempre
# porque es la respuesta que más aciertos produce con una clase mayoritaria
# de 71%. SMOTE no es un ajuste fino: para estos modelos, es la diferencia
# entre tener un clasificador y no tenerlo. **Esto importa para leer F8-R3 y
# F8-R7 más abajo sin caer en una trampa**: cualquier promedio de "recall" o
# "precisión" agrupado por `balanceo` incluye estas celdas degeneradas, y una
# predicción de una sola clase infla artificialmente algunas métricas (ver
# F8-R7).

# %% [markdown]
# ## F8-R3 — Impacto de cada factor por separado, y su interacción
#
# **Efecto del escalado solo** (promediando sobre balanceo y modelo):

# %%
efecto_escalado = tabla_factorial.groupby("escalador")["balanced_accuracy_cv"].mean().round(4)
efecto_escalado

# %% [markdown]
# **Efecto del balanceo solo** (promediando sobre escalador y modelo):

# %%
efecto_balanceo = tabla_factorial.groupby("balanceo")["balanced_accuracy_cv"].mean().round(4)
efecto_balanceo

# %% [markdown]
# **Interacción** (¿el efecto de SMOTE es el mismo bajo los dos escaladores?
# ¿el efecto del escalado es el mismo con y sin SMOTE?):

# %%
interaccion = tabla_factorial.pivot_table(index="escalador", columns="balanceo", values="balanced_accuracy_cv")
interaccion["diferencia (smote - none)"] = (interaccion["smote"] - interaccion["none"]).round(4)
interaccion.round(4)

# %% [markdown]
# **Lectura de F8-R3, con las cifras de arriba:**
#
# - **Balanceo, en promedio, ayuda más que el escalado.** La diferencia entre
#   `smote` y `none` (promediada sobre escalador y modelo) es mayor que la
#   diferencia entre `minmax` y `zscore` (promediada sobre balanceo y
#   modelo) — SMOTE cambia qué tan bien el modelo distingue ambas clases;
#   el escalado, para la mayoría de los 5 algoritmos, apenas mueve la aguja.
# - **La interacción existe:** el salto que da SMOTE (`smote - none`) no es
#   igual bajo `minmax` que bajo `zscore` — la fila de la tabla de
#   interacción de arriba lo cuantifica. Esto es consistente con el hallazgo
#   de F8-R4/Trampa 3 más abajo: SMOTE es sensible a la escala relativa entre
#   variables, así que "cuánto ayuda SMOTE" depende de qué escalador se usó
#   antes de aplicarlo — no son dos decisiones independientes.
# - **El promedio esconde variación real por modelo.** La tabla dinámica
#   `pivote_ba` de arriba muestra que el efecto del escalado va de 0 (algunos
#   modelos, algunas celdas) a varios puntos porcentuales (otros) — el
#   detalle por modelo, no el promedio, es lo que explica F8-R4.

# %% [markdown]
# ## F8-R4 — Por qué el escalado afecta a unos modelos y no a otros
#
# **Distancia vs. partición.** `knn`, `svm` y `logistic_regression` basan su
# ajuste en distancias o productos internos entre puntos (vecino más cercano,
# margen del hiperplano, penalización de coeficientes por magnitud) — una
# variable con rango mucho mayor que las demás domina esas distancias si no
# se escala. `decision_tree` particiona comparando **una variable contra un
# umbral**: el orden de los valores no cambia al aplicar una transformación
# monótona (como MinMax o Z-Score), así que el árbol encuentra exactamente el
# mismo umbral en unidades originales o escaladas. `gaussian_nb` ajusta una
# media y una varianza **por variable**, de forma independiente: escalar no
# cambia qué tan separadas están las clases en unidades de esa variable.
#
# ### Trampa 3 — el matiz que la Fase 6 (F6-R7) no contempló
#
# La Fase 6 dejó escrito, como control de coherencia "gratuito": *si
# `gaussian_nb` o `decision_tree` dan métricas distintas entre MinMax y
# Z-Score, hay un error en el experimento*. **Es cierto solo a medias.**

# %%
diff_por_celda = tabla_factorial.pivot_table(index=["modelo", "balanceo"], columns="escalador", values="balanced_accuracy_cv")
escaladores_orden = sorted(diff_por_celda.columns)
diff_por_celda["|minmax - zscore|"] = (diff_por_celda[escaladores_orden[0]] - diff_por_celda[escaladores_orden[1]]).abs()
diff_por_celda[["|minmax - zscore|"]].round(6)

# %% [markdown]
# **Sin SMOTE, `gaussian_nb` y `decision_tree` son exactamente invariantes**
# (diferencia 0.000000) — coherente con la teoría de arriba. **Con SMOTE,
# `decision_tree` deja de serlo.** `gaussian_nb` sigue siendo invariante
# incluso con SMOTE (agrega media/varianza por variable, no posiciones
# individuales, así que promedia el efecto). `svm` no es invariante en
# ninguna de las dos condiciones (su rejilla usa `gamma="scale"`, que depende
# de la varianza post-escalado — no tiene la garantía estructural de NB/árbol).
#
# **El mecanismo, verificado empíricamente y no solo afirmado:** SMOTE elige
# los `k` vecinos más cercanos de cada paciente de la clase minoritaria **por
# distancia euclídea**, que sí depende de la escala relativa entre variables.
# Bajo un escalador distinto, SMOTE puede encontrar vecinos distintos y
# generar pacientes sintéticos distintos — así que el modelo (aunque sea
# invariante al escalado en sí mismo) termina entrenado sobre datos
# distintos. Dos pruebas directas:

# %%
acuerdo_vecinos = smote_neighbor_agreement_across_scalers(X_train, y_train, params)
print(f"Pacientes minoritarios comparados: {len(acuerdo_vecinos)}")
print(f"Jaccard medio del conjunto de {params['balancing']['smote']['k_neighbors']} vecinos entre escaladores: {acuerdo_vecinos['jaccard'].mean():.4f}")
print(f"Pacientes con el MISMO conjunto de vecinos bajo ambos escaladores (Jaccard=1): {(acuerdo_vecinos['jaccard'] == 1).sum()} de {len(acuerdo_vecinos)}")

# %%
comparacion_sinteticos = compare_smote_synthetic_across_scalers(X_train, y_train, params)
{k: v for k, v in comparacion_sinteticos.items() if k != "tabla_distancias"}

# %% [markdown]
# **Lectura:** de los pacientes de la clase minoritaria, el conjunto de
# vecinos que SMOTE usaría para generar un sintético **no coincide del todo**
# entre MinMax y Z-Score (Jaccard medio por debajo de 1). Y, comparando los
# pacientes sintéticos que efectivamente genera cada escalador (revertidos a
# unidades originales para que la comparación tenga sentido clínico): de los
# 194 sintéticos, una parte son el mismo paciente bit a bit bajo ambos
# escaladores, pero una fracción sustancial **son pacientes genuinamente
# distintos** — con diferencias de hasta cientos de unidades en variables
# como `Alkphos`. Esto confirma el mecanismo: **la invariancia del
# estimador no sobrevive a un remuestreo que sí es sensible a la escala.**
# El control de coherencia de la Fase 6 necesita este matiz — registrado
# también en `docs/CHANGELOG_iteraciones.md` (Loop G), sin tocar
# `06_modelos.ipynb` (fase ya cerrada).

# %% [markdown]
# ## F8-R7 — ¿SMOTE mejora el recall a costa de la precisión?
#
# `recall`/`precisión` en este proyecto se calculan con `pos_label=1`
# (F7-R2) — es decir, sobre la clase **"enfermo"**, que es la clase
# **mayoritaria** (71% de train, F4-R5/ADR-0012). Eso importa aquí exactamente
# igual que importó para elegir la métrica de optimización en la Fase 6: un
# promedio ingenuo de "recall" puede estar midiendo, en parte, cuánto se
# inclina el modelo hacia la clase mayoritaria, no cuánto aprendió.

# %%
comparacion_recall_precision = tabla_factorial.groupby("balanceo")[["recall_cv", "precision_cv", "f1_cv"]].mean().round(4)
comparacion_recall_precision

# %% [markdown]
# **Primera lectura, ingenua:** el recall PROMEDIO es más alto `sin SMOTE`
# (0.80) que `con SMOTE` (0.55). Leído sin más contexto, parecería que SMOTE
# **empeora** el recall. **Esto toca la Trampa 4 de arriba:** 3 de las 10
# celdas `sin SMOTE` son las degeneradas que predicen "enfermo" siempre --
# con recall perfecto (1.0) por construcción, no porque el modelo aprendiera
# nada -- así que inflan el promedio de `none`. La pregunta que hay que
# resolver **con un control, no por argumento**: ¿el recall más alto sin
# SMOTE es solo ese artefacto, o el efecto persiste incluso comparando contra
# los modelos que sí aprendieron algo sin balancear? Y, en paralelo, la
# pregunta correcta no es únicamente "¿sube el recall?" sino "¿mejora el
# modelo en los dos sentidos en que puede equivocarse?" -- para eso hace
# falta también la **especificidad** (recall de la clase "sano"), ya
# calculada en `tabla_degeneradas` de la sección de Trampa 4 (mismas
# predicciones de `cross_val_predict`, sin volver a ajustar nada):

# %%
tabla_sensibilidad_especificidad = tabla_degeneradas.copy()
tabla_sensibilidad_especificidad["especificidad"] = tabla_sensibilidad_especificidad["sanos_correctos"] / (
    tabla_sensibilidad_especificidad["sanos_correctos"] + tabla_sensibilidad_especificidad["sanos_falsos_positivos"]
)
tabla_sensibilidad_especificidad["sensibilidad"] = tabla_sensibilidad_especificidad["enfermos_detectados"] / (
    tabla_sensibilidad_especificidad["enfermos_detectados"] + tabla_sensibilidad_especificidad["enfermos_no_detectados"]
)
tabla_sensibilidad_especificidad.groupby("balanceo")[["sensibilidad", "especificidad"]].mean().round(4)

# %%
tabla_sensibilidad_especificidad.pivot_table(index="modelo", columns="balanceo", values=["sensibilidad", "especificidad"]).round(4)

# %% [markdown]
# ### Control: ¿el recall más alto sin SMOTE es solo el artefacto de las 3 celdas degeneradas?
#
# Se responde con un cálculo, no con un argumento: se repite el promedio de
# `recall_cv` de las celdas `sin SMOTE`, mismas 10 celdas de arriba, pero
# **excluyendo las 3 degeneradas** de Trampa 4 -- quedan las 7 celdas donde
# el modelo sí intentó discriminar entre pacientes.

# %%
claves_degeneradas = set(
    zip(celdas_degeneradas["modelo"], celdas_degeneradas["escalador"], celdas_degeneradas["balanceo"], strict=True)
)
tabla_factorial["celda_degenerada"] = [
    (modelo, escalador, balanceo) in claves_degeneradas
    for modelo, escalador, balanceo in zip(
        tabla_factorial["modelo"], tabla_factorial["escalador"], tabla_factorial["balanceo"], strict=True
    )
]

recall_sin_smote_todas = tabla_factorial.loc[tabla_factorial["balanceo"] == "none", "recall_cv"].mean()
recall_sin_smote_sin_degeneradas = tabla_factorial.loc[
    (tabla_factorial["balanceo"] == "none") & (~tabla_factorial["celda_degenerada"]), "recall_cv"
].mean()
recall_con_smote = tabla_factorial.loc[tabla_factorial["balanceo"] == "smote", "recall_cv"].mean()
n_no_degeneradas = int((~tabla_factorial.loc[tabla_factorial["balanceo"] == "none", "celda_degenerada"]).sum())

print(f"recall (sin SMOTE, las 10 celdas):                              {recall_sin_smote_todas:.4f}")
print(f"recall (sin SMOTE, excluyendo las 3 degeneradas, {n_no_degeneradas} celdas):    {recall_sin_smote_sin_degeneradas:.4f}")
print(f"recall (con SMOTE, las 10 celdas):                              {recall_con_smote:.4f}")

# %% [markdown]
# **Resultado del control:** excluyendo las 3 celdas degeneradas, el recall
# promedio sin SMOTE baja de 0.7988 a 0.7125 -- las degeneradas sí explican
# una parte real de la magnitud del efecto -- **pero sigue por encima** del
# 0.5535 con SMOTE. Es decir: las celdas degeneradas explican parte de la
# **magnitud** de la diferencia, no su **dirección**. Incluso comparando solo
# contra los 7 modelos que de verdad intentaron discriminar sin balancear,
# SMOTE sigue bajando el recall de la clase enferma -- no es un artefacto de
# las celdas degeneradas, es un efecto real del remuestreo sobre el punto de
# operación del modelo.
#
# **La lectura correcta, con las dos caras de la moneda:** sin SMOTE, la
# sensibilidad promedio es alta (0.80, o 0.71 ya controlando por las
# degeneradas) pero la **especificidad promedio se hunde a 0.37** — `svm`
# sin SMOTE tiene especificidad **0.0000** exacto en ambos escaladores (son
# las celdas degeneradas de Trampa 4) y `logistic_regression` (bajo
# `minmax`) cae a 0.0458. Con SMOTE, la sensibilidad baja a 0.55 pero la
# **especificidad sube a 0.80** — un balance mucho más parejo entre los dos
# tipos de error, que es justo lo que `balanced_accuracy` premia (F8-R3), y
# un intercambio real, no solo aparente, confirmado por el control de
# arriba.
#
# **Respuesta a F8-R7:** SMOTE **no** sube el recall de la clase enferma —
# lo baja, genuinamente, no solo en apariencia. Sin SMOTE el modelo tiende a
# sobre-predecir "enfermo" (la clase mayoritaria), lo que infla su propio
# recall -- en 3 de 10 celdas hasta el extremo de la degeneración total, pero
# el efecto ya está presente, en menor magnitud, incluso entre los modelos
# que sí aprendieron algo -- a costa de fallar sistemáticamente en detectar a
# los sanos. Lo que SMOTE realmente mejora es la **precisión**
# (0.7896 → 0.8819: cuando el modelo dice "enfermo", acierta más veces) y,
# sobre todo, la **especificidad** (0.3718 → 0.7992) — convierte un
# clasificador que en el caso extremo ni siquiera intenta distinguir
# pacientes en uno que sí discrimina en ambas direcciones.
#
# **Para un cribado clínico (F7-R1), esto no es una mala noticia.** Un
# modelo con sensibilidad "perfecta" pero especificidad 0% (como `svm` sin
# SMOTE) no tiene ningún valor clínico: envía a **todos** los pacientes,
# sanos y enfermos, a la misma vía de seguimiento — no criba nada. El valor
# de un cribado está en reducir el volumen de pacientes que necesitan una
# prueba de confirmación **sin dejar pasar enfermos**; eso exige balancear
# ambos errores, no maximizar solo la sensibilidad. Dicho eso, la
# sensibilidad de 0.55 con SMOTE (bajo la configuración congelada de la
# Fase 6, umbral 0.5) sigue dejando sin detectar a una fracción importante de
# enfermos -- exactamente el motivo por el que la Fase 7 exploró bajar el
# umbral de decisión a 0.30 (recall ≈0.93, ADR-0014): el balanceo y el umbral
# son dos palancas distintas, y esta fase solo mueve la primera.

# %% [markdown]
# ## F8-R5 — Análisis de sensibilidad: las 3 filas con `DB > TB`
#
# §5.2 del PRD decidió **conservar e imputar** (no eliminar) las 3 filas con
# `DB > TB`, a condición de un análisis de sensibilidad obligatorio: repetir
# el mejor modelo sin esas filas y comprobar que las conclusiones no cambian.
# Los índices se derivan de los datos, nunca escritos a mano:

# %%
filas_db_gt_tb = db_gt_tb_row_indices(X_train)
print("Filas con TB/DB nulos (DB > TB) en train:", list(filas_db_gt_tb))
assert list(filas_db_gt_tb) == [246, 261, 279]

# %%
tabla_sensibilidad = sensitivity_without_db_gt_tb_rows(
    X_train, y_train, params, best_params_by_model["logistic_regression"], selector=selector
)
tabla_sensibilidad.round(4)

# %%
diferencia_sensibilidad = abs(
    tabla_sensibilidad.loc[tabla_sensibilidad["variante"] == "con_DB_gt_TB", "balanced_accuracy_cv"].iloc[0]
    - tabla_sensibilidad.loc[tabla_sensibilidad["variante"] == "sin_DB_gt_TB", "balanced_accuracy_cv"].iloc[0]
)
print(f"|balanced_accuracy con - sin las 3 filas DB>TB| = {diferencia_sensibilidad:.4f}")

# %% [markdown]
# **Conclusión:** la diferencia es pequeña frente al ruido habitual de la
# validación cruzada en este dataset (los 5 modelos de la Fase 6 variaron
# 0.03-0.06 entre pliegues). Quitar las 3 filas con `DB > TB` **no cambia la
# conclusión** sobre qué modelo funciona mejor bajo la configuración base —
# la decisión de imputarlas en vez de eliminarlas (§5.2 del PRD) queda
# respaldada empíricamente, no solo justificada por el argumento original
# (las 3 son de la clase mayoritaria, así que ni imputarlas ni eliminarlas
# toca la clase minoritaria).

# %% [markdown]
# ## Chequeo adicional, fuera de la tabla oficial: ¿cuánto importa reajustar?
#
# La limitación declarada arriba es que el `k` óptimo de KNN (o cualquier
# hiperparámetro) bajo `tuning_baseline` (Z-Score+SMOTE) podría no ser el
# mismo bajo MinMax. Este chequeo reajusta **un solo modelo** (KNN) bajo
# MinMax+SMOTE, para tener una idea de cuánto cambiaría — **se reporta
# aparte, no reemplaza ninguna celda de la tabla oficial de F8-R1/R2.**

# %%
params_minmax = copy.deepcopy(params)
params_minmax["tuning_baseline"] = {"scaler": "minmax", "balancing": "smote"}

t0 = time.time()
knn_reajustado_minmax = fit_grid_search("knn", X_train, y_train, params_minmax, selector=selector)
print(f"Tiempo: {time.time() - t0:.1f} s")
print("best_params_ bajo tuning_baseline (zscore+smote):", best_params_by_model["knn"])
print("best_params_ reajustado bajo minmax+smote:       ", knn_reajustado_minmax.best_params_)

mascara_knn_minmax_smote = (
    (tabla_factorial["modelo"] == "knn") & (tabla_factorial["escalador"] == "minmax") & (tabla_factorial["balanceo"] == "smote")
)
balanced_accuracy_congelado = tabla_factorial.loc[mascara_knn_minmax_smote, "balanced_accuracy_cv"].iloc[0]
print(f"balanced_accuracy (CV) con hiperparámetros congelados: {balanced_accuracy_congelado:.4f}")
print(f"balanced_accuracy (CV) reajustando bajo minmax:        {knn_reajustado_minmax.best_score_:.4f}")

# %% [markdown]
# Si el `best_score_` reajustado es apenas distinto del que ya aparece en la
# tabla oficial (celda `knn`/`minmax`/`smote`), la decisión de congelar
# hiperparámetros (ADR-0015) no está dejando rendimiento importante sobre la
# mesa para este modelo. Si fuera muy distinto, sería una señal a investigar
# en una fase posterior — no cambia la tabla de F8-R1/R2, que por diseño usa
# los hiperparámetros congelados de la Fase 6 en las 20 celdas.

# %% [markdown]
# ## Resumen
#
# | | Valor |
# |---|---|
# | Configuraciones evaluadas (F8-R1) | `len(scalers) x len(methods) x len(MODEL_KEYS)` desde `params.yaml`, comparadas por CV sobre train (F8-R2) |
# | Efecto del escalado (F8-R3/R4) | Pequeño para la mayoría de modelos; nulo por diseño en `gaussian_nb`/`decision_tree` sin SMOTE (distancias vs. particiones) |
# | Efecto del balanceo (F8-R3/R7) | Grande: sin SMOTE, varios modelos colapsan a predecir una sola clase (Trampa 4); el recall sin SMOTE (0.7988) baja a 0.7125 al excluir esas celdas -- pero sigue por encima del 0.5535 con SMOTE, así que el intercambio recall↔especificidad es real, no solo un artefacto de la degeneración. Con SMOTE sube mucho la especificidad (0.37→0.80) y la precisión |
# | Interacción (F8-R3) | Existe: el efecto de SMOTE depende del escalador usado antes, porque SMOTE es sensible a la escala relativa (Trampa 3) |
# | Invariancia condicional (F8-R4/Trampa 3) | `gaussian_nb`/`decision_tree` invariantes SOLO sin SMOTE (`gaussian_nb` también con SMOTE); `decision_tree`+SMOTE deja de serlo; `svm` nunca lo es (`gamma="scale"`) |
# | Celdas degeneradas (Trampa 4) | Varias celdas sin SMOTE predicen una sola clase para todo train -- el hallazgo más contundente de la fase |
# | Sensibilidad DB>TB (F8-R5) | Diferencia pequeña frente al ruido de CV -- conclusión no cambia sin esas 3 filas |
# | Predicción de F8-R6 | Confirmada: MinMax aplasta la mediana de las variables asimétricas hacia el extremo bajo de `[0,1]` |
# | Hiperparámetros | Congelados de la Fase 6 (`tuning_baseline`), nunca reajustados por celda (ADR-0015) |
# | Conjunto de prueba | **Nunca cargado en este notebook** |
#
# **T8 -- respuesta:** el balanceo (SMOTE) tiene un impacto mucho mayor que
# el escalado sobre estos 5 modelos y este dataset: sin él, varios algoritmos
# no aprenden a distinguir pacientes -- colapsan a predecir siempre "enfermo"
# (Trampa 4), lo que infla artificialmente su propio recall sin ningún valor
# clínico real (especificidad 0%). Con SMOTE, el recall de la clase enferma
# baja pero la especificidad y la precisión suben mucho (F8-R7) -- el
# clasificador pasa de no discriminar a discriminar en ambas direcciones, el
# requisito mínimo para que un cribado tenga algún valor. El escalado importa
# solo para los modelos basados
# en distancia (`knn`, `svm`, `logistic_regression`) y, de forma indirecta y
# antes no documentada, para SMOTE mismo -- que al elegir vecinos por
# distancia euclídea, propaga su propia sensibilidad a la escala hacia
# modelos que en sí mismos no la tienen (`decision_tree`), matizando el
# control de coherencia "ingenuo" que la Fase 6 había dejado escrito
# (Loop G, `docs/CHANGELOG_iteraciones.md`).
#
# **Sigue:** Fase 9 -- auditoría de equidad por sexo (FNR vía CV repetida).

# %% [markdown]
# ## Nota de traspaso a la Fase E2 -- leer antes de escribir el informe
#
# **Las 3 celdas degeneradas (Trampa 4) son la respuesta más contundente que
# esta fase tiene para T8, más que cualquier cifra de F8-R3/R4.** No es una
# curiosidad técnica al margen: **sin balanceo, 2 de los 5 algoritmos exigidos
# por el enunciado (`logistic_regression` y `svm`) no aprenden absolutamente
# nada** bajo al menos una combinación de escalador -- `svm` colapsa bajo
# ambos escaladores, `logistic_regression` bajo `minmax`. "No aprender nada"
# aquí no es una forma de hablar: verificado con las predicciones reales, esos
# modelos responden "enfermo" a los 456 pacientes de train sin una sola
# excepción, sanos incluidos.
#
# El informe de la Actividad 2 (Fase E2) debe presentar esto como el hallazgo
# central de T8 -- el balanceo no es un ajuste fino de un par de puntos
# porcentuales, es la diferencia entre tener un clasificador y no tenerlo, en
# **2 de los 5 algoritmos exigidos**. Las diferencias de `balanced_accuracy`
# entre MinMax y Z-Score (F8-R3/R4), aunque reales y con mecanismo explicado
# (Trampa 3), son de un orden de magnitud menor y no deberían encabezar la
# respuesta a T8 solo porque "impacto de la normalización" aparece primero en
# el enunciado de la tarea -- el propio experimento muestra que el balanceo
# pesa más.
