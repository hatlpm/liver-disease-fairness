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
# # Fase 5 — Pipeline de preprocesamiento y balanceo
#
# **Actividad 2 · Criterio 1 (C1, 25%) · Tareas T1 y T2 del enunciado.**
#
# Este notebook responde:
#
# - **T1** — ¿Cómo se aseguró que los datos estuvieran limpios y listos
#   para el modelado? (imputación y escalado, dentro de un `Pipeline`)
# - **T2** — ¿Se aplicaron técnicas de balanceo como SMOTE? ¿Proceso e
#   impacto?
#
# La lógica vive en `src/pipelines.py` (`build_pipeline`) — este notebook
# solo la invoca y narra (P-R2 del PRD). Ningún número de este notebook se
# copia de la documentación: todos salen de una celda ejecutada aquí mismo.
# El split train/test **no se recalcula**: se carga tal cual quedó
# congelado en la Fase 4 (F4-R9).

# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from src.config import NUMERIC_COLS, PROJECT_ROOT, TARGET_COL, load_params
from src.data import load_modeling_data
from src.pipelines import build_pipeline
from src.splitting import load_split_indices

params = load_params()
params

# %% [markdown]
# ## Carga de datos y split congelado
#
# `load_modeling_data()` reproduce el dataset de la Fase 4 (dedup,
# reconstrucción de `A/G Ratio`, `TB`/`DB` marcados como nulos, `Gender`/
# `Selector` codificados). `load_split_indices()` recupera los índices de
# train/test exactamente como quedaron persistidos — **si se recalculara el
# split aquí, se rompería la trazabilidad entre fases**.

# %%
df = load_modeling_data()
split = load_split_indices(PROJECT_ROOT / params["split"]["indices_path"])

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
X_train, y_train = X.loc[split["train_index"]], y.loc[split["train_index"]]
X_test, y_test = X.loc[split["test_index"]], y.loc[split["test_index"]]

print(f"train: {len(X_train)} filas -> {y_train.value_counts().to_dict()}")
print(f"test:  {len(X_test)} filas -> {y_test.value_counts().to_dict()}")

assert len(X_train) == 456 and len(X_test) == 114
assert y_train.value_counts().to_dict() == {1: 325, 0: 131}
assert y_test.value_counts().to_dict() == {1: 81, 0: 33}

# %% [markdown]
# ### Dónde están los nulos que este Pipeline tiene que manejar

# %%
null_rows = df[df[["TB", "DB"]].isna().any(axis=1)]
print("Filas con TB/DB nulos:", list(null_rows.index))
print("¿Las tres en train?", set(null_rows.index) <= set(X_train.index))
print("¿Alguna en test?", set(null_rows.index) & set(X_test.index))

# %% [markdown]
# Las 3 filas nulas están las tres en train, ninguna en test — pero el
# `Pipeline` tiene que manejarlas de forma general, no solo para el split
# actual: en validación cruzada (Fase 6, 7, 8) esas filas pueden caer en el
# pliegue de *validación* en vez de en el de entrenamiento, y ahí toca
# comprobar que el ancho de la salida no depende de cuántas de ellas haya en
# cada pliegue concreto — se verifica más abajo.

# %% [markdown]
# ## T1 — Imputación y escalado dentro del `Pipeline`
#
# ### Por qué `imblearn.pipeline.Pipeline` y no el de `sklearn` (F5-R1)
#
# F5-R1 exige imputación, escalado y balanceo dentro de un único `Pipeline`.
# El `Pipeline` de `sklearn` no sabe qué hacer con un paso de remuestreo:
# durante `predict` le pediría a `SMOTE` que "transforme" el conjunto de
# prueba, lo cual no tiene sentido (SMOTE no predice, sintetiza filas de
# entrenamiento). El `Pipeline` de `imblearn` aplica los pasos de
# remuestreo **solo** durante `fit` (vía `fit_resample`) y los salta en
# cualquier otra llamada — es la propiedad que garantiza que SMOTE nunca
# toque el test ni los pliegues de validación (F5-R5, verificado más abajo).

# %%
ejemplo = build_pipeline("zscore", True, DummyClassifier(strategy="most_frequent"), params)
print("Tipo:", type(ejemplo).__module__ + "." + type(ejemplo).__name__)
print("Pasos:", [nombre for nombre, _ in ejemplo.steps])

# %% [markdown]
# ### Por qué el indicador de nulos NO usa `SimpleImputer(add_indicator=True)` (F5-R2, F5-R3)
#
# F5-R2 exige imputar `TB`/`DB` por mediana con un indicador de qué filas se
# imputaron; F5-R3 exige que ese indicador **no se escale** (una columna
# binaria escalada deja de significar "imputado sí/no" para convertirse en
# un número sin sentido). Ambos se cumplen aquí: el indicador vive en una
# rama del `ColumnTransformer` separada de la que contiene el escalador
# (ver la salida de la celda siguiente: sus columnas quedan en `{0,1}`,
# nunca tocadas por `StandardScaler`/`MinMaxScaler`). `params.yaml` declara
# `imputer: {strategy: "median", add_indicator: true}`,
# pero `add_indicator=True` de `SimpleImputer` construye el indicador con
# `features="missing-only"`: solo genera una columna para una variable si
# **ese pliegue concreto** que se le pasa a `.fit()` tiene algún nulo en
# ella. Con los 456 de train y `StratifiedKFold(5)` de la Fase 6, cada
# pliegue de entrenamiento se queda con 2-3 de las 3 filas nulas, así que
# hoy no se rompe — pero la Fase 9 usa `RepeatedStratifiedKFold(5,
# n_repeats=10)` (50 ajustes), y ahí la probabilidad de que algún pliegue se
# quede sin ninguna fila nula deja de ser despreciable. Si eso pasa, el
# número de columnas de salida **cambiaría entre pliegues sin avisar**,
# rompiendo cualquier paso posterior que espere un ancho fijo.
#
# `build_pipeline()` (`src/pipelines.py`) usa en su lugar una rama
# independiente del `ColumnTransformer` con `MissingIndicator(features="all")`,
# que siempre genera 2 columnas (`TB`, `DB`), tenga o no nulos el pliegue.
# Detalle completo en [ADR-0007](../docs/adr/0007-tratamiento-de-nulos.md) y
# [ADR-0011](../docs/adr/0011-indicador-nulos-columntransformer-dos-ramas.md).

# %%
pre = ejemplo.named_steps["preprocess"]
X_t = pre.fit_transform(X_train)
print("Columnas de salida:", list(X_t.columns))
print("Forma:", X_t.shape)
print(X_t.dtypes)

# %% [markdown]
# ### Demostración en vivo: ancho estable entre pliegues
#
# Se repite, con los datos reales de train, la misma comprobación que hace
# `tests/test_fase5_pipeline.py::test_ancho_estable_entre_pliegues`: el
# `ColumnTransformer` se ajusta por separado en cada uno de los 5 pliegues
# de entrenamiento de un `StratifiedKFold`, y además se fuerza el escenario
# límite (ningún nulo en el pliegue) quitando a mano las 3 filas nulas.

# %%
skf = StratifiedKFold(
    n_splits=params["cv"]["tuning"]["n_splits"],
    shuffle=params["cv"]["tuning"]["shuffle"],
    random_state=params["seed"],
)
anchos = []
nulos_por_pliegue = []
for train_fold_idx, _ in skf.split(X_train, y_train):
    X_fold = X_train.iloc[train_fold_idx]
    nulos_por_pliegue.append(int(X_fold[["TB", "DB"]].isna().any(axis=1).sum()))
    anchos.append(clone(pre).fit_transform(X_fold).shape[1])

print("Nulos por pliegue de entrenamiento:", nulos_por_pliegue)
print("Ancho de salida por pliegue:", anchos)
assert len(set(anchos)) == 1, "el ancho debería ser idéntico en los 5 pliegues"

ancho_sin_ningun_nulo = clone(pre).fit_transform(X_train.drop(index=list(null_rows.index))).shape[1]
print("Ancho SIN ninguna fila nula (escenario límite de la Fase 9):", ancho_sin_ningun_nulo)
assert ancho_sin_ningun_nulo == anchos[0]

# %% [markdown]
# Con los 5 pliegues actuales, ninguno se queda sin nulos (columna de la
# izquierda arriba), así que hoy el riesgo no se manifiesta — pero el ancho
# tampoco cambia en el escenario forzado sin ningún nulo, que es exactamente
# el caso que puede ocurrir en la Fase 9. `MissingIndicator(features="all")`
# cumple lo que promete.

# %% [markdown]
# ### Demostración en vivo: sin fuga de escalado
#
# El escalador se ajusta **solo** con train. Se compara contra las
# estadísticas calculadas directamente sobre `X_train` (con la mediana de
# train para los 3 nulos, la misma que usa el imputador dentro del
# pipeline) — nunca contra el dataset completo.

# %%
pre_fit = clone(pre)
pre_fit.fit(X_train)
scaler_ajustado = pre_fit.named_transformers_["numeric"].named_steps["scaler"]
referencia = X_train[NUMERIC_COLS].fillna(X_train[NUMERIC_COLS].median())

print("mean_ del escalador (train):", np.round(scaler_ajustado.mean_, 4))
print("mean() calculada aparte (train):", np.round(referencia.mean().to_numpy(), 4))
assert np.allclose(scaler_ajustado.mean_, referencia.mean().to_numpy())
assert np.allclose(scaler_ajustado.scale_, referencia.std(ddof=0).to_numpy())
print("Coinciden exactamente: el escalador nunca vio el test.")

# %% [markdown]
# ### `build_pipeline()` construye las 4 combinaciones que necesitará la Fase 8

# %%
for scaler_nombre in params["preprocessing"]["scalers"]:
    for con_smote in (False, True):
        p = build_pipeline(scaler_nombre, con_smote, DummyClassifier(strategy="most_frequent"), params)
        print(f"scaler={scaler_nombre:7s} smote={con_smote!s:5s} -> pasos={[n for n, _ in p.steps]}")

# %% [markdown]
# ## T2 — SMOTE: proceso e impacto
#
# ### Por qué el orden es imputar → escalar → SMOTE, nunca al revés
#
# SMOTE construye pacientes sintéticos interpolando entre un caso de la
# clase minoritaria y uno de sus `k` vecinos más cercanos, por distancia
# euclídea. Si se aplicara **antes** de escalar, las variables con
# asimetría alta dominarían esa distancia: `Sgot` tiene una asimetría de
# 10.5 (documentada en `02_eda.ipynb`, Actividad 1), y en el clustering
# jerárquico euclídeo de la Fase 2c
# (`02c_eda_clustering.ipynb`) esa misma variable sin transformar bastó
# para que el paciente más extremo quedara a ~17 desviaciones estándar y
# dominara por completo la distancia — el algoritmo terminaba agrupando
# "quién es outlier en `Sgot`", no similitud clínica real. SMOTE usa
# exactamente la misma noción de vecino-más-cercano euclídeo: correrlo
# sobre variables sin escalar repetiría el mismo sesgo, generando vecinos
# (y por tanto pacientes sintéticos) elegidos casi solo por su valor de
# `Sgot`. Por eso el escalador va **dentro** de la rama `"numeric"` del
# `ColumnTransformer`, y `SMOTE` va después de `"preprocess"` en el
# `Pipeline` exterior — nunca antes.

# %% [markdown]
# ### F5-R4 — proceso e impacto de SMOTE sobre tamaño y composición
#
# Se ajusta el pipeline con SMOTE activado y se mide, no se estima, cuántas
# filas sintéticas genera y cómo cambia la composición de clases.

# %%
pipe_smote = build_pipeline("zscore", True, DummyClassifier(strategy="most_frequent"), params)
pipe_smote.fit(X_train, y_train)

Xt_train = pipe_smote.named_steps["preprocess"].transform(X_train)
X_res, y_res = pipe_smote.named_steps["smote"].fit_resample(Xt_train, y_train)

antes = y_train.value_counts().to_dict()
despues = y_res.value_counts().to_dict()
sinteticos = len(y_res) - len(y_train)
print(f"Antes de SMOTE:   {len(y_train)} filas -> {antes}")
print(f"Después de SMOTE: {len(y_res)} filas -> {despues}")
print(f"Sintéticos generados: {sinteticos}")

assert despues == {0: 325, 1: 325}
assert len(y_res) == 650
assert sinteticos == 194

# %% [markdown]
# **F5-R5 — SMOTE nunca toca el test.** Ya se verificó arriba
# (`pipe_smote.predict(X_test)` más adelante), y lo comprueba también
# `tests/test_fase5_pipeline.py::test_smote_no_toca_el_test`: el tamaño del
# test es siempre 114, se use o no SMOTE, porque el `Pipeline` de `imblearn`
# solo remuestrea durante `fit`.

# %%
preds_test = pipe_smote.predict(X_test)
print(f"test tras predict: {len(preds_test)} filas (debe seguir en 114)")
assert len(preds_test) == len(X_test) == 114

# %% [markdown]
# ### F5-R6 — SMOTE frente a submuestreo y a `class_weight='balanced'`
#
# No se implementan las tres alternativas en esta fase (elegir el algoritmo
# de balanceo definitivo no es una comparación de T2, y comparar
# rendimiento de modelos es la Fase 6 en adelante) — se discuten sus
# compromisos:
#
# | Técnica | Qué hace | Ventaja | Desventaja |
# |---|---|---|---|
# | **SMOTE** (usado aquí) | Genera pacientes sintéticos interpolando entre vecinos de la clase minoritaria | No descarta ninguna fila real; el modelo ve más ejemplos minoritarios | Interpola en el espacio de *features*: puede generar combinaciones de valores que ningún paciente real tuvo (más grave en variables muy asimétricas o categóricas — ver Trampa 4 abajo) |
# | **Submuestreo aleatorio** | Descarta filas de la clase mayoritaria hasta igualar tamaños | Simple, no inventa datos | Con 131 sanos de train, igualar a la clase mayoritaria (325) obligaría a **descartar 194 pacientes enfermos reales** — casi tantos como los que SMOTE sintetiza, pero perdiendo información real en vez de añadir información interpolada |
# | `class_weight='balanced'` | No toca los datos; pondera el error de la clase minoritaria más en la función de costo | Cero riesgo de generar pacientes irreales, cero filas descartadas | No todos los algoritmos lo soportan igual (KNN no tiene noción de "peso" nativa); no ayuda a métodos basados en distancia (KNN, SVM) tanto como a los basados en la función de pérdida (regresión logística, árboles) |
#
# La elección de SMOTE ya viene fijada en `params.yaml` desde la Fase 4
# (`balancing.methods: ["none", "smote"]`) — es la técnica que la Fase 8
# comparará contra "sin balanceo" en las 20 configuraciones del experimento
# factorial. Esta tabla dimensiona el costo de oportunidad frente a las
# otras dos, no reabre la decisión.

# %% [markdown]
# ### F5-R7 — SMOTE interpola; con variables muy asimétricas, puede generar pacientes en zonas poco pobladas
#
# El propio mecanismo de interpolación —crear un punto a mitad de camino
# entre dos vecinos reales— asume que el espacio "entre" dos pacientes
# parecidos también es clínicamente plausible. Es una asunción razonable
# para variables con distribución compacta, pero más frágil cuanto más
# asimétrica es la variable: en `Sgot` (asimetría 10.5), la mayoría de los
# pacientes está muy concentrada cerca de valores bajos y unos pocos casos
# extremos se alejan mucho — un vecino "cercano" en el espacio ya escalado
# todavía puede estar lejos en términos clínicos reales para esa variable
# específica, y el punto sintético que cae entre ambos puede no corresponder
# a ningún perfil de paciente real observado. No es un defecto del pipeline:
# es una propiedad conocida de SMOTE, y el motivo por el que conviene no
# tratar los pacientes sintéticos como si fueran datos medidos.

# %% [markdown]
# ### F5-R8 — ¿SMOTE altera la proporción de sexos del train?
#
# SMOTE equilibra la **clase** (`Selector`), no el sexo. Pero como
# sintetiza nuevas filas interpolando sobre *todas* las columnas
# disponibles —incluida `Gender`, que viaja en el pipeline como una columna
# numérica 0/1 más—, el sexo de los pacientes sintéticos no está bajo
# ningún control explícito. Se mide, no se asume.
#
# **La comparación correcta no es train-completo-antes vs.
# train-completo-después.** Ese contraste diluye cualquier efecto entre las
# 456 filas de train, de las cuales 456 (el 100%) sobreviven sin tocar y
# solo se le suman 194 sintéticas — un sesgo real en esas 194 filas apenas
# mueve el promedio de las 650. La comparación que aísla el efecto es
# **la clase minoritaria original (a la que SMOTE dice replicar) frente a
# las filas que SMOTE efectivamente sintetiza a partir de ella**.

# %%
prop_mujeres_train_antes = (X_train["Gender"] == 1).mean()
gender_col = next(c for c in X_res.columns if c.startswith("remainder__Gender"))
prop_mujeres_train_despues = (X_res[gender_col] == 1).mean()
print("Comparación diluida (train completo, NO es la correcta):")
print(f"  antes:   {prop_mujeres_train_antes:.4%}")
print(f"  después: {prop_mujeres_train_despues:.4%}")
print(f"  diferencia: {(prop_mujeres_train_despues - prop_mujeres_train_antes) * 100:+.2f} pp -- parece casi nula, pero es un artefacto de la dilución")

minoria_train = X_train.loc[y_train == 0]
sinteticas = X_res.iloc[len(X_train) :]

prop_mujeres_minoria = (minoria_train["Gender"] == 1).mean()
prop_mujeres_sinteticas = (sinteticas[gender_col] == 1).mean()
print("\nComparación correcta (minoría replicada vs. sintéticos que la replican):")
print(f"  minoría de train (sanos): {int((minoria_train['Gender']==1).sum())}/{len(minoria_train)} = {prop_mujeres_minoria:.4%}")
print(f"  los {len(sinteticas)} sintéticos:      {int((sinteticas[gender_col]==1).sum())}/{len(sinteticas)} = {prop_mujeres_sinteticas:.4%}")
brecha_total = (prop_mujeres_sinteticas - prop_mujeres_minoria) * 100  # destino - origen
print(f"  brecha: {brecha_total:+.2f} pp")

assert round(prop_mujeres_minoria, 4) == 0.2977
assert round(prop_mujeres_sinteticas, 4) == 0.2371

# %% [markdown]
# ### Trampa 4, cuantificada — por qué la brecha de F5-R8 ocurre
#
# SMOTE interpola: `x_nuevo = x_i + λ·(x_j - x_i)`. Sobre una columna
# binaria eso puede producir valores fraccionarios (0.37, 0.82…), que no
# significan nada como sexo de un paciente. Se mide en tres capas: el
# indicador de `TB`/`DB` primero, después las dos causas de la brecha de
# `Gender` medida arriba.
#
# **Capa 1 — el indicador `TB`/`DB` nunca puede fraccionarse, y es
# demostrable, no solo observable:** las 3 filas con el indicador activo
# son las tres `Selector=1` (enfermo, la clase MAYORITARIA — ver
# `docs/adr/0007-tratamiento-de-nulos.md`). SMOTE con `sampling_strategy='auto'`
# **solo** sintetiza la clase minoritaria (`Selector=0`, sanos), buscando
# vecinos dentro de esa misma clase. Como consecuencia, dentro del
# subconjunto de train que SMOTE efectivamente interpola, el indicador es
# **constante** (0 en las dos filas de cualquier par vecino), y
# `x_i + λ·(0 - 0) = 0` siempre, para cualquier `λ`. No es una casualidad de
# esta ejecución: es una consecuencia estructural de que las 3 filas nulas
# caen del lado que SMOTE nunca toca.

# %%
print("Nulos TB/DB dentro de la clase minoritaria (sanos) de train:")
print(minoria_train[["TB", "DB"]].isna().sum())
assert minoria_train[["TB", "DB"]].isna().sum().sum() == 0

indicador_cols = [c for c in X_res.columns if c.startswith("tb_db_indicator__")]
valores_indicador = np.unique(X_res[indicador_cols].to_numpy())
print("Valores únicos del indicador tras SMOTE:", valores_indicador)
assert set(valores_indicador).issubset({0, 1})

# %% [markdown]
# **Capa 2 — `Gender` sí se fracciona en la interpolación cruda, y el
# pipeline lo resuelve de una forma que hay que declarar explícitamente.**
# `build_pipeline()` usa `ColumnTransformer.set_output(transform="pandas")`
# para que el resto del notebook (y los tests) puedan referirse a columnas
# por nombre. Eso tiene una consecuencia no obvia: `imbalanced-learn`
# preserva el tipo de dato original de cada columna (`Gender` es `int64`) al
# reconstruir el DataFrame tras `fit_resample`, así que **cualquier valor
# fraccionario de `Gender` se convierte a entero antes de que se pueda ver**.
# Para saber qué está pasando de verdad, se repite la misma interpolación
# sobre un array de NumPy (sin reconstrucción a `DataFrame`), que conserva
# los valores fraccionarios crudos.

# %%
pre_array = clone(pre)
pre_array.set_output(transform="default")  # deshace el output "pandas" solo para esta comparación
Xt_array = pre_array.fit_transform(X_train)

# Misma configuración de SMOTE que usa el pipeline real (misma semilla,
# mismo k_neighbors) -- clonada del paso ya usado arriba, no reimportada.
sm_diagnostico = clone(pipe_smote.named_steps["smote"])
Xr_array, yr_array = sm_diagnostico.fit_resample(Xt_array, y_train)

gender_pos = list(pre_array.get_feature_names_out()).index(gender_col)
gender_sintetico_crudo = Xr_array[len(Xt_array) :, gender_pos]
n_fraccionarios = int(((gender_sintetico_crudo % 1) != 0).sum())
print(f"De {len(gender_sintetico_crudo)} filas sintéticas, {n_fraccionarios} tienen Gender fraccionario")
print("Ejemplos:", np.round(gender_sintetico_crudo[gender_sintetico_crudo % 1 != 0][:8], 3))

redondeado_correctamente = np.round(gender_sintetico_crudo).astype(int)
truncado_hacia_cero = gender_sintetico_crudo.astype(int)  # lo que hace imbalanced-learn al preservar dtype int64
prop_redondeada = redondeado_correctamente.mean()
print("\nSi se redondeara al más cercano:", pd.Series(redondeado_correctamente).value_counts().to_dict(), f"({prop_redondeada:.4%} mujeres)")
print("Truncado hacia 0 (lo que realmente hace la librería):", pd.Series(truncado_hacia_cero).value_counts().to_dict())
print("Lo que efectivamente entrega el pipeline (X_res):", sinteticas[gender_col].value_counts().to_dict())

assert np.array_equal(truncado_hacia_cero, sinteticas[gender_col].to_numpy())
assert round(prop_redondeada, 4) == 0.2680

# %% [markdown]
# **Capa 3 — descomposición de la brecha de F5-R8 en sus dos causas.** La
# brecha total medida arriba (minoría 29.77% mujeres → sintéticos 23.71%,
# **-6.06 pp**) no viene de una sola fuente: se descompone en el
# truncamiento de la Capa 2 (ya medido) más un segundo efecto que **no**
# depende de ningún error de tipo de dato.

# %%
# Convención de signo: brecha = destino - origen. Negativo = disminución.
gap_total = (prop_mujeres_sinteticas - prop_mujeres_minoria) * 100
gap_truncamiento = (prop_mujeres_sinteticas - prop_redondeada) * 100
gap_geometria = (prop_redondeada - prop_mujeres_minoria) * 100

print(f"Brecha TOTAL (minoría {prop_mujeres_minoria:.2%} -> sintéticos reales {prop_mujeres_sinteticas:.2%}): {gap_total:+.2f} pp")
print(f"  Componente 1 -- truncamiento int64 (redondeado {prop_redondeada:.2%} -> real {prop_mujeres_sinteticas:.2%}): {gap_truncamiento:+.2f} pp")
print(f"  Componente 2 -- geometría de la interpolación (minoría {prop_mujeres_minoria:.2%} -> redondeado {prop_redondeada:.2%}): {gap_geometria:+.2f} pp")
print(f"  Suma de componentes: {gap_truncamiento + gap_geometria:+.2f} pp (debe igualar la brecha total)")

assert round(gap_total, 2) == -6.06
assert round(gap_truncamiento, 2) == -3.09
assert round(gap_geometria, 2) == -2.97

# %% [markdown]
# **El componente 2 es el más interesante, porque no lo causa ningún error
# de implementación.** Aun si `imbalanced-learn` redondeara `Gender` al
# valor más cercano en vez de truncarlo (eliminando el componente 1 por
# completo), los sintéticos seguirían teniendo **2.97 puntos porcentuales
# menos mujeres** que la clase minoritaria que dicen replicar. La causa es
# geométrica: SMOTE elige, para cada paciente minoritario, sus `k=5` vecinos
# más cercanos por distancia euclídea sobre las **9 variables numéricas
# escaladas + el indicador** — el sexo no participa en ese criterio de
# cercanía más que como una dimensión más entre otras 11. No hay ninguna
# garantía de que los vecinos clínicamente más parecidos a una paciente
# concreta sean mayoritariamente mujeres: si por azar (o por cómo se
# distribuyen las demás variables) las mujeres de la clase minoritaria
# quedan con vecinos más cercanos del lado masculino más a menudo que al
# revés, el `Gender` interpolado tiende hacia 0 con más frecuencia de la
# que su peso real (29.77%) justificaría.
#
# **Conclusión de F5-R8, reescrita:** SMOTE genera pacientes sintéticos que
# **subrepresentan a las mujeres respecto del grupo que dice replicar**, sin
# que ninguna decisión explícita del pipeline lo pida — la brecha aparece
# por la propia mecánica de la interpolación, en dos capas independientes
# (geometría de vecinos + un artefacto de tipo de dato encima). Es
# exactamente la tesis central de este proyecto —que un procesamiento
# aparentemente neutro puede introducir sesgo por sexo sin que nadie lo
# programe— ocurriendo dentro de nuestro propio pipeline, no solo en el
# dataset original. **Queda anotado para que la Fase 9 lo retome**: al
# comparar la brecha de FNR por sexo con y sin SMOTE (`F9-R4`), esta
# subrepresentación de mujeres en los sintéticos es un mecanismo candidato
# a considerar si la brecha empeora con balanceo activado.
#
# **Decisión explícita sobre el componente 1 (truncamiento): se acepta y se
# documenta, no se cambia el algoritmo de balanceo en esta fase.** Razones:
#
# 1. Es la parte menor de la brecha total (3.09 de 6.06 pp) y ya queda
#    cuantificada arriba — no es un hallazgo oculto.
# 2. La alternativa correcta desde la teoría (`SMOTENC`, que trata columnas
#    marcadas como categóricas usando la moda de los vecinos en vez de
#    interpolación) es un cambio de **algoritmo** de balanceo — una decisión
#    de mayor alcance que le corresponde a la Fase 8 (el experimento
#    factorial completo, 25% de la nota), no a la Fase 5 de forma unilateral.
#    `SMOTENC` tampoco eliminaría el componente 2 (geométrico): trataría
#    `Gender` categóricamente en vez de interpolarlo, pero seguiría
#    heredando la moda de los mismos `k` vecinos, que ya está sesgada hacia
#    hombres por la misma razón geométrica.
# 3. La Fase 6A todavía no decidió si `Gender` entra como variable del
#    modelo (`F6A-R4`). Si la variante sin `Gender` resulta ser la elegida,
#    esta distorsión deja de tener efecto alguno sobre el modelo final —
#    pero el componente 2 seguiría afectando indirectamente a cualquier otra
#    variable correlacionada con el sexo, así que el hallazgo sigue siendo
#    relevante más allá de si `Gender` es o no una *feature* explícita.
#
# Ambos componentes quedan como antecedente cuantificado para la Fase 8/9,
# no resueltos aquí.

# %% [markdown]
# ## Resumen
#
# | | Valor |
# |---|---|
# | `Pipeline` | `imblearn.pipeline.Pipeline` (nunca `sklearn.pipeline.Pipeline`) |
# | Orden de pasos | imputador (mediana) → escalador → indicador (sin escalar) → [SMOTE] → estimador |
# | Ancho de salida | 12 columnas, estable entre pliegues (verificado con y sin las 3 filas nulas) |
# | Fuga de escalado | Ninguna: estadísticas del escalador == estadísticas de train |
# | SMOTE — train | 456 (325/131) → 650 (325/325), 194 sintéticos |
# | SMOTE — test | 114 antes y después (nunca cambia) |
# | F5-R8 — mujeres, minoría de train | 29.77% (39/131) |
# | F5-R8 — mujeres, sintéticos de SMOTE | 23.71% (46/194) — **brecha -6.06 pp** |
# | Trampa 4 — indicador TB/DB | 0 fraccionarios, estructuralmente garantizado |
# | Trampa 4 — `Gender`, componente 1 (truncamiento int64) | -3.09 pp de la brecha |
# | Trampa 4 — `Gender`, componente 2 (geometría de vecinos) | -2.97 pp de la brecha, no depende de ningún error de tipo de dato |
#
# **Hallazgo central de esta fase:** SMOTE subrepresenta a las mujeres entre
# los pacientes sintéticos frente a la clase que dice replicar, por dos
# mecanismos independientes — aceptado y documentado, anotado para que la
# Fase 9 lo retome al comparar la brecha de FNR por sexo con y sin SMOTE.
#
# **Sigue:** Fase 6A — selección de variables (decisión sobre `Gender` como
# *feature*) y Fase 6 — los 5 algoritmos con Grid Search, reutilizando
# `build_pipeline()` sin cambios.
