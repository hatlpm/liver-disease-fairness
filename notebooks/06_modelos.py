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
# # Fase 6A + Fase 6 — Selección de variables y algoritmos
#
# **Actividad 2 · Criterio 2 (C2, 25%) · Tareas T4 y T5 del enunciado ·
# Fase 6A marcada `[V]`.**
#
# Este notebook responde:
#
# - **T4** — ¿Qué algoritmos se seleccionaron (regresión logística, KNN,
#   Naive Bayes, árboles de decisión, SVM) y cómo se configuró cada uno?
# - **T5** — ¿Cómo se encontraron los mejores hiperparámetros? Grid Search o
#   Random Search.
#
# Van juntas la Fase 6A (selección de variables) y la Fase 6 (algoritmos)
# porque el selector es un **paso más del `Pipeline`** que la Fase 6 ajusta:
# su `k` se optimiza junto a los hiperparámetros de cada modelo, dentro de
# la misma validación cruzada (F6A-R2). La lógica vive en `src/pipelines.py`
# y `src/models.py` — este notebook solo la invoca y narra (P-R2 del PRD).
# Ningún número se copia de otro documento: todos salen de una celda
# ejecutada aquí mismo. El split train/test **no se recalcula** (F4-R9), y
# **el conjunto de prueba no se carga en ningún momento de este notebook** —
# se toca una sola vez, en la Fase 7 (F7-R4).
#
# **Requisitos citados aquí:** F6A-R1, F6A-R2, F6A-R3, F6A-R4, F6A-R5 (Fase
# 6A) · F6-R1, F6-R2, F6-R3, F6-R4, F6-R5, F6-R6, F6-R7, F6-R8 (Fase 6).
#
# **Fuera de alcance de este notebook** (se detiene aquí a propósito):
# evaluar sobre el test (Fase 7), el experimento factorial de 20
# configuraciones {MinMax, Z-Score} x {con/sin SMOTE} (Fase 8), la
# auditoría de equidad por sexo (Fase 9), y crear cualquier variable nueva
# (*feature engineering*, fuera de alcance del proyecto completo — §2.2 del
# PRD).

# %%
import sys
import time
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

from src.config import NUMERIC_COLS, PROJECT_ROOT, TARGET_COL, load_params
from src.data import load_modeling_data
from src.models import (
    MODEL_KEYS,
    build_cv,
    build_search_grid,
    fit_grid_search,
    null_classifier_floor,
)
from src.pipelines import build_pipeline
from src.splitting import load_split_indices

params = load_params()
params

# %% [markdown]
# ## Carga de datos — **solo train**
#
# `load_modeling_data()` reproduce el dataset de la Fase 4.
# `load_split_indices()` recupera los índices congelados en
# `data/processed/split_indices.json`. A diferencia de `05_pipeline.ipynb`,
# este notebook **no carga `X_test`/`y_test` en ningún momento** — no hace
# falta para nada de lo que exige la Fase 6A/6, y así queda estructuralmente
# imposible tocar el test por accidente (lo mismo que verifica
# `tests/test_fase6_modelos.py::test_test_no_tocado` sobre el código de
# `src/`).

# %%
df = load_modeling_data()
split = load_split_indices(PROJECT_ROOT / params["split"]["indices_path"])

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
X_train, y_train = X.loc[split["train_index"]], y.loc[split["train_index"]]

print(f"train: {len(X_train)} filas -> {y_train.value_counts().to_dict()}")
assert len(X_train) == 456
assert y_train.value_counts().to_dict() == {1: 325, 0: 131}

# %% [markdown]
# ## El suelo del clasificador nulo — el punto de comparación obligatorio de toda la fase
#
# Antes de ajustar ningún modelo, hay que fijar contra qué se compara.
# **Un clasificador nulo es el que no mira ningún dato de entrada**: aquí,
# el que responde "enfermo" para cualquier paciente, siempre. Cualquier
# modelo real tiene que superarlo para poder decir que aprendió algo de los
# datos.
#
# `Selector = 1` (enfermo, F4-R5) es la clase **mayoritaria** de este
# dataset: 71.27% de train. Eso importa porque **F1** — la métrica que
# `params.yaml` declaraba como criterio de ajuste desde la Fase 4 — se
# recomienda habitualmente para clases desbalanceadas, pero ese consejo
# asume que la clase positiva es la minoritaria. Aquí es al revés.

# %%
suelo_train = null_classifier_floor(y_train)
tabla_suelo = pd.DataFrame(
    {
        "métrica": ["accuracy", "recall", "precision", "F1", "balanced_accuracy"],
        "suelo nulo (train, calculado aquí)": [
            suelo_train["accuracy"],
            suelo_train["recall"],
            suelo_train["precision"],
            suelo_train["f1"],
            suelo_train["balanced_accuracy"],
        ],
        "suelo nulo (test, cifra de referencia — NO recalculada aquí)": [0.7105, 1.0000, 0.7105, 0.8308, 0.5000],
    }
)
tabla_suelo

# %%
assert round(suelo_train["accuracy"], 4) == 0.7127
assert suelo_train["recall"] == 1.0
assert round(suelo_train["precision"], 4) == 0.7127
assert round(suelo_train["f1"], 4) == 0.8323
assert suelo_train["balanced_accuracy"] == 0.5

# %% [markdown]
# **Por qué la columna de test es una cifra citada y no calculada aquí:**
# el test se congeló en la Fase 4 y la regla del proyecto es tocarlo una
# sola vez, en la Fase 7 (F7-R4) — ni siquiera para un cálculo tan inocuo
# como el suelo nulo, que no ajusta nada pero sí requiere leer las etiquetas
# de test. La cifra citada viene del propio contrato de esta fase (verificada
# de forma independiente antes de empezar) y coincide, como era de esperar,
# con la de train: ambas son ~71% porque el split estratifica por clase
# (F4-R6/R7).
#
# **La trampa, en una frase:** con `pos_label=1` sobre una clase positiva
# mayoritaria, un modelo con **F1 = 0.83 puede no haber aprendido nada**.
# `balanced_accuracy` —media del recall de cada clase por separado— no tiene
# ese problema: su suelo nulo es **0.50 en cualquier split**, porque un
# clasificador que solo acierta en una clase (recall 1.0 en una, 0.0 en la
# otra) promedia exactamente al azar. La decisión de usar
# `balanced_accuracy` como métrica de **selección** de hiperparámetros
# —manteniendo accuracy/precisión/recall/F1 como lo que se **reporta**,
# tal como exige T6/T7— está documentada en
# [ADR-0012](../docs/adr/0012-metrica-de-optimizacion-balanced-accuracy.md)
# y en `docs/CHANGELOG_iteraciones.md` § Loop F.

# %%
print("metrics.optimize_for en params.yaml:", params["metrics"]["optimize_for"])
assert params["metrics"]["optimize_for"] == "balanced_accuracy"
assert params["metrics"]["optimize_for"] != "f1"

# %% [markdown]
# ## Fase 6A — Selección de variables
#
# ### F6A-R1 — ¿Por qué seleccionar variables con solo 10 candidatas?
#
# Con **9 variables numéricas + `Gender`**, la selección de variables **no**
# es una necesidad de dimensionalidad — no hay ningún riesgo de "maldición
# de la dimensionalidad" con 10 candidatas y 456 filas de entrenamiento. Su
# valor aquí es otro:
#
# 1. **Control de redundancia** — `A/G Ratio` es aritmética de `TP` y `ALB`
#    (F4-R3/§5.1 del PRD): las tres viajan juntas y eso introduce
#    multicolinealidad, relevante para la interpretabilidad de los
#    coeficientes de la regresión logística (F6A-R3, más abajo).
# 2. **La decisión sobre `Gender`** (F6A-R4) — la pregunta central de un
#    proyecto de equidad, no una cuestión técnica menor.
# 3. **Comparabilidad entre modelos** — dejar que `SelectKBest` compita entre
#    variables da una medida objetiva (F-estadístico de ANOVA) de cuáles
#    separan mejor las clases en este dataset concreto, en vez de decidirlo
#    por intuición clínica sin formación en el equipo.
#
# ### F6A-R3 — Redundancia por construcción de `A/G Ratio`

# %%
Xnum_train = X_train[NUMERIC_COLS].fillna(X_train[NUMERIC_COLS].median())  # solo para este chequeo descriptivo

corr_tres = Xnum_train[["TP", "ALB", "A/G Ratio"]].corr().round(3)
corr_tres


# %%
def _vif(df_num: pd.DataFrame) -> dict:
    """VIF por regresión lineal (1/(1-R²)) -- sin depender de `statsmodels`."""
    out = {}
    for col in df_num.columns:
        y_col = df_num[col].to_numpy()
        X_resto = df_num.drop(columns=[col]).to_numpy()
        r2 = LinearRegression().fit(X_resto, y_col).score(X_resto, y_col)
        out[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    return out


vif_valores = _vif(Xnum_train)
tabla_vif = pd.DataFrame(
    sorted(vif_valores.items(), key=lambda kv: -kv[1]), columns=["variable", "VIF"]
).round(2)
tabla_vif

# %% [markdown]
# `TP` (5.08) y sobre todo `ALB` (8.99) muestran multicolinealidad
# considerable —por encima del umbral de 5 que suele usarse como señal de
# alerta—, coherente con que `A/G Ratio` se construye a partir de ambas.
# **Lo que no confirma la intuición ingenua:** `A/G Ratio` en sí (3.39)
# **no** es la variable con mayor VIF del grupo. Tiene sentido: es una
# transformación **no lineal** de `TP` y `ALB`
# (`ALB / (TP - ALB)`), y el VIF mide colinealidad **lineal** — una
# transformación no lineal de dos variables correlacionadas no hereda
# automáticamente toda su redundancia lineal. `TB`/`DB` (4.25/4.59) también
# muestran colinealidad moderada entre sí, esperable: la bilirrubina directa
# es por definición una fracción de la total (§5.2 del PRD).
#
# **Consecuencia práctica:** esta redundancia afecta principalmente a la
# **interpretabilidad de los coeficientes** de la regresión logística (un
# coeficiente inestable no implica un modelo peor, pero sí una explicación
# menos confiable de "cuánto pesa cada variable"). No afecta a árboles de
# decisión, KNN ni SVM de la misma manera. Es exactamente el tipo de
# redundancia que `SelectKBest` puede ayudar a resolver, sin necesidad de
# eliminar manualmente ninguna de las tres.
#
# ### F6A-R5 — `Selector` es una etiqueta indirecta (*proxy label*)
#
# `Selector` no es "la enfermedad hepática verificada por biopsia o
# imagenología": es **el juicio de un especialista** en el momento del
# diagnóstico (Ramana & Venkateswarlu, ILPD). Una variable que "separa bien"
# las dos clases puede estar reflejando en qué se fijó ese especialista al
# decidir —por ejemplo, si aplicó con más frecuencia cierto umbral de
# laboratorio a un subgrupo de pacientes— y no necesariamente el daño
# hepático real. Esto no cambia ningún número de este notebook, pero sí
# cómo se debe leer cualquier variable que `SelectKBest` puntúe alto: como
# "predice bien el diagnóstico registrado", no como "causa la enfermedad".
# Se retoma en la Fase 9 (auditoría de equidad).

# %% [markdown]
# ### F6A-R2 — El selector va DENTRO del `Pipeline`, nunca antes
#
# Es la misma razón por la que el escalador y `SMOTE` viven dentro del
# `Pipeline` desde la Fase 5: ajustar el selector con el dataset completo y
# validar después usa información de las filas de validación para decidir
# qué variables quedan — el sesgo de selección documentado por
# [Ambroise & McLachlan (2002)](https://doi.org/10.1073/pnas.102162699),
# citado en §20/§9.3 del PRD.
#
# ```python
# # ❌ MAL -- el selector ve las etiquetas de las filas que después validan
# X_sel = SelectKBest(k=5).fit_transform(X_train, y_train)
# GridSearchCV(modelo, grid).fit(X_sel, y_train)
#
# # ✅ BIEN -- el selector se reajusta en cada pliegue
# pipe = build_pipeline(..., selector=SelectKBest(), ...)
# GridSearchCV(pipe, {"selector__k": [5, 7, 10], "estimator__C": [...]})
# ```
#
# `build_pipeline()` acepta ahora un parámetro `selector` opcional
# (`src/pipelines.py`), colocado entre `"preprocess"` y `"smote"`. Con
# `selector=None` (por defecto) el `Pipeline` es idéntico al de la Fase 5 —
# verificado en `tests/test_fase6a_seleccion.py::test_selector_none_equivale_a_fase5`.

# %%
ejemplo = build_pipeline("zscore", True, None, params, selector=SelectKBest(f_classif, k=5))
print("Pasos con selector:", [n for n, _ in ejemplo.steps])
sin_selector = build_pipeline("zscore", True, None, params)
print("Pasos sin selector (= Fase 5):", [n for n, _ in sin_selector.steps])
assert [n for n, _ in ejemplo.steps] == ["preprocess", "selector", "smote", "estimator"]
assert [n for n, _ in sin_selector.steps] == ["preprocess", "smote", "estimator"]

# %% [markdown]
# ### Demostración en vivo: el selector elige variables distintas en cada pliegue
#
# Si el selector estuviera realmente "fuera" del proceso de validación
# (aunque el código lo escriba dentro del `Pipeline`), debería elegir
# siempre el mismo subconjunto. Se ajusta `preprocess + selector` (sin
# `SMOTE` ni estimador, para aislar el efecto) en cada uno de los 5 pliegues
# de `StratifiedKFold` sobre train, con `k=5`, y se listan las variables
# elegidas en cada uno.

# %%
from imblearn.pipeline import Pipeline as ImbPipeline

pre_sel_template = ImbPipeline(ejemplo.steps[:2])  # "preprocess" + "selector"
cv_demo = build_cv(params)

seleccionadas_por_pliegue = []
for i, (train_fold_idx, _) in enumerate(cv_demo.split(X_train, y_train)):
    p = clone(pre_sel_template)
    X_fold, y_fold = X_train.iloc[train_fold_idx], y_train.iloc[train_fold_idx]
    p.fit(X_fold, y_fold)
    columnas = list(p.named_steps["preprocess"].transform(X_fold).columns)
    soporte = p.named_steps["selector"].get_support()
    elegidas = [c for c, s in zip(columnas, soporte, strict=True) if s]
    seleccionadas_por_pliegue.append(elegidas)
    print(f"pliegue {i}: {elegidas}")

todas_iguales = all(s == seleccionadas_por_pliegue[0] for s in seleccionadas_por_pliegue)
print("\n¿Los 5 pliegues eligieron exactamente el mismo subconjunto?", todas_iguales)
assert not todas_iguales, "si coincidieran siempre, el selector no se estaría reajustando de verdad"

# %% [markdown]
# No coinciden: `TB`, `DB` y `Alkphos` aparecen en los 5 pliegues, pero
# `ALB`, `Sgpt` y `A/G Ratio` compiten por los 2 puestos restantes según qué
# filas concretas le tocan a cada pliegue de entrenamiento. Esa variación
# es exactamente lo que se pierde si el selector se ajustara una sola vez
# sobre todo train: el `k` óptimo y las variables elegidas dejarían de
# reflejar la incertidumbre real de "con estos datos, ¿qué tan estable es
# esta elección?" — la validación cruzada solo es honesta si el selector
# también se reajusta dentro de ella. **Nota adicional:** `Gender` no entra
# en el top-5 de ninguno de los 5 pliegues — primera señal empírica (se
# retoma abajo con más detalle) de que su aporte marginal, en presencia de
# las 9 variables bioquímicas, es limitado en este dataset.

# %% [markdown]
# ### F6A-R4 — La decisión sobre `Gender`: ambas variantes, medidas
#
# Quitar `Gender` **no elimina el sesgo**: la Fase 2b de la Actividad 1 ya
# documentó que los umbrales de ALT y los cuartiles de Tukey usados en este
# campo están calibrados sobre poblaciones mayoritariamente masculinas —
# otras variables pueden codificar información de sexo indirectamente, sin
# que `Gender` esté presente como columna explícita. Por eso F6A-R4 exige
# **medir las dos variantes**, no elegir una por intuición.
#
# Se ajustan los 5 algoritmos, con y sin `Gender` como columna de entrada,
# bajo la configuración base de ajuste (`params["tuning_baseline"]`:
# Z-Score + SMOTE, justificada en la Fase 6 más abajo), optimizando
# `balanced_accuracy` con el `k` del selector incluido en la misma rejilla
# (F6A-R2). **Puede tardar ~1-2 minutos** — son 10 `GridSearchCV`
# (5 modelos x 2 variantes), cada uno con hasta 201 configuraciones x 5
# pliegues.

# %%
t0 = time.time()
resultados_variantes = {}
for variante, X_variante in [("con_gender", X_train), ("sin_gender", X_train.drop(columns=["Gender"]))]:
    resultados_variantes[variante] = {}
    for nombre in MODEL_KEYS:
        search = fit_grid_search(nombre, X_variante, y_train, params, selector=SelectKBest(f_classif))
        cv_scores = cross_validate(
            search.best_estimator_,
            X_variante,
            y_train,
            cv=build_cv(params),
            scoring=["accuracy", "precision", "recall", "f1", "balanced_accuracy"],
        )
        resultados_variantes[variante][nombre] = {
            "best_params_": search.best_params_,
            "balanced_accuracy_cv": search.best_score_,
            "accuracy_cv": cv_scores["test_accuracy"].mean(),
            "precision_cv": cv_scores["test_precision"].mean(),
            "recall_cv": cv_scores["test_recall"].mean(),
            "f1_cv": cv_scores["test_f1"].mean(),
        }
print(f"Tiempo total: {time.time() - t0:.1f} s")

# %%
filas_variantes = []
for variante, modelos in resultados_variantes.items():
    for nombre, r in modelos.items():
        filas_variantes.append(
            {
                "variante": variante,
                "modelo": nombre,
                "balanced_accuracy (CV)": round(r["balanced_accuracy_cv"], 4),
                "accuracy (CV)": round(r["accuracy_cv"], 4),
                "precision (CV)": round(r["precision_cv"], 4),
                "recall (CV)": round(r["recall_cv"], 4),
                "F1 (CV)": round(r["f1_cv"], 4),
                "mejores hiperparámetros": r["best_params_"],
            }
        )
tabla_variantes = pd.DataFrame(filas_variantes)
tabla_variantes.sort_values(["modelo", "variante"]).reset_index(drop=True)

# %%
comparacion = tabla_variantes.pivot(index="modelo", columns="variante", values="balanced_accuracy (CV)")
comparacion["diferencia (sin - con)"] = (comparacion["sin_gender"] - comparacion["con_gender"]).round(4)
comparacion

# %% [markdown]
# **Lectura de la comparación.** Las diferencias de `balanced_accuracy`
# entre variantes son diminutas (≤ 0.004 en valor absoluto) frente a la
# desviación estándar de la propia validación cruzada (~0.03-0.06 en estos
# 5 modelos, calculada más abajo en la sección de Fase 6): **no son
# distinguibles del ruido de muestreo.** `gaussian_nb` y `svm` dan
# resultados **idénticos** entre variantes porque, con su `k` óptimo,
# `SelectKBest` no incluyó `Gender` entre las variables elegidas en ninguna
# de las dos — coherente con el hallazgo del bloque anterior (`Gender` no
# entró en el top-5 de ningún pliegue).
#
# **Variante oficial para las Fases 6-8: SIN `Gender`.** No porque quitarla
# resuelva el problema de equidad del proyecto — no lo hace, como advierte
# F6A-R5 y ya se señaló arriba —, sino porque:
#
# 1. El costo de rendimiento de quitarla es, dentro de este dataset,
#    indistinguible de cero (y en 2 de los 5 modelos, ligeramente mejor).
# 2. Sin una ganancia de rendimiento que lo justifique, mantener una
#    variable protegida como entrada explícita del modelo no aporta nada
#    y sí complica la lectura de cualquier coeficiente o importancia de
#    variable en las fases posteriores.
# 3. Es la postura por defecto más simple para auditar: "el modelo no ve
#    el sexo del paciente directamente" es una afirmación verificable, y
#    la Fase 9 (F9-R5) existe *precisamente* para comprobar si, aun así,
#    la brecha de FNR por sexo persiste a través de variables que
#    correlacionan con `Gender` sin ser `Gender`.
#
# **Ambas variantes quedan construibles** (`fit_grid_search(..., X_variante,
# ...)` acepta cualquiera de las dos) para que la Fase 9 las compare
# directamente, tal como exige F9-R5.

# %%
mejor_modelo_variante_oficial = comparacion["sin_gender"].idxmax()
print("Mejor balanced_accuracy (CV) entre los 5, variante oficial (sin Gender):")
print(f"  {mejor_modelo_variante_oficial}: {comparacion.loc[mejor_modelo_variante_oficial, 'sin_gender']:.4f}")

# %% [markdown]
# ## Fase 6 — Algoritmos e hiperparámetros
#
# ### F6-R1 / F6-R2 — Los 5 algoritmos exigidos y por qué configurarlos así
#
# `src/models.py::build_estimator` construye cada uno con la semilla del
# proyecto (`params["seed"]`, nunca literal — verificado por
# `tests/test_global.py::test_semilla_fijada`) donde aplica.

# %%
tabla_algoritmos = pd.DataFrame(
    [
        {
            "algoritmo": "logistic_regression",
            "hiperparámetros ajustados": "C, solver",
            "por qué importa aquí": (
                "C controla la regularización L2 -- relevante porque TP/ALB/A-G Ratio "
                "son redundantes (F6A-R3, VIF hasta 8.99) y sin regularizar sus "
                "coeficientes pueden volverse inestables. 'saga' soporta L1 y datasets "
                "más grandes que 'liblinear', pero ambos son razonables con 456 filas."
            ),
        },
        {
            "algoritmo": "knn",
            "hiperparámetros ajustados": "n_neighbors, weights",
            "por qué importa aquí": (
                "Basado en distancia euclídea -- sensible a la escala (F6-R7). "
                "'weights=distance' pondera más a los vecinos más cercanos, relevante "
                "porque SMOTE genera sintéticos por interpolación entre vecinos "
                "(Fase 5): un k grande diluye esa estructura local."
            ),
        },
        {
            "algoritmo": "gaussian_nb",
            "hiperparámetros ajustados": "var_smoothing",
            "por qué importa aquí": (
                "var_smoothing añade varianza artificial a cada variable para evitar "
                "divisiones por cero -- amortigua parcialmente el problema de las "
                "variables con colas extremas (Sgot, asimetría 10.5), aunque no "
                "corrige el supuesto de normalidad en sí (F6-R8, más abajo)."
            ),
        },
        {
            "algoritmo": "decision_tree",
            "hiperparámetros ajustados": "max_depth, min_samples_leaf, criterion",
            "por qué importa aquí": (
                "max_depth y min_samples_leaf controlan el sobreajuste con solo 456 "
                "filas de train (menos tras el 80/20 de folds de CV) -- un árbol sin "
                "límite de profundidad memoriza fácilmente un dataset de este tamaño. "
                "Insensible a la escala (F6-R7): las particiones comparan una variable "
                "contra un umbral, no distancias entre variables."
            ),
        },
        {
            "algoritmo": "svm",
            "hiperparámetros ajustados": "C, kernel, gamma",
            "por qué importa aquí": (
                "C controla el margen frente a errores de clasificación; kernel "
                "'linear' vs 'rbf' decide si la frontera de decisión puede ser curva; "
                "gamma solo afecta a 'rbf'. Basado en distancia -- sensible a la "
                "escala (F6-R7), y con 'rbf' sin escalar es patológicamente lento "
                "(advertencia del PRD, Fase 6)."
            ),
        },
    ]
)
tabla_algoritmos

# %% [markdown]
# ### F6-R8 — Advertencia sobre `GaussianNB`: viola su propio supuesto
#
# `GaussianNB` asume que, condicionada a la clase, cada variable numérica
# sigue una distribución normal. La Actividad 1 documentó asimetría de
# hasta **10.5** en `Sgot` — el supuesto se viola abiertamente en varias de
# las 9 variables de este dataset. Esto se reporta como **limitación del
# modelo**, no como un error a corregir: es uno de los cinco algoritmos que
# el enunciado exige evaluar, y su desempeño (peor o mejor) es en sí mismo
# evidencia de cuánto le cuesta el incumplimiento del supuesto — se
# contrasta empíricamente más abajo, junto a los otros 4.

# %% [markdown]
# ### F6-R3 / F6-R4 / F6-R5 — Grid Search con rejillas de `params.yaml`, CV estratificada
#
# Ninguna rejilla vive en este notebook: `build_search_grid()` las lee
# íntegras de `params["grids"]` (verificado en
# `tests/test_fase6_modelos.py::test_rejillas_desde_params`, que falla si
# alguna rejilla quedara copiada literal en `src/models.py`).

# %%
for nombre in MODEL_KEYS:
    grid = build_search_grid(nombre, params)
    n_combos = 1
    for k, v in grid.items():
        if k != "selector__k":
            n_combos *= len(v)
    n_total = n_combos * len(grid["selector__k"])
    print(f"{nombre:20s} {n_combos:3d} combos de hiperparámetros x {len(grid['selector__k'])} valores de k = {n_total:3d}")

# %%
cv = build_cv(params)
print(type(cv).__name__, "n_splits =", cv.n_splits, "| random_state =", cv.random_state)
assert type(cv).__name__ == "StratifiedKFold"

# %% [markdown]
# Los resultados de `GridSearchCV` (`best_params_`, `best_score_`) para la
# **variante oficial (sin `Gender`)** ya se calcularon arriba, en la
# comparación de F6A-R4 — no se repite el ajuste. Se completan aquí las
# métricas que exige T6/T7 (accuracy, precisión, recall, F1), calculadas
# por validación cruzada sobre train (**no** sobre el test, que se reserva
# íntegro para la Fase 7).

# %%
tabla_t7_preliminar = (
    tabla_variantes[tabla_variantes["variante"] == "sin_gender"]
    .drop(columns=["variante"])
    .rename(columns={"modelo": "Modelo", "mejores hiperparámetros": "Hiperparámetros de mejor rendimiento"})
    .set_index("Modelo")
)
tabla_t7_preliminar

# %% [markdown]
# > ⚠️ **Esta NO es la tabla oficial de T7.** El formato de §3.2 del PRD
# > (columnas "Modelo", "Hiperparámetros y cuadrícula de búsqueda",
# > "Hiperparámetros de mejor rendimiento", "Métricas de mejor rendimiento")
# > se arma en la **Fase 7**, con las métricas evaluadas **una sola vez
# > sobre el test congelado** (F7-R4). Esta tabla usa las mismas
# > `best_params_` que la Fase 7 heredará, pero sus métricas son de
# > validación cruzada sobre train — un adelanto honesto, no un sustituto.
#
# ### F6-R6 — La métrica de optimización, verificada empíricamente
#
# Con `balanced_accuracy` como criterio de selección (ADR-0012), se
# confirma lo que predice la teoría: los 5 modelos superan su suelo nulo
# (0.50) por un margen claro, y **precisamente por eso** su F1 de CV queda
# por debajo del suelo nulo de F1 (0.8323) — dejan de sobre-predecir
# "enfermo" para también acertar en la clase sana, lo que baja el recall de
# la clase enferma (y con él, el F1) frente al clasificador degenerado que
# nunca se equivoca en esa clase porque nunca predice la otra.

# %%
resumen_final = tabla_t7_preliminar[
    ["balanced_accuracy (CV)", "accuracy (CV)", "precision (CV)", "recall (CV)", "F1 (CV)"]
].copy()
resumen_final["¿supera suelo balanced_accuracy (0.50)?"] = resumen_final["balanced_accuracy (CV)"] > 0.5
resumen_final["¿supera suelo F1 (0.8323)?"] = resumen_final["F1 (CV)"] > suelo_train["f1"]
resumen_final

# %%
assert (resumen_final["¿supera suelo balanced_accuracy (0.50)?"]).all(), (
    "los 5 modelos deben superar el suelo nulo en la métrica que de verdad se optimiza"
)
assert not (resumen_final["¿supera suelo F1 (0.8323)?"]).any(), (
    "ningún modelo optimizado por balanced_accuracy debería superar el suelo nulo de F1 -- "
    "si esto cambia, es un hallazgo real que hay que investigar, no un error de este assert"
)

# %% [markdown]
# **`gaussian_nb` es, de los 5, el que peor F1 obtiene (0.6234) y el
# segundo peor `balanced_accuracy`** — su precisión es la más alta del
# grupo (0.9521: cuando dice "enfermo", casi siempre acierta) pero su
# recall es el más bajo (0.4646: se le escapa más de la mitad de los
# pacientes sanos... o, leído desde la asimetría, es el que menos
# distingue ambas clases en conjunto). Es coherente con F6-R8: es el único
# de los 5 algoritmos cuyo supuesto matemático central (normalidad por
# variable) se viola de forma demostrable en este dataset. No se descarta
# por eso — es uno de los cinco exigidos por el enunciado —, pero el
# resultado empírico refuerza la advertencia en vez de contradecirla.
#
# ### F6-R7 — Qué modelos son sensibles a la escala (fundamento teórico de la Fase 8)

# %%
tabla_sensibilidad_escala = pd.DataFrame(
    [
        {"algoritmo": "logistic_regression", "sensible a la escala": True, "por qué": "la regularización L2 penaliza coeficientes según la magnitud de cada variable -- variables en escalas distintas reciben penalización desigual sin escalar"},
        {"algoritmo": "knn", "sensible a la escala": True, "por qué": "la distancia euclídea entre vecinos está dominada por la variable de mayor rango absoluto si no se escala"},
        {"algoritmo": "svm", "sensible a la escala": True, "por qué": "el margen y el kernel (sobre todo 'rbf') se calculan sobre distancias/productos internos entre puntos"},
        {"algoritmo": "gaussian_nb", "sensible a la escala": False, "por qué": "estima media y varianza por variable de forma independiente; un cambio de escala no altera qué tan separadas están las clases en unidades de esa variable"},
        {"algoritmo": "decision_tree", "sensible a la escala": False, "por qué": "cada partición compara una variable contra un umbral -- el orden de los valores no cambia al escalar, y es el orden lo único que determina la partición"},
    ]
)
tabla_sensibilidad_escala

# %% [markdown]
# Esta tabla es la base teórica del control de coherencia que la Fase 8
# automatiza (`tests/test_fase8_factorial.py::test_modelos_invariantes_a_escala`,
# pendiente de esa fase): si `gaussian_nb` o `decision_tree` dieran métricas
# distintas entre MinMax y Z-Score, sería señal de un error en el
# experimento factorial, no un hallazgo real.

# %% [markdown]
# ## Resumen
#
# | | Valor |
# |---|---|
# | Suelo nulo (train) | accuracy 0.7127 · recall 1.0000 · precision 0.7127 · **F1 0.8323** · balanced_accuracy 0.5000 |
# | Métrica de selección (`metrics.optimize_for`) | `balanced_accuracy` (ADR-0012) -- no `f1` |
# | Selector de variables | `SelectKBest(f_classif)`, `k ∈ {5,7,10}` optimizado en la misma CV (F6A-R2) |
# | Redundancia (F6A-R3) | `ALB` VIF=8.99, `TP` VIF=5.08, `A/G Ratio` VIF=3.39 (no lineal, no hereda toda la colinealidad) |
# | Variante oficial de `Gender` (F6A-R4) | **Sin `Gender`** -- costo de rendimiento indistinguible del ruido; no elimina el sesgo, solo dejar de usarlo como entrada explícita (F6A-R5) |
# | CV de ajuste | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
# | Rejillas | 67 combinaciones de hiperparámetros x 3 valores de k = 201 configuraciones por variante, desde `params.yaml` (`tests/test_fase6_modelos.py::test_rejillas_desde_params`) |
# | Los 5 algoritmos superan el suelo de `balanced_accuracy` (0.50) | Sí, los 5 (0.67-0.72) |
# | Los 5 algoritmos superan el suelo de F1 (0.8323) | **No, ninguno** -- esperado y explicado (F6-R6) |
# | Peor desempeño | `gaussian_nb` -- coherente con F6-R8 (viola el supuesto de normalidad) |
# | Conjunto de prueba | **Nunca cargado en este notebook** (verificado por `tests/test_fase6_modelos.py::test_test_no_tocado`) |
#
# **Sigue:** Fase 7 — evaluación de los 5 modelos sobre el test congelado,
# **una sola vez**, con la tabla oficial en el formato de §3.2 del PRD.

# %% [markdown]
# ## Nota de traspaso a la Fase 7 -- leer antes de empezar esa fase
#
# Los 5 modelos de esta fase quedan **por debajo del clasificador nulo en
# 2 de las 4 métricas que exige T6/T7** (accuracy y F1): el mejor de los 5
# (`logistic_regression`, variante sin `Gender`) tiene `recall (CV) = 0.60`
# frente al `recall = 1.00` del nulo. Es la consecuencia correcta de
# optimizar `balanced_accuracy` en vez de F1 (F6-R6, arriba) -- pero
# **la tabla de T7, leída sin este contexto, se ve como una derrota en 2
# de las 4 métricas exigidas por el enunciado**, no como una corrección
# metodológica. La Fase 7 tiene que evitar ese malentendido explícitamente
# con tres cosas:
#
# 1. **Justificar las métricas en términos CLÍNICOS, no solo estadísticos
#    (F7-R1).** Este notebook justifica la elección de métrica con
#    matemática (suelos nulos, sensibilidad de cada clase) pero **nunca
#    usa las palabras que el usuario no tiene por qué inferir solo**:
#    - Un **falso negativo** aquí es **un paciente enfermo al que el
#      modelo le dice que está sano** -- se va a casa sin tratamiento.
#    - Un **falso positivo** es **un paciente sano al que el modelo
#      marca como enfermo** -- una prueba de más, con su costo y ansiedad,
#      pero sin el daño de una enfermedad no tratada.
#    Con `recall = 0.60` del mejor modelo, **4 de cada 10 pacientes
#    enfermos de esta muestra quedarían clasificados como sanos** si el
#    modelo se usara con el umbral 0.5 por defecto. La Fase 7 tiene que
#    decir esto en esas palabras, no solo reportar el número.
# 2. **Tratar el umbral de decisión como una decisión de producto, no un
#    detalle técnico (F7-R7).** Con el umbral fijo en 0.5, se está
#    aceptando implícitamente ese `recall = 0.60` -- una decisión real
#    sobre cuántos enfermos se dejan sin detectar, tomada sin que nadie la
#    haya discutido como tal todavía. La Fase 7 debe **evaluar explorar
#    umbrales más bajos que 0.5** (que suben el recall a costa de la
#    precisión) **usando solo train/CV para elegirlo, nunca el test**
#    (B6/F7-R7 del PRD), y reportar el intercambio recall-precisión
#    resultante como una decisión justificada, no solo un número.
# 3. **Presentar la tabla de T7 SIEMPRE junto a la fila del clasificador
#    nulo** (accuracy 0.7105 · recall 1.0000 · precision 0.7105 ·
#    F1 0.8308, cifras de test -- F7-R3 exige exactamente esta
#    comparación), para que cualquiera que lea la tabla vea de inmediato
#    que un modelo con accuracy/F1 más bajos que el nulo puede seguir
#    siendo el modelo correcto, sin depender de que lea también el texto
#    de alrededor.
