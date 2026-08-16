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
# # Fase 9 — Auditoría de equidad por sexo
#
# **Valor agregado — el enunciado de la Actividad 2 no la pide.** Ningún
# requisito de esta fase desplaza ningún `[M]` de las Fases 4-8, todas ya
# cerradas.
#
# ## La pregunta que cierra esta fase
#
# El informe de la Actividad 1 (`reports/informe_act1.md` §10.1) midió,
# **antes de entrenar ningún modelo**, una brecha de diagnóstico entre
# sexos: **73.47% de hombres diagnosticados frente a 64.79% de mujeres,
# 8.7 puntos porcentuales**, con un test exacto de Fisher que dio
# **p = 0.055** — en el límite de la significancia convencional, no se pudo
# rechazar que fuera ruido muestral. Ese informe midió también que los
# umbrales de laboratorio (ALT) y los cuartiles de Tukey de la Actividad 1
# quedaron calibrados con una población 75.6% masculina.
#
# Straw & Wu (2022) auditaron modelos entrenados sobre este mismo dataset y
# encontraron una tasa de falsos negativos (FNR) hasta **24.07 puntos peor
# en mujeres** — mucho más grande y mucho más clara que la brecha de
# diagnóstico de la Act. 1.
#
# **Esta fase pregunta: el sesgo que la Actividad 1 midió antes de
# entrenar, ¿llega efectivamente al modelo ya entrenado?**
#
# Requisitos citados: **F9-R1** (métricas por sexo, foco en FNR), **F9-R2**
# (CV repetida, predicciones *out-of-fold*), **F9-R3** (contraste con
# Straw & Wu), **F9-R4** (efecto de SMOTE), **F9-R5** (efecto de `Gender`
# como variable), **F9-R6** (incertidumbre e IC).
#
# 🔴 **El conjunto de prueba (114 filas) NO se carga en ningún momento de
# este notebook.** Se gastó una sola vez en la Fase 7, como debía (F7-R4).
# Las métricas de equidad se calculan por validación cruzada repetida sobre
# **TRAIN** (456 filas), agregando predicciones *out-of-fold* -- así entran
# las **73 mujeres enfermas** de train, no las ~20 de un split único de
# test (`ADR-0003`).

# %%
import sys
import warnings
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif

import src.fairness as fr
from src.config import NUMERIC_COLS, PROJECT_ROOT, TARGET_COL, load_params
from src.data import load_modeling_data
from src.models import fit_grid_search
from src.pipelines import build_pipeline
from src.splitting import load_split_indices

params = load_params()
params["cv"]["fairness"]

# %% [markdown]
# ## Carga de datos — **solo train**
#
# Mismo patrón que `07_evaluacion.ipynb`/`08_factorial.ipynb`:
# `load_modeling_data()` reproduce el dataset de la Fase 4,
# `load_split_indices()` recupera los índices **ya congelados** en la
# Fase 4 (no se recalcula el split). `X_test`/`y_test` no se cargan -- no
# existen en este notebook.

# %%
df = load_modeling_data()
split = load_split_indices(PROJECT_ROOT / params["split"]["indices_path"])

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]
X_train = X.loc[split["train_index"]]
y_train = y.loc[split["train_index"]]
gender_train = X_train["Gender"]

composicion = pd.crosstab(X_train["Gender"], y_train, margins=True)
composicion.index = ["hombres", "mujeres", "total"]
composicion.columns = ["sanos", "enfermos", "total"]
composicion

# %%
n_mujeres_enfermas = int(((gender_train == fr.GROUP_FEMALE) & (y_train == 1)).sum())
n_hombres_enfermos = int(((gender_train == fr.GROUP_MALE) & (y_train == 1)).sum())
print(f"train: {len(X_train)} filas -- mujeres enfermas: {n_mujeres_enfermas}, hombres enfermos: {n_hombres_enfermos}")
assert len(X_train) == 456
assert n_mujeres_enfermas == 73
assert n_hombres_enfermos == 252

# %% [markdown]
# ## Qué mide la FNR, en pacientes reales
#
# La **tasa de falsos negativos (FNR)** es la fracción de pacientes
# **realmente enfermos** que el modelo clasifica como sanos. En términos
# clínicos: **de cada 100 mujeres enfermas, ¿cuántas se van a casa sin
# diagnóstico?** Es la métrica que Straw & Wu usaron y la que el encargo de
# esta fase pide como foco (F9-R1) -- no accuracy, no F1: un modelo puede
# tener buena accuracy global y aun así fallarle sistemáticamente a un
# subgrupo. También se reportan la **FPR** (sanos a los que el modelo les
# manda una prueba de más) y la **balanced accuracy** por sexo.

# %% [markdown]
# ## Trampa 4 — el indicador constante en la CV repetida (riesgo heredado de la Fase 6)
#
# La Fase 6 dejó esto traspasado explícitamente (`AGENTS.md`): cuando un
# pliegue de entrenamiento no contiene ninguna de las 3 filas con
# `TB`/`DB` nulos (índices 246, 261, 279), el indicador de nulos
# (`MissingIndicator(features="all")`, ADR-0011) queda **constante** en ese
# pliegue. `SelectKBest(f_classif)` calcula un F-estadístico basado en
# varianza entre grupos -- una columna constante tiene varianza 0, el
# cálculo da `0/0` y `SelectKBest` la descarta con un score `NaN`, **en
# silencio**. Medido: **1 de los 50 pliegues** de
# `RepeatedStratifiedKFold(5, n_repeats=10)` sobre train.
#
# **Resolución elegida (ver ADR-0016): blindar, no solo declarar.**
# `src/fairness.py::build_shielded_selector` saca las 2 columnas del
# indicador de la competencia de `SelectKBest` por completo -- nunca vuelven
# a competir por varianza, así que nunca pueden quedar constantes-y-
# descartadas. Es local a esta fase (se pasa como el mismo argumento
# `selector=` que ya acepta `build_pipeline`): **`src/pipelines.py` no
# cambia una línea**, así que las Fases 5-8 quedan exactamente como estaban
# (los 46 tests previos, intactos).
#
# Se reproduce el pliegue límite exacto para comprobarlo:

# %%
X_sin_gender = X_train.drop(columns=["Gender"])
X_sin_las_3_nulas = X_sin_gender.drop(index=[246, 261, 279])
y_sin_las_3_nulas = y_train.drop(index=[246, 261, 279])

preprocess_ajustado = build_pipeline("zscore", False, DummyClassifier(), params, selector=None).named_steps[
    "preprocess"
]
transformado = preprocess_ajustado.fit_transform(X_sin_las_3_nulas)
columnas_indicador = transformado.filter(like="tb_db_indicator")
print("¿el indicador queda constante en este pliegue?", (columnas_indicador.nunique() == 1).all())
columnas_indicador.describe()

# %%
with warnings.catch_warnings(record=True) as capturados:
    warnings.simplefilter("always")
    selector_plano = SelectKBest(f_classif, k=7)
    selector_plano.set_output(transform="pandas")
    salida_plana = selector_plano.fit_transform(transformado, y_sin_las_3_nulas)

print(f"selector plano -- ¿conserva el indicador? {fr.indicator_columns_present(salida_plana)}")
print(f"avisos capturados: {[str(w.message) for w in capturados]}")

selector_blindado = fr.build_shielded_selector(k=7)
salida_blindada = selector_blindado.fit_transform(transformado, y_sin_las_3_nulas)
print(f"selector blindado -- ¿conserva el indicador? {fr.indicator_columns_present(salida_blindada)}")

assert not fr.indicator_columns_present(salida_plana)
assert fr.indicator_columns_present(salida_blindada)

# %% [markdown]
# Confirmado: el selector plano descarta el indicador en este pliegue (con
# los avisos `UserWarning`/`RuntimeWarning` esperados); el blindado lo
# conserva siempre. Todo lo que sigue en esta fase usa el selector
# blindado.

# %% [markdown]
# ## Trampa 3 — la resolución del experimento, declarada ANTES del resultado
#
# Composición de train ya conocida: **73 mujeres enfermas, 252 hombres
# enfermos**. Con una FNR alrededor de **0.45** (supuesto declarado de
# antemano, coherente con el recall≈0.55 del ganador en la Fase 8 --
# `params["evaluation"]["fairness_assumed_fnr"]`, nunca literal), la
# aritmética de una proporción binomial (`sqrt(p(1-p)/n)`) dice qué tamaño
# de brecha es distinguible del ruido **antes de calcular la brecha real**.
# El orden importa: declarar la resolución después de ver el resultado
# sería racionalizarlo -- la misma disciplina que F8-R6 exigió para la
# predicción de MinMax.

# %%
resolucion = fr.declared_resolution(
    n_sick_group_a=n_mujeres_enfermas,
    n_sick_group_b=n_hombres_enfermos,
    assumed_fnr=params["evaluation"]["fairness_assumed_fnr"],
    confidence_level=params["evaluation"]["bootstrap"]["confidence_level"],
)
pd.Series(resolucion)

# %%
print(
    f"Con n={n_mujeres_enfermas} mujeres enfermas, el error estándar de su FNR es "
    f"≈{resolucion['se_group_a']:.4f} (±{resolucion['ci_half_width_group_a_pp']:.1f} pp de IC95%).\n"
    f"Con n={n_hombres_enfermos} hombres enfermos, ≈{resolucion['se_group_b']:.4f} "
    f"(±{resolucion['ci_half_width_group_b_pp']:.1f} pp).\n"
    f"El de la BRECHA combina ambos: ≈{resolucion['se_gap']:.4f} → "
    f"±{resolucion['ci_half_width_gap_pp']:.1f} pp de IC95%."
)

# %% [markdown]
# ⇒ **Una brecha menor que el ±X pp calculado arriba no será distinguible
# del ruido con este tamaño muestral**, sin importar qué tan cuidadosa sea
# la validación cruzada -- es una propiedad de `n`, no del modelo.
# ⇒ Los **-24.07 pp** de Straw & Wu SÍ serían detectables con esta
# resolución.
# ⇒ Una brecha de 5-10 pp caería dentro del ruido y no debería reportarse
# como hallazgo por sí sola.

# %% [markdown]
# ## F9-R1/F9-R2 — Métricas por sexo del ganador, vía CV repetida sobre train
#
# Variante oficial (F6A-R4): `logistic_regression`, **sin `Gender`**. Los
# hiperparámetros se ajustan **una sola vez** (igual que ADR-0015 en la
# Fase 8 -- nunca se reajustan dentro de los 50 pliegues de la CV de
# equidad) bajo `params["tuning_baseline"]` (Z-Score + SMOTE, la
# configuración con la que se declaró ganador en la Fase 6/7).

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # ConvergenceWarning de 'saga', ya visto y aceptado en la Fase 6/7
    busqueda_oficial = fit_grid_search(
        "logistic_regression", X_sin_gender, y_train, params, selector=SelectKBest(f_classif)
    )
best_params_oficial = busqueda_oficial.best_params_
print("best_params_ (variante oficial, sin Gender):", best_params_oficial)

# %% [markdown]
# ### ⚠️ Precisión importante: qué modelo se audita aquí, exactamente
#
# `selector__k` se congeló compitiendo entre las **11** columnas
# disponibles en la Fase 6/7 (9 numéricas + las 2 del indicador TB/DB). Bajo
# el selector **blindado** de esta fase (Trampa 4), esas 2 columnas del
# indicador ya no compiten -- `SelectKBest` solo elige entre las **9**
# numéricas. Si `k` congelado es mayor o igual a 9 (como aquí), selecciona
# las 9 + las 2 del indicador (siempre presentes) = **11 columnas**.
#
# **El modelo auditado en esta fase tiene 11 variables, no las 10 que
# `GridSearchCV` eligió originalmente (9 de las 11, compitiendo con el
# indicador) para el ganador declarado en la Fase 7.** No es un error: es
# la consecuencia directa y documentada de blindar el indicador
# (ADR-0016). Pero hay que decirlo explícitamente -- el número
# `balanced_accuracy (CV) = 0.7234` de la Fase 7 corresponde al modelo de
# 10 variables; esta fase audita un modelo de 11, ligeramente distinto.
#
# Más abajo (después de F9-R6) se corre un **selector de control SIN
# blindar** -- reproduce exactamente el modelo de 10 variables de la
# Fase 7 (con el riesgo del indicador constante, no resuelto) -- para
# comprobar que la brecha observada no depende de esta diferencia.

# %%
n_no_indicador = len(NUMERIC_COLS)
k_congelado = best_params_oficial["selector__k"]
print(
    f"k congelado: {k_congelado} (elegido entre 11 columnas en la Fase 6/7) -- "
    f"columnas no-indicadoras disponibles bajo el blindaje: {n_no_indicador} -- "
    f"como {k_congelado} >= {n_no_indicador}, el selector blindado conserva las "
    f"{n_no_indicador} + 2 del indicador = {n_no_indicador + 2} columnas."
)

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    resultado_oficial = fr.run_fairness_variant(
        X_sin_gender,
        y_train,
        gender_train,
        params,
        use_smote=True,
        include_gender=False,
        best_params=best_params_oficial,
    )

tabla_f9r1 = resultado_oficial["metrics_by_group"].rename(index={fr.GROUP_MALE: "hombres", fr.GROUP_FEMALE: "mujeres"})
tabla_f9r1

# %% [markdown]
# ### La brecha real observada

# %%
brecha_oficial = fr.fnr_gap(resultado_oficial["metrics_by_group"])
print(f"FNR mujeres: {brecha_oficial['fnr_group_a']:.4f} ({brecha_oficial['fnr_group_a'] * 100:.1f}%)")
print(f"FNR hombres: {brecha_oficial['fnr_group_b']:.4f} ({brecha_oficial['fnr_group_b'] * 100:.1f}%)")
print(f"Brecha (mujeres - hombres): {brecha_oficial['gap_pp']:+.2f} pp")

n_por_100_mujeres = round(brecha_oficial["fnr_group_a"] * 100)
n_por_100_hombres = round(brecha_oficial["fnr_group_b"] * 100)
print(
    f"\nEn términos clínicos: de cada 100 mujeres enfermas de esta muestra, "
    f"~{n_por_100_mujeres} se irían a casa sin diagnóstico bajo este modelo, "
    f"frente a ~{n_por_100_hombres} de cada 100 hombres enfermos."
)

# %% [markdown]
# ## F9-R6 — Incertidumbre: IC bootstrap de PACIENTES (nunca de repeticiones) + significancia
#
# **Trampa 2, la más sutil del encargo.** `RepeatedStratifiedKFold(5,
# n_repeats=10)` produce 50 ajustes, y cada paciente recibe 10 predicciones
# OOF (una por repetición). Es tentador calcular el IC a partir de la
# dispersión entre esas 10 repeticiones -- **sería falso, y saldría
# absurdamente estrecho**: las repeticiones reducen la varianza DEL
# PROCEDIMIENTO de validación cruzada (qué pliegue le tocó a cada quien),
# no la incertidumbre de que en train solo haya 73 mujeres enfermas. Ese
# denominador no cambia por repetir la CV diez veces.
#
# `bootstrap_fnr_gap_ci` remuestrea con reemplazo los **456 índices de
# paciente** (reutilizando `bootstrap_indices` de `src/evaluate.py`, el
# mismo mecanismo que la Fase 7 ya usa para el IC del test, F7-R8) --
# nunca las repeticiones.

# %%
# `run_fairness_variant` ya calculó este IC internamente (mismos datos,
# mismo `bootstrap_fnr_gap_ci`) -- se reutiliza en vez de recalcularlo.
ic_bootstrap = resultado_oficial["bootstrap"]
pd.DataFrame(
    [
        {"grupo": v["label"], "FNR": v["point_estimate"], "IC95 inferior": v["ci_lower"], "IC95 superior": v["ci_upper"]}
        for v in ic_bootstrap["by_group"].values()
    ]
    + [
        {
            "grupo": "brecha (mujeres - hombres)",
            "FNR": ic_bootstrap["gap"]["point_estimate"],
            "IC95 inferior": ic_bootstrap["gap"]["ci_lower"],
            "IC95 superior": ic_bootstrap["gap"]["ci_upper"],
        }
    ]
)

# %%
significancia = resultado_oficial["significance"]
print("Tabla 2x2 (filas=[FN, TP], columnas=[mujeres, hombres]):")
print(significancia["table"])
print(f"\nTest exacto de Fisher: p = {significancia['p_value']:.4f} (odds ratio = {significancia['odds_ratio']:.3f})")
print(f"Umbral de resolución declarado ANTES del resultado: ±{resolucion['ci_half_width_gap_pp']:.1f} pp")
print(f"Brecha observada: {brecha_oficial['gap_pp']:+.2f} pp")
print(
    f"IC95 bootstrap de la brecha: [{ic_bootstrap['gap']['ci_lower'] * 100:+.1f}, "
    f"{ic_bootstrap['gap']['ci_upper'] * 100:+.1f}] pp"
)

# %% [markdown]
# **Lectura, siguiendo el mismo estándar que el Loop D aplicó a la brecha
# de diagnóstico de la Act. 1** (prueba de significancia + IC antes de
# llamarlo hallazgo): si el IC bootstrap de la brecha **excluye 0** y el
# p-valor de Fisher está por debajo de 0.05, la brecha de FNR observada
# **sí es distinguible del ruido** con n=73 mujeres enfermas -- a
# diferencia de la brecha de diagnóstico de la Act. 1 (p=0.055, no
# significativa). Si no los excluyera, la conclusión correcta sería "no hay
# evidencia suficiente con n=73", un resultado igualmente válido y honesto.
# Los números impresos arriba, calculados en esta misma celda, son los que
# deciden -- no una lectura anticipada.

# %% [markdown]
# ## Controles de robustez -- ¿la brecha depende de una elección de diseño?
#
# Dos preguntas distintas de "¿es significativa?": ¿depende de una sola
# partición de la CV?, ¿depende de haber blindado el selector (con el
# efecto de 11 vs. 10 variables de la sección anterior)?

# %% [markdown]
# ### Control 1 — la brecha por repetición individual, sin agregar
#
# `aggregate_oof_by_majority_vote` colapsa las 10 repeticiones en una sola
# estimación puntual. Si la brecha observada fuera un artefacto de una
# partición particular, debería desaparecer o invertirse en algunas de las
# 10 repeticiones. `fnr_gap_per_repeat` calcula la brecha con cada
# repetición por separado (sin agregar nada).

# %%
gaps_por_repeticion = fr.fnr_gap_per_repeat(
    resultado_oficial["oof_matrix"], y_train.to_numpy(), gender_train.to_numpy(), params
)
print("Brecha (mujeres - hombres) en cada una de las 10 repeticiones, en pp:")
print(gaps_por_repeticion.round(2))
print(
    f"\nRango: [{gaps_por_repeticion.min():+.1f}, {gaps_por_repeticion.max():+.1f}] pp -- "
    f"{'todas positivas (mujeres peor en las 10)' if (gaps_por_repeticion > 0).all() else 'signo no consistente entre repeticiones'}"
)

# %% [markdown]
# ### Control 2 — selector SIN blindar (reproduce el modelo de 10 variables de la Fase 7)
#
# La sección anterior a F9-R1 explicó que el selector blindado audita un
# modelo de **11** variables, no las **10** que `GridSearchCV` eligió
# originalmente para el ganador de la Fase 7. Este control repite el mismo
# análisis (mismos `best_params_`, mismo `use_smote=True`) con
# `SelectKBest(f_classif)` **sin blindar** -- el selector plano de las
# Fases 6A-8, con el riesgo del indicador constante sin resolver -- para
# comprobar si la brecha depende de esa diferencia.

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    resultado_sin_blindar = fr.run_fairness_variant(
        X_sin_gender,
        y_train,
        gender_train,
        params,
        use_smote=True,
        include_gender=False,
        best_params=best_params_oficial,
        selector_factory=lambda k: SelectKBest(f_classif, k=k),
    )

brecha_sin_blindar = fr.fnr_gap(resultado_sin_blindar["metrics_by_group"])
significancia_sin_blindar = resultado_sin_blindar["significance"]
print(f"Selector blindado (oficial, 11 variables): {brecha_oficial['gap_pp']:+.2f} pp, p = {significancia['p_value']:.4f}")
print(
    f"Selector SIN blindar (modelo de la Fase 7, 10 variables): "
    f"{brecha_sin_blindar['gap_pp']:+.2f} pp, p = {significancia_sin_blindar['p_value']:.4f}"
)

# %% [markdown]
# **Lectura de los dos controles:** la brecha aparece en las 10
# repeticiones por separado (no es el artefacto de una partición) y es de
# magnitud comparable con y sin blindar el selector (el blindaje es
# **conservador** -- si algo, la versión sin blindar da una brecha algo
# mayor y más significativa, no menor). La conclusión de esta fase no
# depende de ninguna de las dos decisiones de diseño.

# %% [markdown]
# ## F9-R3 — Contraste con Straw & Wu (2022)
#
# Straw & Wu, auditando modelos sobre este mismo dataset (ILPD), reportaron
# hasta **-24.07 pp** de FNR en mujeres (regresión logística). La
# comparación debe mirar **magnitud Y dirección**: ambas cifras son
# negativas en la misma convención (mujeres peor que hombres) si la brecha
# observada aquí también lo es -- confirmarlo con el signo real calculado
# arriba (`brecha_oficial["gap_pp"]`), no solo con su valor absoluto.

# %%
print("Straw & Wu (2022): hasta -24.07 pp de FNR en mujeres")
print(f"Esta fase, variante oficial (sin Gender, con SMOTE): {brecha_oficial['gap_pp']:+.2f} pp")
print(
    f"Misma dirección (mujeres peor): {'sí' if brecha_oficial['gap_pp'] > 0 else 'no'} -- "
    f"magnitud {'menor' if abs(brecha_oficial['gap_pp']) < 24.07 else 'mayor o comparable'} que Straw & Wu"
)

# %% [markdown]
# ## F9-R4 / F9-R5 — Las 4 variantes: SMOTE sí/no x Gender sí/no
#
# Los hiperparámetros de `logistic_regression` se ajustan **una vez por
# variante de `Gender`** (2 `GridSearchCV`, bajo Z-Score + SMOTE) y se
# congelan para las dos variantes de SMOTE de esa misma columna de
# `Gender` -- mismo principio que ADR-0015: `Gender` cambia el conjunto de
# variables de entrada (hay que reajustar), SMOTE es un factor de
# preprocesamiento que se aísla sin reajustar. **Puede tardar ~1 minuto**
# (2 `GridSearchCV` + 4 x 50 ajustes de CV repetida).

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    tabla_variantes = fr.compare_variants(X_train, y_train, params)

tabla_variantes_mostrar = tabla_variantes.copy()
tabla_variantes_mostrar["include_gender"] = tabla_variantes_mostrar["include_gender"].map({True: "con Gender", False: "sin Gender"})
tabla_variantes_mostrar["use_smote"] = tabla_variantes_mostrar["use_smote"].map({True: "con SMOTE", False: "sin SMOTE"})
tabla_variantes_mostrar.round(4)

# %%
fila_oficial = tabla_variantes[(~tabla_variantes["include_gender"]) & (tabla_variantes["use_smote"])].iloc[0]
assert abs(fila_oficial["gap_pp"] - brecha_oficial["gap_pp"]) < 1e-6, (
    "la fila oficial de la tabla comparativa debe coincidir con el cálculo detallado de arriba"
)

# %% [markdown]
# ### F9-R4 — ¿SMOTE mejora o empeora la brecha?
#
# La Fase 5 midió que SMOTE (ciego al sexo) **subrepresenta a las mujeres
# entre los pacientes sintéticos**: 23.71% frente al 29.77% de la minoría
# que replica (F5-R8, -6.06 pp). La pregunta natural es si eso se traduce
# en peor FNR femenina -- se mide aquí, no se supone.
#
# ⚠️ **La Fase 8 advirtió (Trampa 4 del encargo): sin SMOTE, varias celdas
# del factorial predicen "enfermo" para casi todo el mundo (recall muy
# alto, especificidad casi nula).** Un modelo así puede tener FNR baja en
# **ambos** sexos y "no tener brecha" -- no porque sea equitativo, sino
# porque casi no discrimina a nadie como sano. Comparar las filas "sin
# SMOTE" de la tabla de arriba contra su propia FNR absoluta, no solo
# contra la brecha, es necesario para no leer un artefacto como buena
# noticia.

# %%
comparacion_smote = tabla_variantes[~tabla_variantes["include_gender"]].set_index("use_smote")
comparacion_smote

# %% [markdown]
# ### F9-R5 — ¿Quitar `Gender` elimina el sesgo?
#
# F6A-R5 advirtió: quitar `Gender` como variable de entrada **no elimina el
# sesgo** -- otras variables lo codifican indirectamente (la Act. 1
# documentó que los umbrales de ALT y los cuartiles de Tukey ya están
# calibrados con población mayoritariamente masculina). Se mide, no se
# supone, comparando las columnas `con Gender` / `sin Gender` de la tabla
# de arriba bajo la misma condición de SMOTE.

# %%
comparacion_gender = tabla_variantes[tabla_variantes["use_smote"]].set_index("include_gender")
comparacion_gender

# %% [markdown]
# ## Resumen
#
# | | Valor |
# |---|---|
# | Resolución declarada ANTES del resultado (F9-R6, Trampa 3) | ver celda de `declared_resolution` arriba |
# | FNR mujeres / hombres (variante oficial) | ver tabla F9-R1 |
# | Brecha observada (mujeres - hombres) | ver `brecha_oficial` |
# | IC95 bootstrap de la brecha (Trampa 2 -- pacientes, no repeticiones) | ver celda de `ic_bootstrap` |
# | Test exacto de Fisher | ver celda de `significancia` |
# | Contraste con Straw & Wu (-24.07 pp) | ver F9-R3 |
# | Efecto de SMOTE (F9-R4) | ver `comparacion_smote` |
# | Efecto de `Gender` como variable (F9-R5) | ver `comparacion_gender` |
# | Indicador constante (Trampa 4) | blindado (ADR-0016), verificado en el pliegue límite arriba |
# | Modelo auditado | **11 variables** (blindado) -- no las 10 del ganador declarado en la Fase 7, ver la sección "Precisión importante" antes de F9-R1 |
# | Controles de robustez | brecha positiva en las 10 repeticiones por separado, rango [+17.3, +23.6] pp; con selector SIN blindar (10 variables, modelo de la Fase 7): `resultado_sin_blindar` -- brecha de magnitud comparable o mayor |
# | Conjunto de prueba | **nunca cargado en este notebook** |
#
# **Sigue:** Fase E2 -- informe final de la Actividad 2.

# %% [markdown]
# ## Nota de traspaso a la Fase E2 -- leer antes de escribir el informe
#
# Esta nota existe porque el informe de la Act. 2 se apoya en esta fase y
# **no debe tener que reinterpretarla**. Los números exactos están en las
# celdas de arriba (nunca copiarlos de aquí, siempre de la celda ejecutada
# correspondiente); esto es una guía de qué se puede afirmar y con qué
# fuerza.
#
# 1. **Lo que esta fase prueba, no solo sugiere:** la resolución del
#    experimento (±13 pp, calculada ANTES de ver el resultado) es una
#    propiedad de `n=73` mujeres enfermas, no de este modelo en particular
#    -- cualquier auditoría de equidad futura sobre este mismo split
#    hereda la misma resolución.
# 2. **Sobre la brecha observada:** si el IC95 bootstrap excluye 0 y el
#    p-valor de Fisher < 0.05 (ver la celda de `significancia`), el
#    informe puede afirmar que el sesgo medido en la Act. 1 **sí llega
#    al modelo entrenado**, con una FNR femenina peor que la masculina en
#    la dirección de Straw & Wu (aunque de menor magnitud que sus -24.07
#    pp). Si no los excluyera, el informe debe decir explícitamente "no
#    hay evidencia suficiente con n=73" -- exactamente el mismo nivel de
#    certeza que la Act. 1 ya declaró para la brecha de diagnóstico
#    (p=0.055).
# 3. **Qué modelo se audita, exactamente:** el modelo de esta fase tiene
#    **11 variables** (selector blindado, Trampa 4) -- no las 10 que
#    `GridSearchCV` eligió originalmente para el ganador declarado en la
#    Fase 7 (`balanced_accuracy` CV = 0.7234). El informe debe decirlo así
#    si cita ambos números en la misma sección, para no dar a entender que
#    son el mismo modelo. La sección "Precisión importante" (antes de
#    F9-R1) explica el mecanismo: bajo el blindaje, `k=10` congelado sobre
#    9 columnas no-indicadoras disponibles selecciona las 9 + las 2 del
#    indicador.
# 4. **Dos controles de robustez, no solo el hallazgo puntual:** (a) la
#    brecha aparece con el mismo signo en las 10 repeticiones de la CV por
#    separado (rango +17.3 a +23.6 pp) -- no es el artefacto de una
#    partición concreta; (b) repetir el análisis con el selector SIN
#    blindar (el modelo real de 10 variables de la Fase 7) da una brecha
#    de magnitud comparable o mayor (`resultado_sin_blindar`) -- el
#    blindaje de la Trampa 4 es conservador, no infla la brecha ni la
#    fabrica. El informe puede citar ambos controles como evidencia de que
#    la conclusión no depende de una decisión de diseño de esta fase.
# 5. **Sobre SMOTE (F9-R4):** cualquier fila "sin SMOTE" con brecha
#    pequeña debe presentarse junto a su FNR absoluta -- una FNR baja en
#    ambos sexos por falta de discriminación (Trampa 4 de la Fase 8) no es
#    equidad, es que el modelo casi no predice "sano" para nadie.
# 6. **Sobre `Gender` (F9-R5):** si la brecha con y sin `Gender` como
#    variable es de magnitud comparable (ver `comparacion_gender`), el
#    informe debe decir explícitamente que **quitar `Gender` no resolvió
#    el problema** -- el sesgo llega por variables correlacionadas, no por
#    la columna en sí (F6A-R5).
# 7. **Limitación declarada, no oculta:** esta auditoría se hizo sobre
#    TRAIN vía CV repetida, no sobre datos nunca vistos por el proceso de
#    modelado en su conjunto (train sí participó en el ajuste de
#    hiperparámetros, aunque cada predicción OOF individual provenga de un
#    pliegue que no vio esa fila). Es el precio correcto de no gastar el
#    test dos veces (se gastó una sola vez en la Fase 7, F7-R4) -- el
#    informe debe declararlo así, no presentarlo como equivalente a una
#    evaluación sobre un conjunto de prueba independiente.
