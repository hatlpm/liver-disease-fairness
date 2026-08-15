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
# # Fase 4 — Contrato de datos y split
#
# **Actividad 2 · Criterio 1 (C1, 25%) · Tareas T1 y T3 del enunciado.**
#
# Este notebook responde:
#
# - **T1** — ¿Cómo se aseguró que los datos estuvieran limpios y listos
#   para el modelado?
# - **T3** — ¿Cómo se dividió el conjunto en entrenamiento y prueba? ¿Qué
#   proporciones?
#
# La lógica vive en `src/data.py` y `src/splitting.py` — este notebook solo
# la invoca y narra (regla del proyecto, P-R2 del PRD). Ningún número de
# este notebook se copia de la documentación: todos salen de una celda
# ejecutada aquí mismo.

# %%
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import AGE_TOP_CODE, PROJECT_ROOT, load_params
from src.data import (
    deduplicate,
    encode_gender,
    encode_selector,
    load_modeling_data,
    nullify_biochemical_violations,
    reconstruct_ag_ratio,
    row_signature,
)
from src.splitting import (
    save_split_indices,
    split_selector_only_for_comparison,
    split_train_test,
)
from src.utils import flag_biochemical_violations, load_raw_data

params = load_params()
params

# %% [markdown]
# ## T1 — Limpieza para el modelado
#
# Partimos del CSV crudo (`data/raw/`, **nunca** de `data/processed/`, que
# en la Actividad 1 se escaló sobre las 583 filas sin deduplicar — usarlo
# aquí reintroduciría la fuga que esta fase existe para evitar).
#
# Encadenamos cuatro pasos, en un orden que **no es intercambiable**. Antes
# de ejecutarlos, tres advertencias sobre errores fáciles de cometer aquí:
#
# 1. **Los `NaN` de `TB`/`DB` no vienen en el CSV crudo.** Las 3 filas con
#    `DB > TB` conservan sus valores imposibles tal cual — marcarlas como
#    faltantes fue una decisión de la Fase 3 de la Actividad 1 aplicada al
#    dataset *procesado*, no algo que traiga el archivo fuente. Hay que
#    volver a aplicarla aquí.
# 2. **Hay que deduplicar antes de dividir, nunca después.** Con
#    `random_state=42`, varios de los 13 pares de filas idénticas caen a
#    ambos lados de un split hecho sobre los datos crudos: el modelo
#    "acertaría" en test una fila que ya memorizó en train.
# 3. **No toda operación previa al split es igual de segura.** Reconstruir
#    `A/G Ratio` es aritmética fila a fila (no usa ninguna estadística del
#    conjunto) y por eso puede ir aquí. Imputar `TB`/`DB` usaría la mediana
#    del grupo — sí sería fuga si se calculara sobre el dataset completo —
#    así que esos `NaN` se dejan para el `Pipeline` de la Fase 5.

# %%
raw = load_raw_data()
print(f"Crudo: {raw.shape[0]} filas x {raw.shape[1]} columnas")
raw.head(3)

# %% [markdown]
# ### Paso 1 — Deduplicar (F4-R2)

# %%
n_pairs_duplicated = int(raw.duplicated(keep="first").sum())
duplicated_rows = raw[raw.duplicated(keep=False)]
print(f"Pares de filas idénticas: {n_pairs_duplicated} ({len(duplicated_rows)} filas implicadas)")
print("\nClase de las filas duplicadas (conteo de Selector):")
print(duplicated_rows["Selector"].value_counts())

deduped = deduplicate(raw)
print(f"\nTras deduplicar: {deduped.shape[0]} filas (era {raw.shape[0]})")

# %% [markdown]
# **583 → 570 filas.** Se pierde un 2.2% de la muestra para eliminar una
# fuga garantizada. Esto **diverge deliberadamente de la Actividad 1**,
# donde conservar los 13 pares era la decisión correcta: allí no se
# entrenaba ningún modelo, y no había forma de saber si eran pacientes
# distintos con analítica idéntica o el mismo registro repetido. Aquí el
# criterio cambia porque el riesgo cambia — ver
# [ADR-0006](../docs/adr/0006-deduplicar-antes-del-split.md) para el detalle
# completo, incluida la alternativa de un split agrupado que se consideró y
# se descartó por complejidad innecesaria.
#
# Antes de seguir, dos comprobaciones de que estos duplicados no interactúan
# con otros problemas de calidad ya conocidos del dataset:

# %%
violations_raw = flag_biochemical_violations(raw)
overlap_violations = set(raw[violations_raw["db_gt_tb"]].index) & set(duplicated_rows.index)
overlap_null_ag = set(raw[raw["A/G Ratio"].isna()].index) & set(duplicated_rows.index)
print(f"Solapamiento duplicados x violación DB>TB: {len(overlap_violations)} filas")
print(f"Solapamiento duplicados x A/G Ratio nulo: {len(overlap_null_ag)} filas")

# %% [markdown]
# Cero solapamiento en ambos casos: los tres problemas de calidad
# (duplicados, violaciones bioquímicas, nulos de `A/G Ratio`) son
# independientes entre sí — no hay que resolver ningún conflicto de orden
# entre ellos.

# %% [markdown]
# ### Paso 2 — Reconstruir `A/G Ratio` (F4-R3)
#
# `A/G Ratio` es aritmética de otras dos columnas **del mismo paciente**:
# `ALB / (TP - ALB)`. Es una operación fila a fila — no usa ninguna
# estadística calculada sobre el conjunto — así que **no es fuga de datos**
# y puede aplicarse antes del split.

# %%
null_ag_idx = deduped[deduped["A/G Ratio"].isna()].index
print(f"Filas con A/G Ratio nulo tras deduplicar: {list(null_ag_idx)}")

reconstructed = reconstruct_ag_ratio(deduped)
print("\nValor reconstruido por fila:")
reconstructed.loc[null_ag_idx, ["Gender", "TP", "ALB", "A/G Ratio"]]

# %% [markdown]
# **Declarar la imprecisión.** `TP` y `ALB` vienen redondeados a un
# decimal, y ese redondeo se propaga a la división: no es una recuperación
# exacta del valor que hubiera medido el laboratorio, es la mejor
# estimación disponible. Sigue siendo notablemente mejor que imputar por la
# media — error absoluto mediano 0.031 frente a 0.153, medido sobre las
# filas con valor presente (cifra ya documentada en la Fase 3 de la
# Actividad 1) — pero no es perfecta, y así se declara.

# %% [markdown]
# ### Paso 3 — Marcar `TB`/`DB` bioquímicamente imposibles (F4-R4)
#
# `DB` (bilirrubina directa) es por definición una fracción de `TB`
# (bilirrubina total): `DB > TB` no puede ocurrir en una medición correcta.
# El CSV crudo trae estos valores tal cual se midieron — la decisión de
# tratarlos como faltantes es de la Fase 3 de la Actividad 1, y hay que
# volver a aplicarla aquí porque F4-R1 obliga a partir otra vez del crudo.

# %%
violations = flag_biochemical_violations(reconstructed)
violated_idx = reconstructed[violations["db_gt_tb"]].index
print("Filas con DB > TB (valores originales, antes de marcar):")
print(reconstructed.loc[violated_idx, ["TB", "DB", "Selector"]])

nulled = nullify_biochemical_violations(reconstructed)
print(f"\nNulos totales tras marcar TB/DB: {int(nulled.isna().sum().sum())}")
nulled.loc[violated_idx, ["TB", "DB"]]

# %% [markdown]
# Estos 6 `NaN` (`TB` y `DB` de 3 filas, las tres de la clase mayoritaria
# `Selector=1`) se **dejan sin imputar aquí a propósito**. Imputar por
# mediana usa una estadística calculada sobre el grupo — si se calculara
# sobre el dataset completo antes del split, la mediana de test ya habría
# influido en un valor de entrenamiento. Por eso la imputación va **dentro**
# del `Pipeline` de la Fase 5, ajustada solo con los datos de entrenamiento
# en cada partición de validación cruzada.

# %% [markdown]
# ### Otras limpiezas heredadas de la Actividad 1 (F4-R4)
#
# Un hecho de calidad de datos más, documentado en detalle en
# `docs/data_dictionary.md`, que este dataset hereda tal cual y no se
# vuelve a re-derivar aquí: **censura de edad en 90**. El propio proveedor
# del dataset documenta que "cualquier paciente cuya edad superara 89 años
# se registró con edad '90'" — es un límite de captura, no una edad real.

# %%
n_top_coded = int((deduped["Age"] == AGE_TOP_CODE).sum())
print(f"Pacientes con Age={AGE_TOP_CODE} (top-coded, no es una edad real): {n_top_coded}")

# %% [markdown]
# ### Paso 4 — Codificar `Gender` y `Selector` (F4-R5)
#
# Se fija aquí, explícitamente, cuál es la clase positiva: **`Selector=1`
# (enfermo) es la clase positiva** — de eso depende el signo de todas las
# métricas de recall/FNR de las fases siguientes (ambigüedad A4 del PRD).

# %%
encoded = encode_selector(encode_gender(nulled))
print("Mapeo de Gender:  {'Male': 0, 'Female': 1}")
print("Mapeo de Selector: {1 (enfermo) -> 1, 2 (sano) -> 0}")
encoded[["Gender", "Selector"]].head(3)

# %% [markdown]
# `load_modeling_data()` en `src/data.py` encadena estos cuatro pasos en
# exactamente este orden. Verificación de que produce lo mismo que se armó
# aquí a mano, paso a paso:

# %%
modeling_df = load_modeling_data()
assert modeling_df.equals(encoded), "load_modeling_data() debe reproducir la cadena manual"
print(f"Dataset de modelado: {modeling_df.shape[0]} filas x {modeling_df.shape[1]} columnas")
print(f"Nulos totales: {int(modeling_df.isna().sum().sum())} (solo TB/DB de las 3 filas de la Trampa 1)")

# %% [markdown]
# ### Trampa 3, en vivo — por qué codificar va antes del split
#
# El orden de codificación de `Selector`/`Gender` no es un detalle
# cosmético: `train_test_split(..., stratify=clave)` recorre las clases
# únicas de `clave` en el orden en que aparecen, y ese orden depende de qué
# valores tiene la columna. Con la misma semilla, estratificar por
# `Selector` **crudo** (`{1, 2}`) y por `Selector` **ya mapeado** (`{1, 0}`)
# produce composiciones de test distintas — se muestra a continuación, antes
# de fijar el split oficial.

# %%
test_size = params["split"]["test_size"]
seed = params["seed"]

_, test_raw_key = train_test_split(
    modeling_df.index,
    test_size=test_size,
    random_state=seed,
    stratify=deduped.loc[modeling_df.index, "Selector"],
)
_, test_mapped_key = train_test_split(
    modeling_df.index, test_size=test_size, random_state=seed, stratify=modeling_df["Selector"]
)


def _mujeres_sanas(idx):
    subset = modeling_df.loc[idx]
    mujeres = int((subset["Gender"] == 1).sum())
    sanas = int(((subset["Gender"] == 1) & (subset["Selector"] == 0)).sum())
    return mujeres, sanas


mujeres_raw, sanas_raw = _mujeres_sanas(test_raw_key)
mujeres_mapped, sanas_mapped = _mujeres_sanas(test_mapped_key)
print(f"Estratificando por Selector CRUDO {{1,2}}:    test -> {mujeres_raw} mujeres, {sanas_raw} sanas")
print(f"Estratificando por Selector YA MAPEADO {{1,0}}: test -> {mujeres_mapped} mujeres, {sanas_mapped} sanas")

# %% [markdown]
# Misma semilla, mismo `test_size`, mismas 570 filas — y aun así distintas
# filas exactas en test, porque cambió el orden en que `sklearn` procesa las
# clases de la estratificación. **Por eso `load_modeling_data()` codifica
# `Gender`/`Selector` antes de que cualquier función de split los reciba**:
# de lo contrario, el split dejaría de ser reproducible con solo reordenar
# dos líneas de código.

# %% [markdown]
# ## T3 — Split train/test (F4-R6..R10)
#
# Dos variantes:
#
# - **Comparación** — estratifica solo por `Selector`.
# - **Oficial (F4-R8)** — estratifica por la clave compuesta
#   `Selector x Gender`. Es la que se usa en el resto del proyecto: en un
#   proyecto de equidad, dejar que el sexo se reparta como caiga es
#   exactamente el tipo de decisión "neutra" que puede introducir sesgo sin
#   querer.

# %%
train_idx_cmp, test_idx_cmp = split_selector_only_for_comparison(modeling_df, params)
train_idx, test_idx = split_train_test(modeling_df, params)

print(f"Split oficial: train={len(train_idx)}, test={len(test_idx)}")


def _resumen(idx, nombre):
    mujeres, sanas = _mujeres_sanas(idx)
    print(f"{nombre}: test -> {len(idx)} filas, {mujeres} mujeres, {sanas} sanas")


_resumen(test_idx_cmp, "Comparación (solo Selector)      ")
_resumen(test_idx, "Oficial (Selector x Gender)      ")

# %% [markdown]
# La variante oficial reparte mejor el subgrupo que más importa para la
# auditoría de equidad de la Fase 9 (mujeres sanas, el grupo más chico del
# dataset) — se ve arriba, con las cifras reales de esta ejecución. Por eso
# es la variante que se congela para el resto del proyecto.

# %% [markdown]
# ### F4-R7 — Verificación de que las proporciones se conservan

# %%
def _proporciones(idx=None):
    subset = modeling_df if idx is None else modeling_df.loc[idx]
    return pd.Series(
        {
            "Selector=1 (enfermo)": subset["Selector"].mean(),
            "Gender=1 (mujer)": subset["Gender"].mean(),
        }
    )


tabla_proporciones = pd.DataFrame(
    {
        "Completo": _proporciones(),
        "Train": _proporciones(train_idx),
        "Test": _proporciones(test_idx),
    }
).round(4)
tabla_proporciones

# %%
diffs = (tabla_proporciones[["Train", "Test"]].sub(tabla_proporciones["Completo"], axis=0)).abs()
print("Máxima desviación absoluta respecto del dataset completo:", diffs.max().max())
assert (diffs <= 0.02).all().all(), "Alguna proporción se desvía más de ±2pp"

# %% [markdown]
# Las proporciones de clase y de sexo se mantienen dentro de ±2 puntos
# porcentuales entre el dataset completo, train y test (verificado también
# por `tests/test_fase4_split.py::test_estratificacion_clase` y
# `test_estratificacion_sexo`).

# %% [markdown]
# ### F4-R10 — Por qué `test_size=0.2`
#
# Con 570 filas y 24.6% de mujeres, un test del 20% deja las cifras que se
# vieron arriba: 114 filas, 28 mujeres, 10 de ellas sanas. Es una muestra
# **insuficiente para medir con confianza una tasa de falsos negativos por
# sexo** — una sola predicción distinta movería el resultado varios puntos
# porcentuales sobre una base de 10 personas. Subir `test_size` reduciría
# aún más el train (456 filas ya es ajustado para Grid Search); bajarlo
# reduciría más el test. 0.2 es el punto convencional que no sacrifica
# ninguno de los dos extremos más de lo necesario — y precisamente por esta
# limitación, la Fase 9 no calculará la brecha de equidad sobre este split
# único, sino por validación cruzada repetida sobre las 142 mujeres
# completas del dataset (`RepeatedStratifiedKFold`, F9-R2).

# %% [markdown]
# ### Verificación en vivo de la fuga de duplicados (§5.3)
#
# Con el dataset ya deduplicado, ninguna fila idéntica puede cruzar el
# split — se comprueba directamente, no solo se afirma.

# %%
signatures = row_signature(modeling_df)
train_sigs = set(signatures.loc[train_idx])
test_sigs = set(signatures.loc[test_idx])
print(f"Firmas de fila compartidas entre train y test: {len(train_sigs & test_sigs)}")
assert train_sigs.isdisjoint(test_sigs)

# %% [markdown]
# ### Persistencia del split (F4-R9)
#
# Se guardan los índices en `data/processed/split_indices.json` para que
# toda fase posterior cargue exactamente este split, sin volver a
# calcularlo.

# %%
indices_path = PROJECT_ROOT / params["split"]["indices_path"]
save_split_indices(
    train_idx,
    test_idx,
    indices_path,
    seed=params["seed"],
    test_size=params["split"]["test_size"],
    stratify_by=params["split"]["stratify_by"],
)
print(f"Guardado en {indices_path.relative_to(PROJECT_ROOT)}")

# %% [markdown]
# ## Resumen
#
# | | Valor |
# |---|---|
# | Crudo | 583 x 11 |
# | Tras deduplicar (F4-R2) | 570 filas |
# | A/G Ratio reconstruido (F4-R3) | 4 valores, fila a fila, no es fuga |
# | TB/DB marcados como nulos (F4-R4) | 6 NaN, dejados para el Pipeline de Fase 5 |
# | Clase positiva (F4-R5) | Selector=1 (enfermo) |
# | Split (F4-R6, F4-R8) | train=456, test=114, estratificado por Selector x Gender |
# | Fuga de duplicados | 0 filas cruzan el split |
#
# **Las tres trampas de esta fase, resueltas:**
#
# 1. Los `NaN` de `TB`/`DB` no vienen en el crudo — se re-marcaron aquí
#    (Paso 3) usando la misma regla bioquímica de la Fase 3 de la
#    Actividad 1.
# 2. Se dedujo **antes** de dividir (Paso 1), evitando que pares de filas
#    idénticas memorizadas crucen el split — verificado en vivo más arriba,
#    no solo declarado.
# 3. `A/G Ratio` se reconstruyó antes del split porque es aritmética fila a
#    fila; `TB`/`DB` se dejaron sin imputar porque su imputación usa una
#    estadística de grupo — esa distinción, no la mera cronología, es el
#    criterio que separa lo que puede ir antes del split de lo que no.
#
# **Sigue:** Fase 5 — `Pipeline` de imputación, escalado y balanceo (SMOTE),
# construido sobre este mismo split congelado.
