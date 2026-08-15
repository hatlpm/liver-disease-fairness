# 11. Indicador de nulos vía `ColumnTransformer` de dos ramas, no `SimpleImputer(add_indicator=True)`

**Fecha:** 2026-08-15
**Estado:** Aceptada

## Contexto

ADR-0007 fija que `TB`/`DB` se imputan por mediana dentro del `Pipeline`,
con un indicador binario que marque las filas imputadas (`add_indicator=True`,
tal como lo nombra literalmente ese ADR y `params.yaml`). Al implementar la
Fase 5 apareció un problema con el mecanismo concreto que hay detrás de ese
parámetro: `SimpleImputer(add_indicator=True)` construye el indicador con
`MissingIndicator(features="missing-only")` por debajo, que solo genera una
columna para una variable si **el pliegue de datos que recibe `.fit()`**
tiene al menos un nulo en ella.

Con el split actual (train = 456, 3 filas nulas) y `StratifiedKFold(5)` de
la Fase 6, cada pliegue de entrenamiento se queda con 2 o 3 de esas 3 filas
(verificado en `notebooks/05_pipeline.ipynb`: `[3, 3, 2, 2, 2]`), así que
hoy el número de columnas de salida no varía. Pero la Fase 9 usa
`RepeatedStratifiedKFold(5, n_repeats=10)` — 50 ajustes en vez de 5 — y ahí
la probabilidad de que algún pliegue de entrenamiento no contenga ninguna de
las 3 filas nulas deja de ser despreciable. Si eso ocurre, `add_indicator=True`
no generaría la columna del indicador para ese pliegue, y el ancho de la
salida cambiaría silenciosamente entre ajustes — rompiendo cualquier paso
posterior (selección de variables, el propio estimador) que asuma un número
fijo de columnas.

## Decisión

Construir el indicador en una rama independiente del `ColumnTransformer`,
con `MissingIndicator(features="all")` en vez del parámetro `add_indicator`
de `SimpleImputer`:

```python
ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                           ("scaler", scaler)]), NUMERIC_COLS),
    ("tb_db_indicator", MissingIndicator(features="all"), ["TB", "DB"]),
], remainder="passthrough")
```

`features="all"` genera siempre las 2 columnas del indicador (`TB`, `DB`),
tenga o no nulos el pliegue concreto que se le pasa. Verificado en
`tests/test_fase5_pipeline.py::test_ancho_estable_entre_pliegues` y en vivo
en el notebook, incluido el escenario límite forzado (entrenar sin ninguna
de las 3 filas nulas): el ancho de salida se mantiene en 12 columnas en
todos los casos.

Esto **refina** el mecanismo de ADR-0007, no lo contradice: la intención
(imputar por mediana, marcar qué filas se imputaron, no escalar esa marca)
sigue siendo la misma. Por eso este documento no marca a ADR-0007 como
superado — sigue vigente en todo lo demás (reconstrucción de `A/G Ratio`,
decisión de conservar las 3 filas, análisis de sensibilidad `F8-R5`).

## Consecuencias

- El ancho de la salida del preprocesamiento (12 columnas: 9 numéricas + 2
  del indicador + `Gender` sin tocar) queda garantizado estable entre
  pliegues, incluida la Fase 9 con 50 ajustes — no solo hoy, con 5.
- `params.yaml` sigue declarando `imputer.add_indicator: true` como
  descripción de la intención (imputar CON indicador), pero
  `build_pipeline()` no lee ese campo para construir nada: siempre añade el
  indicador vía `MissingIndicator(features="all")`. Si en el futuro
  `add_indicator` pasara a `false` en `params.yaml`, el código actual lo
  ignoraría en vez de fallar — riesgo aceptado por ahora, documentado aquí
  para quien retome el trabajo.
- La columna del indicador nunca se escala (rama separada del
  `ColumnTransformer`, fuera de la rama que contiene el escalador), cumple
  F5-R3.
