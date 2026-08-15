# 6. Deduplicar antes del split, no después

**Fecha:** 2026-08-15
**Estado:** Aceptada

## Contexto

El ILPD tiene 13 pares de filas exactamente idénticas (26 de 583 filas). La
Actividad 1 decidió conservarlos: no se entrenaba ningún modelo, y no había
forma de saber si eran pacientes distintos con una analítica idéntica o el
mismo registro capturado dos veces — eliminarlos sin evidencia habría sido
una decisión arbitraria.

La Actividad 2 sí entrena modelos y los evalúa sobre un `train_test_split`.
Con `random_state=42`, **5 de los 13 pares caen a ambos lados del split**
cuando se divide sin deduplicar antes: el modelo ve una fila en
entrenamiento y la misma fila, byte a byte, en test. No generaliza sobre
ella — la memoriza — y su métrica de test queda inflada sin que eso se
refleje en ningún número visible. Medido sobre 20 semillas distintas, el
número de pares partidos nunca fue cero (mínimo 1, media 3.7, máximo 6): no
es un riesgo teórico de esta semilla en particular, es estructural mientras
el dataset conserve los duplicados y se use estratificación estándar.

Se consideró una alternativa: un split **agrupado**
(`GroupShuffleSplit`, usando la fila completa como clave de grupo) que
garantiza que ningún par se reparta entre train y test, sin necesidad de
eliminar ninguna fila. Se descartó por ser más compleja de auditar (dos
mecanismos de agrupación coexistiendo — sexo/clase por un lado, identidad de
fila por otro) sin ganar muestra efectiva: las filas duplicadas no aportan
información nueva al modelo, están repetidas.

## Decisión

Deduplicar con `drop_duplicates(keep="first")` **antes** de construir el
split, dentro de `src/data.py::deduplicate()`, que es el primer paso de
`load_modeling_data()`. El dataset de modelado pasa de 583 a 570 filas.

Esto es una **divergencia deliberada** respecto de la Actividad 1, no una
corrección de un error anterior: el criterio de la Actividad 1 (conservarlos
porque no se entrenaba nada) seguía siendo correcto para lo que esa fase
hacía. Aquí el riesgo cambia — entrenar y evaluar sobre las mismas filas
repartidas produce una fuga de datos garantizada — y por eso la decisión
cambia con él.

## Consecuencias

- Se pierde el 2.2% de la muestra cruda (13 filas de 583). Compensa: elimina
  una fuga que de otro modo infla toda la tabla comparativa de T7 sin que
  quede evidencia visible del motivo.
- El dataset de modelado (570 filas) es distinto al dataset usado para el
  EDA de la Actividad 1 (583 filas). Cualquier comparación de cifras entre
  ambos informes debe declarar explícitamente sobre cuál de los dos se
  calculó.
- `tests/test_fase4_split.py::test_ninguna_fila_identica_cruza_el_split`
  deja esta garantía verificada en CI, no solo documentada: si en el futuro
  alguien reordena `load_modeling_data()` y el paso de deduplicación deja de
  ejecutarse antes del split, ese test falla.
