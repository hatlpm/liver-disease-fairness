# 7. Tratamiento de nulos: reconstrucción determinista + imputación con indicador

**Fecha:** 2026-08-15
**Estado:** Aceptada

## Contexto

El dataset de modelado de la Fase 4 tiene dos fuentes de valores faltantes,
de naturaleza distinta, y tratarlas con el mismo método sería un error:

1. **`A/G Ratio`** tiene 4 nulos en el CSV crudo (filas 209, 241, 253, 312).
   Es una variable **derivada** de otras dos columnas del mismo paciente:
   `ALB / (TP − ALB)`. El valor faltante no representa una medición perdida
   — puede reconstruirse exactamente con aritmética sobre datos que sí están
   presentes en esa misma fila.
2. **`TB`/`DB`** no tienen nulos en el crudo, pero 3 filas violan la
   restricción bioquímica `DB ≤ TB` (bilirrubina directa, fracción de la
   total) con valores imposibles (`TB=1.8/DB=9.0`, `TB=1.5/DB=7.0`,
   `TB=1.0/DB=1.4`). Son errores de medición o captura, no valores válidos:
   se tratan como faltantes, pero a diferencia de `A/G Ratio` no hay ninguna
   fórmula que los reconstruya a partir de otras columnas — son mediciones
   genuinamente perdidas.

## Decisión

**`A/G Ratio`: reconstrucción determinista, antes del split.**
`ALB / (TP − ALB)` es aritmética fila a fila — no usa ninguna estadística
calculada sobre el conjunto de datos — así que aplicarla antes del split no
es fuga. Medido sobre las 579 filas con valor presente, el error absoluto
mediano de esta fórmula (0.031) es 4.9x menor que imputar por la media
(0.153, el método que usó tanto la Fase 3 de la Actividad 1 como el paper
original del dataset). Se declara la imprecisión: `TP` y `ALB` vienen
redondeados a un decimal, ese redondeo se propaga a la división, y la
correlación entre el valor registrado y la fórmula reconstruida es 0.85, no
1.0 — es derivada por construcción, pero el redondeo impide recuperarla de
forma perfecta.

**`TB`/`DB`: se conservan, marcadas como nulas, e imputadas dentro del
`Pipeline` de la Fase 5** (mediana, `add_indicator=True`), no en esta fase.
Decisión del usuario sobre la alternativa de eliminar las 3 filas. Se
blinda con tres condiciones ya fijadas para la Fase 5:

1. Imputación por **mediana**, no por media — ambas variables tienen
   asimetría > 3.
2. `add_indicator=True` — una columna binaria marca qué filas fueron
   imputadas, para que el modelo distinga un valor medido de uno
   reconstruido.
3. Análisis de sensibilidad obligatorio (`F8-R5`): repetir el mejor modelo
   sin esas 3 filas y confirmar que las conclusiones no cambian.

La razón de que la imputación NO ocurra en esta fase, a diferencia de la
reconstrucción de `A/G Ratio`, es el criterio que distingue ambos casos:
imputar por mediana usa una estadística calculada sobre un **grupo** de
filas. Si esa mediana se calculara sobre el dataset completo antes del
split, la mediana de test ya habría influido en el valor imputado de una
fila de entrenamiento — es fuga. Calculada dentro del `Pipeline`, en cambio,
se ajusta solo con las filas de entrenamiento de cada partición de
validación cruzada.

## Consecuencias

- El dataset que sale de la Fase 4 (`load_modeling_data()`) tiene cero
  nulos salvo los 6 de `TB`/`DB` (3 filas x 2 columnas), dejados
  intencionalmente para el `Pipeline`.
- Las 3 filas de `DB>TB` son todas `Selector=1` (clase mayoritaria): ni
  imputarlas ni eliminarlas afecta la representación de la clase
  minoritaria.
- `tests/test_fase4_split.py::test_reconstruccion_ag_ratio` y
  `test_tb_db_marcados_como_nulos` verifican, con los valores exactos, que
  ambas reglas se aplicaron y que ninguna de las dos se confundió con la
  otra (que no se haya imputado `TB`/`DB` aquí, y que sí se haya
  reconstruido `A/G Ratio`).
