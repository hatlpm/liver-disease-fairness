# 14. La tabla de T7 reporta a umbral 0.5; el punto de cribado explorado no la reemplaza

**Fecha:** 2026-08-16
**Estado:** Aceptada

## Contexto

La nota de traspaso de la Fase 6 (`notebooks/06_modelos.ipynb`) advirtió que,
al umbral por defecto (0.5), el mejor modelo por CV (`logistic_regression`)
tiene `recall = 0.60` sobre train: 4 de cada 10 pacientes enfermos de esa
muestra quedarían clasificados como sanos. En un contexto de cribado, donde
un falso negativo es un enfermo enviado a casa sin tratamiento y un falso
positivo es solo una prueba de más, ese punto de corte no debería aceptarse
en silencio solo porque es el valor por defecto de `sklearn`.

Al mismo tiempo, F7-R5 exige que la tabla de T7 compare los 5 modelos bajo
un criterio común, y B6 del PRD exige que cualquier ajuste de umbral se haga
**solo sobre train/CV, nunca sobre el test** -- evaluar el test a varios
umbrales y quedarse con el que mejor se ve sería la misma fuga que ya
prohíbe la elección del modelo ganador (Trampa 2).

Ejecutado `threshold_curve_cv` sobre `logistic_regression` con
`cross_val_predict` en train (nunca en test), con
`screening_min_recall = 0.90` (`params.yaml`) el punto recomendado es
`threshold = 0.30` (recall 0.9323, precisión 0.7354, balanced_accuracy
0.5501) -- el mismo punto de operación que ya señalaba el enunciado del
encargo como "recall 0.93 y discriminación real", distinto del umbral 0.20
que solo reproduce al clasificador nulo.

## Decisión

Dos cosas separadas, no una:

1. **La tabla oficial de T7 reporta los 5 modelos a `threshold = 0.5`**
   (`params["metrics"]["decision_threshold"]`) -- el criterio común que hace
   la tabla comparable entre modelos, tal como pide §3.2 del PRD.
2. **El punto de cribado (`threshold = 0.30`, recall ≥ 0.90) queda declarado
   como recomendación de discusión**, calculado y documentado en
   `notebooks/07_evaluacion.ipynb`, pero **no sustituye automáticamente**
   ningún número de la tabla de T7 ni se propaga a `params.yaml` como nuevo
   `decision_threshold` por defecto.

Cualquier fase posterior (Fase 8, Fase 9, o la Fase P al construir el
tablero) que quiera *adoptar* el punto de cribado en vez del umbral 0.5 debe
hacerlo de forma explícita, citando este ADR, no heredarlo en silencio.

## Consecuencias

- La tabla de T7 se mantiene comparable y reproducible bajo una única
  convención (F7-R5), sin que la elección clínica del umbral la contamine.
- El punto de cribado recomendado es reproducible (misma semilla, misma CV)
  y queda disponible como insumo para quien deba tomar la decisión de
  producto -- pero permanece explícitamente **no vinculante** hasta que
  alguien lo adopte a propósito.
- Si una fase futura decide que `threshold = 0.30` (u otro valor) debe ser
  el umbral de producción, la forma correcta de hacerlo es un ADR nuevo que
  cambie `params["metrics"]["decision_threshold"]` y marque a este documento
  como superado en ese punto -- no editar `params.yaml` sin ese registro.
