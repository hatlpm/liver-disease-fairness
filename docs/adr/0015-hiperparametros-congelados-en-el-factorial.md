# 15. Hiperparámetros congelados de la Fase 6 en el factorial de la Fase 8, vía `set_params`, no reajustados por celda

**Fecha:** 2026-08-16
**Estado:** Aceptada

## Contexto

La Fase 8 (T8, C4 -- 25%) tiene que aislar el efecto del preprocesamiento
({MinMax, Z-Score} x {sin SMOTE, SMOTE}) sobre las métricas de los 5
algoritmos. `params["tuning_baseline"]` (Z-Score + SMOTE) ya fijó, en la Fase
6, los hiperparámetros de cada modelo -- incluido `selector__k`, el tamaño
del subconjunto de variables -- mediante `GridSearchCV` ejecutado **una sola
vez**, bajo esa única configuración.

Si el factorial de la Fase 8 volviera a correr `GridSearchCV` dentro de cada
una de las 20 celdas, el efecto que T8 pregunta (¿qué impacto tiene el
preprocesamiento?) quedaría confundido con el efecto de volver a buscar
hiperparámetros: una diferencia entre `minmax` y `zscore` en la métrica final
ya no se podría atribuir al escalado en sí, porque también podría deberse a
que `GridSearchCV` encontró, para cada escalador, una combinación de
hiperparámetros distinta y no comparable.

## Decisión

Los `best_params_` de los 5 modelos (obtenidos bajo `tuning_baseline`) se
recalculan **una vez** al principio del notebook
(`src/factorial.py::collect_frozen_best_params`, mismo patrón que
`07_evaluacion.ipynb`: no se persisten a disco, se recomputan por sesión) y
se aplican, sin cambios, a cada una de las 20 celdas mediante
`pipe.set_params(**best_params)`. Ninguna función de `src/factorial.py`
importa ni instancia `GridSearchCV`
(`tests/test_fase8_factorial.py::test_hiperparametros_no_reajustados` lo
verifica sobre el código fuente).

## Consecuencias

- **Limitación declarada, no oculta:** el `k` óptimo de `SelectKBest`, o
  cualquier otro hiperparámetro, ajustado bajo Z-Score+SMOTE, podría no ser
  el óptimo bajo MinMax. `08_factorial.ipynb` incluye, aparte de la tabla
  oficial, un chequeo de sensibilidad que reajusta un solo modelo (KNN) bajo
  MinMax+SMOTE para medir cuánto importa esto en la práctica -- sin cambiar
  ninguna celda de la tabla de F8-R1/R2.
- **Consecuencia directa e inesperada de esta decisión:** dentro de un
  pipeline con hiperparámetros congelados, el único paso que sigue siendo
  sensible a la escala relativa entre variables para `gaussian_nb` y
  `decision_tree` -- estimadores en sí mismos invariantes al escalado
  afín -- es `SMOTE`, porque elige vecinos por distancia euclídea. Medido
  sobre este proyecto (CV en train, variante oficial sin `Gender`,
  `|balanced_accuracy(MinMax) - balanced_accuracy(Z-Score)|`):

  | modelo | sin SMOTE | con SMOTE |
  |---|---|---|
  | `gaussian_nb` | 0.000000 | 0.000000 |
  | `decision_tree` | 0.000000 | 0.020427 |
  | `svm` | 0.000000 | 0.053219 |

  `decision_tree` deja de ser invariante al escalado en presencia de SMOTE;
  `gaussian_nb` no, porque agrega media/varianza por variable en vez de
  depender de las posiciones sintéticas individuales que SMOTE genera de
  forma distinta bajo cada escalador (verificado directamente comparando los
  pacientes sintéticos generados bajo ambos escalados,
  `src/factorial.py::compare_smote_synthetic_across_scalers`: ~47% de los
  194 sintéticos son pacientes distintos entre `MinMax` y `Z-Score`). Esto
  matiza la afirmación incondicional de F6-R7 ("Naive Bayes y árboles son
  insensibles al escalado") -- registrado como Loop G en
  `docs/CHANGELOG_iteraciones.md`, sin editar `06_modelos.ipynb` (fase
  cerrada; decisión explícita del usuario de no reabrirla).
- Si una fase futura necesitara medir el efecto conjunto de preprocesamiento
  + reajuste de hiperparámetros (una pregunta distinta a la de T8), debería
  hacerlo con un diseño experimental nuevo que lo declare explícitamente, no
  reutilizando esta tabla.
