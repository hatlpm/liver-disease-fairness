# 16. Selector de variables blindado en la Fase 9, local a `src/fairness.py`, sin modificar `build_pipeline`

**Fecha:** 2026-08-16
**Estado:** Aceptada

## Contexto

La Fase 6 dejó traspasado a la Fase 9 un riesgo medido pero no resuelto
(`AGENTS.md`, "Riesgo conocido para la Fase 9"): `SelectKBest(f_classif)`
calcula un F-estadístico basado en varianza entre grupos. Si un pliegue de
entrenamiento de `RepeatedStratifiedKFold(5, n_repeats=10)` no contiene
ninguna de las 3 filas con `TB`/`DB` nulos (índices 246, 261, 279), el
indicador de nulos (`MissingIndicator(features="all")`, ADR-0011) queda
constante en ese pliegue -- varianza 0, el cálculo da `0/0`, `SelectKBest`
lo descarta con un score `NaN`, **en silencio**. Medido: 1 de los 50
pliegues de la CV de equidad (0 de 5 en el `StratifiedKFold(5)` de las
Fases 6-8, que nunca dispara el caso).

El PRD de la Fase 9 planteaba dos salidas: declararlo como limitación
cuantificada, o blindar excluyendo las columnas del indicador de la
competencia de `SelectKBest`. La segunda elimina el problema de raíz, pero
`build_pipeline` (`src/pipelines.py`) es compartido por las Fases 5-8, ya
cerradas y con números reportados (`balanced_accuracy` = 0.7234 del
ganador, entre otros). Modificar el selector global habría cambiado qué
variables compiten por el `top-k` en **todas** las celdas de `GridSearchCV`
y del factorial de esas fases -- no solo en el 2% de pliegues donde el
problema ocurre -- obligando a re-verificar resultados ya entregados por
un riesgo que ni siquiera las afecta.

## Decisión

Blindar, pero de forma **local a la Fase 9**: `build_pipeline(...,
selector=...)` ya acepta cualquier transformer de `sklearn` en el paso
`"selector"` -- el mismo mecanismo genérico que usan las Fases 6A-8. En vez
de pasar `SelectKBest(f_classif)` como hacen esas fases,
`src/fairness.py::build_shielded_selector` construye un `ColumnTransformer`
de dos ramas: `SelectKBest(f_classif, k)` compite solo entre las columnas
que no son del indicador (vía `make_column_selector` con un patrón regex
negativo sobre el prefijo `"tb_db_indicator__"`); las 2 columnas del
indicador pasan siempre, sin competir nunca por varianza. Se pasa como
`selector=` de `build_pipeline`, exactamente igual que cualquier otro
selector -- **`src/pipelines.py` no cambia una línea**.

## Consecuencias

- Blast radius cero sobre las Fases 5-8: `src/pipelines.py`, `src/models.py`
  y `src/factorial.py` quedan intactos. Los 46 tests que existían antes de
  la Fase 9 siguen verdes sin cambios (verificado:
  `pytest tests/ -q` con y sin `src/fairness.py`/`tests/test_fase9_fairness.py`
  presentes da el mismo resultado para esos 46).
- `tests/test_fase9_fairness.py::test_indicador_constante_resuelto`
  reproduce el pliegue límite exacto (train sin las filas 246/261/279) y
  verifica que el selector plano descarta el indicador (con los avisos
  `UserWarning`/`RuntimeWarning` esperados) mientras el blindado lo
  conserva siempre.
- 🔴 **Efecto secundario real, no cosmético: el modelo auditado en la
  Fase 9 no es bit a bit el mismo que el ganador declarado en la Fase 7.**
  El `k` congelado de `best_params_` se eligió compitiendo entre TODAS las
  columnas (indicador incluido) -- 11 en la variante oficial sin `Gender`.
  Bajo el selector blindado, la rama de `SelectKBest` compite solo entre
  las columnas no-indicadoras (9 en esa variante), así que ese mismo `k`
  (10) selecciona las 9 + las 2 del indicador (siempre presentes) = **11
  columnas**, no las 10 que `GridSearchCV` había elegido originalmente
  (9 de 11, mezclando numéricas e indicador según su score). El modelo de
  la Fase 9 tiene una variable más que el de la Fase 7
  (`balanced_accuracy` CV = 0.7234, cifra de ese modelo de 10, no de este).
  No invalida la auditoría -- ambos modelos son variantes razonables del
  mismo pipeline -- pero **debe declararse explícitamente**, no asumirse
  equivalente. `notebooks/09_fairness.ipynb` lo explica antes de F9-R1 y
  añade un control con `run_fairness_variant(...,
  selector_factory=lambda k: SelectKBest(f_classif, k=k))` que reproduce
  sin blindar el modelo real de 10 variables de la Fase 7: la brecha
  resultante (+20.45 pp, p=0.0027) es de magnitud comparable o mayor que
  con el selector blindado (+19.08 pp, p=0.0043) -- el blindaje es
  conservador, no fabrica el hallazgo.
- **Higiene: `k` se acota, no solo se deja que `sklearn` lo maneje.**
  `_default_shielded_selector` (usado por defecto en
  `run_fairness_variant`) acota `k = min(k, columnas no-indicadoras
  disponibles)` antes de construir el selector -- mismo resultado que
  dejar que `SelectKBest` seleccione todas al superar `k`, pero sin el
  `UserWarning` ("k=10 is greater than n_features=9") repetido en cada uno
  de los 50 ajustes de cada variante (100 avisos idénticos en la tabla de
  F9-R4/F9-R5, que podrían tapar un aviso real en una fase futura). El
  `selector_factory` del control anterior recibe el `k` **sin acotar** a
  propósito: reproduce el comportamiento real (no blindado) de las
  Fases 6A-8, acotarlo ahí distorsionaría el control.
- Si una fase futura necesitara este mismo blindaje fuera de la Fase 9
  (p. ej. si la Fase P sirviera predicciones bajo una variante de datos que
  también disparara el caso límite), debería decidir explícitamente si
  aplicarlo también a `build_pipeline` -- esta decisión no lo hace por
  adelantado.
