# 12. Métrica de optimización de `GridSearchCV`: `balanced_accuracy` en vez de F1 con clase positiva mayoritaria

**Fecha:** 2026-08-15
**Estado:** Aceptada

## Contexto

`params.yaml` declaraba, desde la Fase 4/§11.2 del PRD, `metrics.optimize_for:
"f1"` con `pos_label: 1` como métrica de selección de hiperparámetros para la
Fase 6. Esa elección se escribió antes de entrenar ningún modelo, bajo el
supuesto implícito (habitual cuando se recomienda F1 para clases
desbalanceadas) de que la clase positiva es la minoritaria.

En este dataset no lo es: `Selector = 1` (enfermo, F4-R5) es la clase
**mayoritaria** (71.27% de train, ratio 2.49:1). Medido sobre train, el
clasificador nulo "todos enfermos" —que no mira ningún dato de entrada—
obtiene F1 = 0.8323 y accuracy = 0.7127. `GridSearchCV` optimizando por F1
selecciona, entre configuraciones empatadas o cercanas a ese comportamiento
degenerado, la que más se le parezca: la métrica no distingue "aprendió algo"
de "ignoró la clase minoritaria". El detalle completo, con las cifras
verificadas, está en `docs/CHANGELOG_iteraciones.md` § Loop F.

Alternativas consideradas, todas mencionadas como candidatas válidas en el
encargo de la fase:

- **`f1_macro`** — promedio no ponderado del F1 de cada clase. Corrige el
  problema (suelo nulo 0.4154, no 0.83), pero es menos intuitivo de explicar:
  requiere entender qué es un promedio macro de F1 y por qué la clase que
  nunca se predice aporta F1=0 al promedio.
- **`roc_auc`** — suelo nulo 0.50 también, pero mide la calidad del *ranking*
  de probabilidades, no del clasificador con el umbral 0.5 fijo que exige
  F7-R7/B6. Además obliga a `probability=True` en `SVC`, que activa una
  calibración interna por CV adicional y encarece considerablemente el ajuste
  sin necesidad, ya que aquí no hace falta la probabilidad, solo la
  clasificación en el umbral fijo.
- **`balanced_accuracy`** — media de la sensibilidad (recall) de cada clase
  por separado: `(recall₁ + recall₀) / 2`. Suelo nulo 0.50 (azar) en
  cualquier split, sin depender de qué clase se llame "positiva". Se
  construye directamente a partir de dos cantidades que ya son centrales en
  el proyecto (recall de la clase enferma y de la clase sana — esta última es
  la especificidad, y su complemento es la FNR que audita la Fase 9), lo que
  la hace más fácil de explicar a un usuario sin formación clínica ni
  estadística que macro-F1.

## Decisión

`params.yaml` → `metrics.optimize_for: "balanced_accuracy"`. Se usa como
`scoring=` de `GridSearchCV` en `src/models.py::fit_grid_search`. Verificado
empíricamente: los 5 algoritmos exigidos, ajustados por esta métrica, obtienen
`balanced_accuracy` de validación cruzada entre 0.67 y 0.72 (todos por encima
del suelo nulo 0.50) — y, como consecuencia esperada y deseada, su F1 de
validación cruzada cae entre 0.62 y 0.72, **por debajo** del suelo nulo de F1
(0.83): dejan de sobre-predecir la clase mayoritaria para también acertar en
la clase sana. Fijado en
`tests/test_fase6_modelos.py::test_scoring_supera_el_suelo_nulo` para que una
sesión futura no lea esa caída de F1 como una regresión.

`accuracy`, precisión, recall y F1 se **siguen reportando** en toda la fase
tal como exige T6/T7 del enunciado (F7-R2) — esta decisión cambia el criterio
de **selección** de hiperparámetros, no las métricas que se informan al
usuario.

## Consecuencias

- Los modelos que resulten "mejores" en la Fase 6 van a tener F1/accuracy más
  bajos que si se hubiera optimizado por F1 — es la corrección funcionando,
  documentada aquí y en el CHANGELOG para que no se lea como un defecto en
  una sesión futura sin este contexto.
- Todo lo que reutilice `params["metrics"]["optimize_for"]` en fases
  posteriores (Fase 7 evaluación, Fase 8 factorial, Fase 9 equidad) hereda
  esta métrica automáticamente, sin cambio de código — es exactamente el
  propósito de que viva en `params.yaml` y no literal en cada notebook.
- `average="binary"`/`pos_label=1` (B4 del PRD) se mantienen intactos para
  las métricas que sí se reportan (precisión, recall, F1): esta decisión no
  los toca, solo cambia qué métrica decide `best_params_`.
- Si una fase futura necesitara optimizar por una métrica sensible al umbral
  de decisión distinta de 0.5 (p. ej. maximizar recall con un piso de
  precisión), esta decisión debería revisarse con un ADR nuevo que marque
  este como superado — no editarlo.
