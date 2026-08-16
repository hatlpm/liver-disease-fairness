# 9. `src/` es la fuente de verdad; los notebooks narran y se emparejan con `jupytext`

**Fecha:** 2026-08-16
**Estado:** Aceptada

## Contexto

La Actividad 1 escribió la lógica dentro de los notebooks: `02_eda.ipynb`,
`03_preprocessing.ipynb` y `act1_anexo.ipynb` contienen el cálculo además de la
narrativa. Funcionó porque nada se desplegaba: el entregable era un informe.

La Actividad 2 entrena modelos y los pone en producción (Fase P: pipeline
serializado, CI/CD, tablero Streamlit). Ahí el patrón deja de servir, por tres
razones distintas:

1. ***Training/serving skew*.** Si el notebook implementa el preprocesamiento y
   el tablero lo reimplementa, tarde o temprano divergen. El resultado típico es
   un modelo que funciona en el notebook y falla en producción, sin que nada
   avise. El código que se evalúa tiene que ser **el mismo objeto** que se
   despliega.
2. **Un `.ipynb` es ilegible en una revisión.** Es JSON con salidas
   incrustadas: un `git diff` de un notebook re-ejecutado son miles de líneas de
   metadatos y base64, aunque el cambio real sea una línea. Sin diffs legibles no
   hay revisión posible.
3. **La lógica dentro de un notebook no es testeable.** El PRD exige un archivo
   de tests por fase (§12) como red de seguridad frente a un agente sin memoria
   entre sesiones. No se puede importar una celda.

## Decisión

**La lógica vive en `src/`, una sola vez.** Los notebooks importan y narran; no
reimplementan. Concretamente:

- Toda función con lógica va en `src/` (`data.py`, `splitting.py`,
  `pipelines.py`, `models.py`, `evaluate.py`, `factorial.py`, `fairness.py`),
  con docstring NumPy y test.
- Los notebooks contienen **solo** narrativa, llamadas a `src/` y visualización.
- Cada notebook se empareja con un `.py` en formato `percent` vía **`jupytext`**,
  y se versionan ambos: el `.py` es el que se lee en un diff.
- Los parámetros no viven en el código sino en `params.yaml` (§11 del PRD).

### Excepción declarada: el anexo académico

`act2_anexo.ipynb` es el entregable que el enunciado pide como *"archivo Jupyter
Notebook que contenga las soluciones a las preguntas"*, y el criterio de
evaluación mira el código. Un anexo que solo importe de `src/` cumpliría esta
decisión pero dejaría al evaluador sin ver lo que tiene que calificar.

Por eso el anexo **sí lleva el código en línea**, en el orden literal T1→T8 del
enunciado, declarado explícitamente como **copia de presentación derivada de
`src/`**, no como una segunda implementación. Se blinda con un test que verifica
que reproduce los mismos números que los módulos de `src/`.

Es un artefacto de entrega académica y no forma parte del módulo de producción:
la Fase P no lo importa, no lo despliega y no depende de él.

## Consecuencias

- El mismo `Pipeline` que la Fase 7 evaluó es el que la Fase P serializará y el
  tablero cargará. Se elimina de raíz el *training/serving skew*.
- Los cambios de lógica son revisables: el `.py` emparejado muestra el diff real.
- Todo lo de `src/` es importable y testeable; los 53 tests actuales dependen de
  ello.
- **Coste asumido:** hay duplicación real en `act2_anexo.ipynb`. Es deliberada,
  está declarada en el propio anexo, y el test de equivalencia impide que las
  dos copias se separen en silencio. Si alguna vez divergen, el test falla.
- Los notebooks quedan más cortos y menos autoexplicativos leídos en aislamiento:
  hay que ir a `src/` para ver el cálculo. Se compensa con docstrings NumPy
  completos, que son la documentación de referencia del proyecto.
