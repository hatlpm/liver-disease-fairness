# 3. Sin *holdout* de validación en las Fases 0–3; *golden set* se extrae más adelante del raw inmutable

**Fecha:** 2026-07-26
**Estado:** Aceptada

## Contexto

Antes de iniciar la Fase 2 (EDA) surgió la duda de si convenía apartar un
pequeño conjunto de filas (se propuso n=29, ≈5% de 583) **antes** de
cualquier análisis, para tener datos "nunca vistos" con los que más adelante
validar el modelo o el sistema (CDSS/agente, Fase 7).

Análisis de la propuesta:

- **Como conjunto de validación estadística**, 29 filas es insuficiente,
  sobre todo para el eje central del proyecto (auditoría de *fairness* por
  sexo): estratificado por `Gender` (75.6% M / 24.4% F), un holdout de 29
  filas contendría solo ≈7 mujeres. Con n=7, el error estándar de cualquier
  proporción (ej. FNR) es tan alto que una sola predicción distinta mueve la
  métrica ~14–37 puntos porcentuales — no permite concluir nada sólido sobre
  la brecha de sesgo que el proyecto busca auditar.
- **Como *golden set* cualitativo** para probar el agente/CDSS de la Fase 7
  con ejemplos nunca analizados individualmente, sí tiene sentido — pero no
  es necesario apartarlo ahora: `data/raw/` es inmutable (§7.3 del PRD) y las
  estadísticas agregadas del EDA/preprocesamiento no "memorizan" filas
  específicas. Cualquier fila del CSV original puede usarse como ejemplo
  nunca visto en el futuro sin haber sido contaminada por el EDA.
- Apartar filas ahora también entra en conflicto con el enunciado académico:
  T1–T8 se responden sobre "el dataset", que corresponde al ILPD completo de
  583 filas (igual que el paper de referencia); reportar un tamaño distinto
  sin ese apartado exigiría justificación adicional frente al evaluador.

## Decisión

No se aparta ningún *holdout*/*validation set* en las Fases 0–3. El *golden
set* de ejemplos para probar el agente (Fase 7) se extraerá más adelante,
directamente del CSV original en `data/raw/`, sin necesidad de reservarlo
de antemano.

La partición estadísticamente rigurosa (train/test, estratificada por
`Selector` y `Gender`, dimensionada para dar métricas de *fairness*
estables — orden de 15–20% de 583) queda para la Fase 4–6 (modelado,
fuera de este PRD), tal como ya establecía §9.3 del PRD.

## Consecuencias

- Las Fases 0–3 se mantienen dentro del alcance declarado en el PRD (§2.3):
  ningún `train/test split` ni preparación de datos "lista para modelar".
- El tamaño muestral reportado en T1–T8 coincide con el dataset ILPD
  conocido (583 filas), sin necesidad de explicar discrepancias.
- Queda pendiente, para cuando se especifique la Fase 4–6 en un PRD futuro,
  decidir el tamaño y la estrategia de estratificación del split real de
  modelado — este ADR deja la referencia de por qué 29 no es adecuado como
  punto de partida para esa decisión.
