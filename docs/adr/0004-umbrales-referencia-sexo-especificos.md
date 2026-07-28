# 4. Umbrales de referencia sexo-específicos como marco de interpretación clínica

**Fecha:** 2026-07-27
**Estado:** Aceptada

## Contexto

El EDA de la Fase 2 reportaba estadísticos descriptivos (medias, asimetría,
correlaciones) sin compararlos contra ningún **rango de referencia clínico**.
Sin ese marco, decir que "la mediana de `TB` es 1.0" no informa si eso es
normal o patológico, y las interpretaciones se quedan en descripción
estadística.

Al buscar rangos de referencia apareció una decisión no trivial: **para ALT
(`Sgpt`) los umbrales publicados difieren por sexo**, y la elección entre un
umbral único y umbrales diferenciados **cambia materialmente los resultados**
del análisis estratificado que es el eje del proyecto.

El equipo **no tiene formación clínica**, por lo que adoptar cualquier umbral
sin fundamento citable sería indefendible ante un evaluador.

## Decisión

**Adoptar los umbrales sexo-específicos de Prati et al. (2002) como marco
principal**, y usar el umbral unisex de 40 U/L como **término de comparación**
para un análisis de sensibilidad — no como criterio alternativo.

| Analito | Hombres | Mujeres | Fuente |
|---|---|---|---|
| ALT (`Sgpt`) — *healthy ULN* | 30 U/L | 19 U/L | Prati et al. 2002, Ann Intern Med, DOI 10.7326/0003-4819-137-1-200207020-00006 |
| ALT — guía clínica actual | 29–33 | 19–25 | ACG 2017 (Kwo et al.); AASLD 2023 (Rinella et al.) |

Los valores viven en `src/config.py` (`ALT_ULN_UNISEX`, `ALT_ULN_BY_SEX`),
nunca escritos a mano en un notebook.

**Protocolo de fundamentación adoptado.** Ningún umbral, criterio metodológico
ni afirmación clínica entra al proyecto sin una fuente citable. El método:
formular una consulta de investigación profunda con contexto explícito
(CONSULTA + CONTEXTO), ejecutarla, y **versionar la respuesta completa con sus
referencias en `docs/fuentes/`**. Esto hace cada afirmación auditable y es
parte del valor de portafolio del proyecto.

## Alternativas consideradas

**(a) Usar solo el umbral unisex de 40 U/L.** Descartada: es el criterio que
la literatura actual considera superado, y usarlo en solitario borraría
precisamente el fenómeno que el proyecto audita.

**(b) Usar solo los umbrales sexo-específicos.** Descartada: perdería el
contraste. El hallazgo del proyecto **no es** "tantas mujeres tienen ALT
elevado" sino "**la clasificación cambia mucho más para ellas al cambiar el
criterio**" — y eso exige ambos umbrales.

**(c) Buscar un rango validado para Andhra Pradesh.** Intentada y **fallida**.
El estudio multicéntrico C-RIDL/IFCC (Shah et al. 2018, n=512, 4 ciudades)
confirma que ALT, AST, ALP, albúmina y bilirrubina total **requieren partición
por sexo** en población india, pero los estudios regionales se contradicen: uno
de Mumbai halla ALT 36.0 (H) vs 23.6 (M), mientras dos de Karnataka **no**
encuentran diferencia significativa en ALT. **Ninguno cubre Andhra Pradesh.**

## Consecuencias

- Las interpretaciones de T3–T5 pueden comparar cada estadístico contra un
  rango de referencia con fuente.
- El análisis de sensibilidad de umbrales es reproducible y auditable: cambiar
  una constante en `config.py` regenera todos los números.
- **Limitación aceptada y declarada:** los umbrales de Prati proceden de
  donantes de sangre de Milán, no de población sur-asiática. La **dirección**
  del hallazgo (ALT normal más bajo en mujeres) es robusta en poblaciones
  occidentales y asiáticas; la **magnitud exacta** es incierta para esta
  cohorte. Debe aparecer como limitación explícita en el informe, no
  disimularse.
- **Limitación adicional:** para bilirrubina directa, proteína total y
  albúmina **no se identificó evidencia sexo-específica validada en población
  india**. Se declara como vacío de evidencia; no se inventan umbrales.
- El mismo criterio de partición se extiende a la **edad** para `Alkphos`
  (CLSI C28-A3 / CALIPER), documentado en el propio notebook.

## Alcance

Estos umbrales se usan **para interpretar y para el análisis de sensibilidad**,
nunca para reetiquetar `Selector` ni para construir una variable objetivo
alternativa. Hacerlo sería inventar un diagnóstico que nadie emitió y violaría
el alcance de las Fases 0–3.
