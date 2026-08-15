# Consulta 5 — Límites biológicamente posibles para T8 (F3-R12)

**Fecha:** 2026-08-15
**Disparador:** auditoría metodológica externa. Los límites usados en
`03_preprocessing.ipynb`/`act1_anexo.ipynb` para decidir si los 84 atípicos
de T8 son *estadísticos* o *erróneos* (`TP` 2.0–12.0, `ALB` 0.5–6.5,
`A/G Ratio` 0.1–5.0, `Sgot` 1–20 000) no tenían ninguna fuente citada —
violación directa del protocolo de fundamentación clínica del proyecto
(`AGENTS.md`).

## CONSULTA + CONTEXTO

¿Existe literatura que documente el **rango analíticamente/clínicamente
reportable** (el techo real que un instrumento de laboratorio puede medir
antes de perder linealidad, o el valor más alto documentado en casos
clínicos reales) para `TP` (proteína total), `ALB` (albúmina), la razón
`A/G Ratio` y `Sgot`/AST? El objetivo es distinguir un valor **raro pero
posible** de uno **fuera de lo que cualquier instrumento o paciente real
podría producir**.

## Resultado

**`Sgot` (AST) — fuente encontrada y verificada.**

> Chang, S.-W., Lee, S.-G., Choi, H.-S., Song, E.-Y., Park, Y.-W., Lee, I.-A.
> (2007). *Study on Analytical and Clinically Reportable Ranges*. Korean
> Journal of Clinical Laboratory Science, 39(1), 31–36.

El estudio mide, con verificación de calibración e instrumentos reales de
laboratorio, hasta qué valor puede reportarse un resultado de AST antes de
que el ensayo pierda linealidad: **rango clínicamente reportable de
24–7 446 U/L**.

**Decisión:** se adopta **7 446 U/L** como techo para `Sgot`, en reemplazo
del 20 000 sin fuente. Es un límite **más estricto** que el anterior. El
valor máximo observado en el ILPD (`Sgot` = 4 929, fila 135) queda **por
debajo** de este techo, con margen — la conclusión "no es un valor
imposible" se sostiene, y ahora con una fuente real detrás.

> ⚠️ **Corrección durante la implementación (2026-08-15).** El primer intento
> usó también el **piso** de Chang et al. (24 U/L) como límite inferior
> "biológicamente posible". Es un error: 24 U/L es el mínimo verificado por
> *ese estudio de calibración de instrumento*, no un piso biológico — un AST
> de 10 U/L es perfectamente normal en una persona sana, y **10 pacientes del
> ILPD** tienen `Sgot` por debajo de 24 sin que eso sea, ni remotamente, un
> valor imposible. Usar 24 como piso habría marcado como "erróneos" a 124
> pacientes sanos. Se corrigió: el piso de `Sgot` vuelve a ser **1** (trivial
> — "no puede ser negativo ni cero" — sin necesidad de cita), y **solo el
> techo (7 446)** queda atribuido a Chang et al. (2007).

**`TP`, `ALB`, `A/G Ratio` — búsqueda sin resultado con la confianza que
exige el protocolo del proyecto.**

El mismo artículo (Chang et al. 2007) reporta también rangos para proteína
total y albúmina, pero solo se pudo acceder a resúmenes de dos fuentes
distintas que **se contradicen entre sí** en las cifras exactas (unidades
ambiguas, posible error de transcripción en al menos una de las dos), y no
fue posible acceder al texto completo del artículo (revista coreana de
2007, sin DOI localizado) para verificar la tabla original. Ninguna otra
búsqueda (OHDSI Data Quality Dashboard, guías de valores críticos/*panic
values*) produjo un valor consolidado y verificable para estas tres
variables.

**Decisión:** no se cita ninguna fuente para `TP`, `ALB` ni `A/G Ratio`.
Los límites usados para estas tres variables (`TP` 2.0–12.0, `ALB` 0.5–6.5,
`A/G Ratio` 0.1–5.0) se declaran explícitamente en el código y en el
informe como **una estimación razonada por el equipo, no una cita
clínica** — coherente con el estándar del proyecto de no presentar como
fuente lo que no se pudo verificar como tal.

## Impacto sobre T8 / F3-R12

La conclusión "cero valores imposibles" (§9.2 del informe) no cambia — con
el límite nuevo y más estricto de `Sgot`, el resultado es el mismo (0
valores fuera de rango). Lo que cambia es la **trazabilidad**: una de las
cuatro variables queda citada con fuente verificable; las otras tres quedan
correctamente etiquetadas como criterio del equipo, no como hecho
publicado.

## Búsquedas realizadas (registro, no todas fructíferas)

1. Rangos críticos/*panic values* de laboratorio para TP, ALB, AST —
   resultados genéricos, sin un valor consolidado citable.
2. Reportes de casos de AST extremo en rabdomiólisis/fallo hepático agudo
   — confirman valores de miles de U/L como clínicamente reales, pero
   ninguno cerca de 20 000; consistente con adoptar un techo más bajo.
3. OHDSI Data Quality Dashboard, checks `plausibleValueLow`/
   `plausibleValueHigh` — el mecanismo existe y es la referencia
   metodológica correcta (ya citada para Q8 en `Consulta_4.md`), pero los
   valores por defecto para estos analitos específicos no están
   documentados públicamente con el detalle necesario.
4. Chang et al. (2007), rango clínicamente reportable — única fuente que
   dio un número verificable y consistente, usada para `Sgot`.
