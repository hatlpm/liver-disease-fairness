# CHANGELOG de iteraciones CRISP-DM

Registro de los *loops* de retroalimentación entre fases (§9.1 del PRD): cada
vez que un hallazgo de datos obliga a revisar una fase anterior, se anota
aquí con fecha, disparador y decisión. Esto es distinto de `docs/adr/`
(decisiones de ingeniería/tooling) y de `reports/informe_act1.md`
(resultados académicos).

## 2026-07-26 — Loop A: EDA → Business Understanding

**Disparador:** en `02_eda.ipynb` (Fase 2), el análisis estratificado por
`Gender` mostró dos hallazgos no anticipados en el detalle de la Fase 1:

1. La tasa de diagnóstico positivo (`Selector == 1`) es **73.47% en hombres
   vs. 64.79% en mujeres** — una brecha de ~9 puntos porcentuales que no se
   puede atribuir, solo con EDA, a una población femenina más sana o a un
   patrón de sub-diagnóstico.
2. `Alkphos` (ALP) es la **única** enzima con media más alta en mujeres
   (302.3) que en hombres (286.8), rompiendo el patrón de todas las demás
   enzimas/bilirrubinas (más altas en hombres). Coincide con el hallazgo de
   Straw & Wu (2022) de que ALP y sexo ganan importancia al corregir la
   subrepresentación femenina.

**Decisión:** no se reformulan las preguntas de investigación de
`01_problem_framing.ipynb` (siguen siendo válidas tal como están), pero se
deja este hallazgo registrado como evidencia concreta que **refuerza** la
pregunta de investigación 3 (¿el desbalance de sexo puede inducir sesgo?) y
sienta una hipótesis específica y verificable para la futura Fase 5
(auditoría de *fairness*, fuera de este PRD): *"la brecha en tasa de
diagnóstico por sexo observada aquí es coherente con un patrón de
sub-diagnóstico en mujeres, no solo con una diferencia real de severidad."*
No se puede confirmar ni descartar con EDA — requiere el modelo y las
métricas de la Actividad 2.

**Impacto:** ninguno sobre el alcance de las Fases 0–3; el hallazgo queda
documentado para informar el diseño de la Fase 5 cuando se especifique.

---

## 2026-07-27 — Loop C: revisión clínica del EDA (Data Understanding ↺ sí misma)

**Disparador:** revisión del usuario sobre `02_eda.ipynb` ya cerrado, con tres
observaciones:

1. La estructura del dataset (T1) **no describía el significado de cada
   columna** — sin eso, los estadísticos de T3–T5 no son interpretables.
2. El rango de edad 4–90 no se había cuestionado: *"¿es coherente tener a
   alguien de 4 años, no es un problema de calidad?"*
3. **Ausencia total de lectura clínica.** El EDA cumplía la rúbrica pero no
   leía los resultados desde la función hepática, por lo que *"no está
   mostrando patrones ocultos, solo se está cumpliendo con la rúbrica"*.

**Metodología aplicada.** Como el equipo no tiene formación clínica, se adoptó
un protocolo explícito de fundamentación bibliográfica: formular consultas de
investigación profunda (CONSULTA + CONTEXTO), ejecutarlas contra fuentes
primarias, y **versionar las respuestas en `docs/fuentes/`** para que cada
umbral y criterio citado sea auditable. Se realizaron cuatro consultas
(rangos de referencia por sexo · ALP dependiente de edad · cociente De Ritis ·
restricciones bioquímicas y calidad de captura), complementadas con
verificación directa de la documentación oficial de UCI.

**Hallazgos nuevos:**

| # | Hallazgo | Evidencia |
|---|---|---|
| **Q8** | **3 filas con `DB > TB`** — violación bioquímica dura, invisible a la inspección univariada | La bilirrubina directa es una fracción de la total. En la literatura solo se produce por interferencia analítica, nunca por fisiología |
| **Q9** | **Age heaping** — índice de Whipple **163.3**, categoría "Tosca" (ONU) | 63% más personas en edades múltiplo de 5 de las esperables. Las edades se estimaron, no se leyeron de un documento |
| **Q7** | **Resuelto:** `Age = 90` es *top-coding*, no una edad | Doc. oficial UCI: *"any patient whose age exceeded 89 is listed as being of age '90'"* |
| — | La metadata de UCI declara "Missing values: No" pese a los 4 nulos reales | Contradicción documentada; se audita el dato, no la ficha |
| — | **ALP confundida por edad:** mediana 320 en menores vs ~200 en adultos | Rangos pediátricos 129–468 U/L según edad y sexo (CALIPER, Zierk 2017). CLSI C28-A3 exige particionar antes de aplicar Tukey |
| — | **Sensibilidad de umbral de ALT por sexo** | Al pasar de umbral unisex (40) a sexo-específico (30 H / 19 M): mujeres +49.3 pp vs hombres +15.6 pp |
| — | **De Ritis:** 18.0% >2, 36.3% <1, mediana 1.22 | Sugiere heterogeneidad no capturada por la etiqueta binaria — **sin bimodalidad clara**, formulado como hipótesis |

**Decisiones tomadas:**

1. **Q7 corrige una afirmación previa.** El PRD (§5.3, Q7) instruía reportar la
   censura de edad como *"verificada, no concluyente — no afirmar que existe un
   cap"*. La evidencia documental existía en la ficha de UCI y no se había
   localizado. **Ahora sí se afirma, con cita.**
2. **La edad de 4 años NO es un problema de calidad.** El dato es real y
   clínicamente coherente (la paciente de 7 años presenta un cuadro de
   hepatitis aguda internamente consistente). Lo que sí abre es un **problema
   de validez metodológica** por mezclar población pediátrica y adulta.
3. **Umbrales de referencia adoptados:** Prati et al. 2002, respaldados por
   ACG 2017 y AASLD 2023 → ver `docs/adr/0004-umbrales-referencia-sexo-especificos.md`.
4. **Tratamiento de las 3 filas `DB > TB`** (a ejecutar en Fase 3): marcar
   **ambas** columnas como faltantes, sin eliminar la fila, documentando la
   regla como violación de plausibilidad (marco de Kahn et al. 2016 / OHDSI).
5. **Estratificación por edad en T8**, como valor agregado que **se suma** al
   Tukey global obligatorio — nunca lo sustituye (§0.4 del PRD).

**Impacto:** los problemas de calidad pasan de **7 a 9**. Se añade
`docs/data_dictionary.md` (FE-R5, pendiente desde la Fase 0). Ningún cálculo
de T1–T5 se modifica: los números que responden a la rúbrica siguen siendo los
mismos, lo que cambia es su **interpretación**. No se entrena ningún modelo ni
se calcula ninguna métrica de clasificación — el alcance del PRD (§2.3) se
respeta íntegro.

**Honestidad metodológica registrada.** Dos conclusiones se atenuaron al
contrastarlas con las fuentes: (a) el argumento de heterogeneidad etiológica
vía De Ritis es **exploratorio, no probatorio** —el cociente refleja también el
estadio de la enfermedad—; (b) el efecto de la confusión ALP/edad resultó
**modesto en esta muestra** (4 vs 2 menores marcados), y se reporta así en vez
de exagerarlo.

---

## 2026-07-28 — Revisión de la Fase 2c: el clustering estaba midiendo outliers

**Disparador:** revisión del usuario sobre la primera versión de
`02c_eda_clustering.ipynb`. Dos observaciones, ambas confirmadas al
investigarlas:

1. Varios "clusters de severidad" tenían **1, 2 o 4 personas**, y sobre ellos
   se calculaban medianas y porcentajes ("100% diagnosticado" con n=1).
2. El notebook se **contradecía**: la tabla de honestidad decía que
   `60-120 · Female` (n=20) *"se reporta pero no se interpreta"*, y tres
   celdas antes lo destacaba como uno de los dos hallazgos principales de De
   Ritis — con una mediana calculada sobre 2 personas.

**Causa raíz (diagnosticada, no supuesta).** El Experimento 1 aplicaba
`StandardScaler` a las 9 variables **crudas**. Estandarizar centra y escala
pero **no cambia la forma**: con `Sgot` en *skewness* 10.5, el paciente más
extremo queda a ~17 desviaciones. Como Ward mide distancias euclidianas, el
mayor salto del dendrograma es el de **unir a esa persona con el resto**, y
`hierarchical_cluster_cut` —que corta exactamente en el mayor salto— la
aislaba como "cluster".

El propio notebook ya contenía el argumento en contra: el Experimento 2
justificaba `log1p` citando la asimetría documentada en T4, y el Experimento 1
omitía esa misma transformación sobre las mismas variables.

**Evidencia más contundente.** En `40-59 · Female` con variables crudas,
**ningún corte del dendrograma** produce clusters donde el más chico llegue a
5 personas — la función recorre todos los saltos y cae al valor por defecto.
No es que el corte elegido fuera malo: es que **no existe ningún corte bueno**
sobre esa estructura sin transformar.

**Decisiones tomadas:**

1. **Experimento 1 se corre en dos variantes** (crudo y `log1p`) y se reportan
   **ambas**. No se elige la que más guste: la diferencia entre ellas *es* el
   resultado. Caso extremo, `40-59 · Male` pasa de `[136, 20]` —un subgrupo
   severo minoritario— a `[78, 78]`, dos mitades parejas. **No es el mismo
   hallazgo**, y presentar solo uno habría atribuido a los datos algo que
   depende de una decisión de preprocesamiento.
2. **`MIN_CLUSTER_SIZE = 5`** en `src/config.py`; `hierarchical_cluster_cut`
   acepta ahora `min_cluster_size` y recorre los saltos de mayor a menor hasta
   encontrar un corte que lo cumpla. Nueva función `cluster_sizes()` para que
   los tamaños se reporten **siempre**.
3. **`MIN_STRATUM_SIZE_FOR_CLUSTERING = 30`**: se excluyen del clustering
   `0-17 · ambos sexos` (n=25) y `60-120 · Female` (n=20). No se reportan "con
   caveat" — simplemente no se les calcula nada.
4. Tabla de De Ritis con **columna `n` visible** y bandera `n_suficiente`.

**Hallazgo que emerge de la corrección.** Con los estratos de muestra
suficiente y `log1p`, la diferencia de De Ritis entre el cluster alterado y el
resto es **consistente en los 5 estratos** (+0.16 a +0.66), cuando antes
parecía esporádica (2 de 7). Lo que la ocultaba era metodológico, no
biológico. Sigue sin poder distinguirse severidad de etiología, y **ningún
estrato cruza el corte clásico de 2.0**.

**Limitación estructural registrada para la Fase 5.** De los dos estratos
excluidos por muestra insuficiente, uno es el de **mujeres de 60+** (n=20). En
un proyecto sobre equidad por sexo, el desbalance 441/142 no solo sesgaría a
un modelo futuro: **impide responder preguntas sobre las mujeres incluso en un
análisis puramente descriptivo.**

**Impacto sobre la Fase 3.** F3-R14 proponía la transformación logarítmica
como alternativa **[V]** para las variables asimétricas. Este resultado
sugiere reforzarla: cualquier método basado en distancias o varianzas sobre
estas variables sin transformar mide, sobre todo, a los casos extremos.
