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
