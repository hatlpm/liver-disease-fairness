# Diccionario de datos — ILPD (Indian Liver Patient Dataset)

Este documento describe **qué mide cada columna** del dataset, en qué unidad, y
contra qué rango de referencia se interpreta. Está escrito para que alguien sin
formación clínica pueda leer los análisis del proyecto y entender qué significan
los números — no solo su comportamiento estadístico.

**Archivo fuente:** `data/raw/act_liver_disease.csv` (583 filas × 11 columnas, inmutable).

---

## 1. Procedencia del dataset

| Campo | Valor |
|---|---|
| Nombre | ILPD — *Indian Liver Patient Dataset* |
| Origen | Noreste de Andhra Pradesh, India |
| Creadores | Bendi Venkata Ramana, M. Surendra Prasad Babu, N. B. Venkateswarlu |
| Donado a UCI | 20 de mayo de 2012 |
| DOI | 10.24432/C5D02C |
| Composición | 416 hepáticos / 167 no hepáticos · 441 hombres / 142 mujeres |

> ⚠️ **La ficha oficial de UCI declara "Missing values: No", pero el archivo
> tiene 4 valores faltantes** en `A/G Ratio` (filas 209, 241, 253, 312). La
> documentación oficial contradice al dato real; se audita el archivo, no la ficha.

---

## 2. Cómo entender estas variables: los tres trabajos del hígado

Las 9 variables numéricas de laboratorio no son una lista arbitraria. Miden
**tres funciones distintas** del hígado, y la dirección de la alarma **no es la
misma en las tres**.

### Eje 1 — Daño celular (¿se están rompiendo las células?)

Las células del hígado guardan enzimas **en su interior**. Si la célula se
rompe, esas enzimas se derraman a la sangre. Encontrarlas elevadas en sangre es
como encontrar aceite de motor en el suelo: debería estar dentro, no fuera.

**Alto = malo.**

### Eje 2 — Excreción biliar (¿está drenando la bilis?)

El hígado elimina desechos por la vía biliar. Si esa vía se obstruye, los
desechos se acumulan en sangre.

**Alto = malo.**

### Eje 3 — Función sintética (¿sigue fabricando?)

El hígado es una fábrica de proteínas. Si la fábrica falla, la producción cae.
Como estas proteínas tienen vida media larga, este eje refleja daño **crónico**,
no agudo.

**Bajo = malo.** ← dirección inversa a los otros dos ejes

---

## 3. Diccionario de variables

### Demográficas

| Columna | Tipo | Unidad | Descripción |
|---|---|---|---|
| `Age` | int | años | Edad del paciente. **Ver advertencias abajo.** |
| `Gender` | str | — | `Male` / `Female`. Única variable categórica; sin faltantes. |

**⚠️ Dos advertencias sobre `Age`:**

1. **Top-coding.** La documentación de UCI dice literalmente: *"Any patient
   whose age exceeded 89 is listed as being of age '90'."* El valor 90 **no
   significa "90 años", significa "≥90"**. Hay 1 registro así. El rango
   observado 4–90 no es un rango real de edades.
2. **Age heaping.** Índice de Whipple = **163,3**, categoría **"Tosca"** en la
   escala de Naciones Unidas (125–175). Hay un 63% más de personas en edades
   terminadas en 0 o 5 de las que debería haber. Las edades se estimaron o se
   reportaron de memoria, no se leyeron de un documento. **Error práctico
   estimado: ±2–3 años.** Consecuencia: no usar franjas etarias estrechas.

   > ⚠️ **Limitación de dominio (añadida tras auditoría externa, 2026-08-15).**
   > El índice de Whipple y la escala de la ONU que lo interpreta fueron
   > diseñados para datos **censales**, donde la estructura etaria refleja a
   > la población general. El ILPD es una cohorte **hospitalaria**: quién
   > aparece en los datos depende de quién buscó atención médica, no de un
   > muestreo poblacional. `docs/fuentes/Consulta_4.md` (la fuente del
   > índice) no discute su aplicabilidad fuera de contexto censal, y el
   > proyecto tampoco lo había hecho hasta ahora. El valor 163,3 en sí es
   > correcto aritméticamente — se verificó de forma independiente — pero la
   > *categoría cualitativa* ("Tosca") toma prestada su validez de una escala
   > calibrada para un tipo de dato distinto. No se retracta la conclusión
   > (las edades sí parecen redondeadas), pero se declara la incertidumbre
   > sobre cuánto pesa esa etiqueta específica en un dataset hospitalario.
   >
   > **Nota abierta, no investigada a fondo:** la distribución de `Age`
   > muestra un pico inusual en **Age = 60** (34 pacientes, frente a 14 en
   > 58 y 5 en 61) — justo en el límite entre las bandas `40-59` y `60-120`
   > que este proyecto usa en sus análisis estratificados. Podría ser otro
   > artefacto de redondeo o un tope administrativo similar al de `Age = 90`
   > (Q7), pero no se confirmó documentalmente. Queda registrado como
   > pregunta abierta para una futura revisión, no como problema de calidad
   > confirmado.

### Eje 1 — Daño celular

| Columna | Nombre clínico | Qué mide | Unidad | Ref. adulto |
|---|---|---|---|---|
| `Sgpt` | **ALT** (alanina aminotransferasa) | Enzima intracelular del hepatocito. La **más específica** del hígado. | U/L | Ver tabla de umbrales |
| `Sgot` | **AST** (aspartato aminotransferasa) | Misma idea, pero AST **también existe en corazón y músculo** → menos específica del hígado. | U/L | ≤ 40 (unisex, referencial) |

**`U/L`** = "unidades por litro". Mide la **actividad** de la enzima, no su peso.
No hace falta saber qué es una "unidad": lo que importa es comparar contra el
rango de referencia.

#### Umbrales de ALT — la variable donde el sexo importa

| Criterio | Hombres | Mujeres | Fuente |
|---|---|---|---|
| Unisex de la época del ILPD | 40 | 40 | Práctica de laboratorio 2007–2012 |
| *Healthy ULN* | **30** | **19** | Prati et al. 2002, Ann Intern Med, DOI 10.7326/0003-4819-137-1-200207020-00006 |
| Guía clínica actual | 29–33 | 19–25 | ACG 2017 (Kwo et al.); AASLD 2023 (Rinella et al.) |

Las mujeres sanas tienen ALT **naturalmente más bajo** que los hombres sanos —
fisiología normal, no mejor salud. Un umbral único de 40 queda calibrado más
cerca del techo masculino (30) que del femenino (19). Ver
`docs/fuentes/Consulta_1.md` y el análisis de sensibilidad en `02_eda.ipynb`.

**Cociente derivado — De Ritis (AST/ALT):** no es una columna del CSV, se
calcula. Indicio **sugerente** del patrón de daño (>2 tipo alcohólico/cirrótico,
<1 tipo viral/esteatosis). **Nunca prueba etiológica**: refleja también el
estadio de la enfermedad y depende del método analítico. Ver `Consulta_3.md`.

### Eje 2 — Excreción biliar

| Columna | Nombre clínico | Qué mide | Unidad | Ref. adulto |
|---|---|---|---|---|
| `TB` | Bilirrubina **total** | Desecho de glóbulos rojos viejos. Si se acumula → ictericia (piel amarilla). | mg/dL | 0,3 – 1,2 |
| `DB` | Bilirrubina **directa** (conjugada) | La fracción que el hígado **ya procesó**. Está **contenida dentro** de `TB`. | mg/dL | 0,0 – 0,3 |
| `Alkphos` | **ALP** (fosfatasa alcalina) | Enzima de los conductos biliares. **También del hueso** — ver advertencia. | U/L | 40 – 129 |

**⚠️ Restricción dura: `DB ≤ TB` siempre.** La directa es una *fracción* de la
total; no puede excederla. **El dataset tiene 3 filas que la violan**
(`TB=1,8/DB=9,0`, `TB=1,5/DB=7,0`, `TB=1,0/DB=1,4`). En la literatura esto se
debe a interferencia analítica (paraproteínas en gammapatías monoclonales,
hemólisis, problemas de calibración) — **nunca a fisiología real**. Tratamiento
adoptado: marcar **ambas** columnas como faltantes en esas 3 filas, sin eliminar
la fila. Ver `Consulta_4.md` (marco de Kahn / OHDSI, violación de plausibilidad).

**⚠️ `Alkphos` depende fuertemente de la edad.** La ALP no viene solo del
hígado: los osteoblastos la producen para mineralizar hueso. **Un niño en
crecimiento tiene ALP alta de forma completamente normal.**

| Edad | Sexo | ALP normal (U/L) |
|---|---|---|
| 1–10 años | ambos | 142 – 335 |
| 10–13 años | ambos | 129 – 417 |
| 13–15 años | niños | 116 – 468 ← pico del estirón |
| 13–15 años | niñas | 57 – 254 |
| 15–17 años | niños | 82 – 331 |
| 15–17 años | niñas | 50 – 117 |
| **>19 años (adulto)** | ambos | **40 – 129** |

*Fuente: la mayoría de estas filas provienen de Children's Minnesota Lab (hoja
de referencia de laboratorio hospitalario, no revisada por pares). CALIPER y
Zierk et al. (2017) respaldan el principio general — que la ALP pediátrica
requiere partición por edad y sexo — pero no son la fuente directa de la
mayoría de estos números tabulados. Corrección de atribución añadida tras
auditoría externa, 2026-08-15. Ver `Consulta_2.md` para el detalle fila por
fila.*

**Consecuencia para este dataset:** hay **25 menores de 18 años** (el más joven
de 4) mezclados con 558 adultos. Su mediana de ALP es **320** vs ~200–215 en
adultos. Un ALP de 320 es **normal** en un niño de 13 años y **patológico** en
un adulto de 45. Detectar outliers con la regla de Tukey sobre la población
mezclada marcará **niños sanos** como atípicos. CLSI C28-A3 y CALIPER exigen
**particionar por edad antes** de aplicar Tukey.

### Eje 3 — Función sintética

| Columna | Nombre clínico | Qué mide | Unidad | Ref. adulto |
|---|---|---|---|---|
| `TP` | Proteína total | Toda la producción proteica en sangre. | g/dL | 6,0 – 8,3 |
| `ALB` | Albúmina | La principal proteína fabricada por el hígado. Tarda semanas en caer → marcador **crónico**. | g/dL | 3,5 – 5,5 |
| `A/G Ratio` | Cociente albúmina/globulina | **Variable derivada:** `ALB / (TP − ALB)`, donde `TP − ALB` = globulina. | — | ~1,0 – 2,0 |

**⚠️ `A/G Ratio` no es una medición independiente**, es aritmética de las otras
dos. Su correlación con `TP` y `ALB` es **algebraica, no empírica**.

**⚠️ Restricción dura: `ALB ≤ TP`** (la albúmina está contenida en la proteína
total). **Verificado: 0 violaciones.**

**Nota sobre la reconstrucción:** sobre 579 filas comparables, el error absoluto
medio al recalcular `A/G Ratio` con la fórmula es **0,051** (mediana 0,031,
máximo 2,30), con **191 filas (33,0%) por encima de 0,05**. La fórmula es
correcta por definición; el error viene del **redondeo en origen** (`TP` y `ALB`
a un decimal, `A/G Ratio` a dos), que se amplifica al dividir cuando la
globulina es pequeña.

### Variable objetivo

| Columna | Tipo | Valores | Descripción |
|---|---|---|---|
| `Selector` | int | 1 / 2 | **1 = paciente hepático · 2 = no hepático** |

**⚠️ Tres advertencias, todas importantes:**

1. **Codificación contraintuitiva.** 1 = enfermo, 2 = sano. Debe recodificarse
   a 1/0 antes de modelar (fuera del alcance de las Fases 0–3).
2. **No es verdad biológica, es un *proxy label*.** Es el **juicio de un
   especialista** registrado en un hospital concreto. El modelo no aprende
   *"quién tiene enfermedad hepática"* sino *"a quién le diagnosticaron
   enfermedad hepática ahí"*. Hereda los sesgos de ese sistema clínico.
3. **No codifica ninguna enfermedad concreta.** El dataset **no tiene variable
   de etiología**: no registra alcohol ni hepatitis viral (ambigüedad A1 del
   PRD). Bajo la misma etiqueta conviven patrones fisiopatológicos distintos.

---

## 4. Resumen de problemas de calidad

| # | Problema | Valor | Origen |
|---|---|---|---|
| Q1 | Dimensiones | 583 × 11 | Fase 2 |
| Q2 | Faltantes | 4 en `A/G Ratio` | Fase 2 |
| Q3 | Duplicados exactos | 13 | Fase 2 |
| Q4 | Desbalance de clase | 416 / 167 ≈ 71/29 | Fase 2 |
| Q5 | Desbalance de sexo | 441 / 142 ≈ 76/24 | Fase 2 |
| Q6 | Codificación del target | 1 = enfermo | Fase 2 |
| Q7 | **Top-coding de edad en 90** | 1 fila; **confirmado por doc. UCI** | Loop C |
| Q8 | **`DB > TB`** (violación bioquímica) | **3 filas** | Loop C |
| Q9 | **Age heaping** | **Whipple 163,3 — "Tosca"** | Loop C |

---

## 4.1 Nulos residuales en `data/processed/` (añadido tras auditoría, 2026-08-15)

⚠️ **Los tres CSV de `data/processed/` (`ilpd_procesado.csv`, `ilpd_zscore.csv`,
`ilpd_log_zscore.csv`) tienen, cada uno, 6 celdas vacías: 3 en `TB` y 3 en
`DB`.** Son las mismas 3 filas de Q8 (`DB > TB`), marcadas como faltantes en
Fase 3 y **deliberadamente no imputadas** (ver §7.3 del informe: son valores
medidos mal, no ausentes, y rellenarlos apilaría una invención sobre un error
de laboratorio).

Esto significa que **ningún archivo de `data/processed/` está 100% completo**,
incluidos los dos que llevan "escalado" en el nombre — `StandardScaler` y
`MinMaxScaler` de scikit-learn no fallan con `NaN`: calculan media/desviación
ignorándolo y dejan el hueco en la salida. Quien use estos archivos en la
Fase 4 (modelado) debe decidir explícitamente qué hacer con esas 3 filas antes
de entrenar cualquier modelo — no van a fallar de forma ruidosa si simplemente
se ignoran.

---

## 5. Fuentes

Las respuestas de investigación que respaldan los rangos de referencia y los
criterios metodológicos están versionadas en `docs/fuentes/`:

| Archivo | Tema |
|---|---|
| `Consulta_1.md` | Rangos de referencia por sexo; Prati 2002; ACG/AASLD; estudios de India |
| `Consulta_2.md` | ALP dependiente de edad; CALIPER; CLSI C28-A3; partición antes de Tukey |
| `Consulta_3.md` | Cociente De Ritis: puntos de corte, validez y limitaciones |
| `Consulta_4.md` | `DB > TB`; marco de calidad Kahn/OHDSI; índice de Whipple |
| `Consulta_5.md` | Límites biológicamente posibles para T8/F3-R12 (añadida tras auditoría externa, 2026-08-15) |

Referencias primarias principales:

- Prati, D. et al. (2002). *Updated definitions of healthy ranges for serum alanine aminotransferase levels.* Ann Intern Med. DOI 10.7326/0003-4819-137-1-200207020-00006
- Kwo, P. et al. (2017). *ACG Clinical Guideline: Evaluation of Abnormal Liver Chemistries.* Am J Gastroenterol.
- Shah, S. et al. (2018). *Reference intervals for 33 biochemical analytes in healthy Indian population.* Clin Chem Lab Med. DOI 10.1515/cclm-2018-0152
- Zierk, J. et al. (2017). *Pediatric reference intervals for alkaline phosphatase.* Clin Chem Lab Med.
- Kahn, M. et al. (2016). *A Harmonized Data Quality Assessment Terminology and Framework.* eGEMs.
- Straw, I. & Wu, H. (2022). *Investigating for bias in healthcare algorithms.* BMJ Health Care Inform. DOI 10.1136/bmjhci-2021-100457
- Ramana, B. & Venkateswarlu, N. ILPD, UCI ML Repository. DOI 10.24432/C5D02C
- Chang, S.-W. et al. (2007). *Study on Analytical and Clinically Reportable Ranges.* Korean J Clin Lab Sci, 39(1), 31–36.
