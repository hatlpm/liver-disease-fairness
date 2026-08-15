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
| — | **ALP confundida por edad:** mediana 320 en menores vs ~200 en adultos | Rangos pediátricos 129–468 U/L según edad y sexo (mayoría de las filas: Children's Minnesota Lab, hoja de laboratorio hospitalario; CALIPER/Zierk 2017 respaldan el principio general, no la mayoría de estos números — corregido tras auditoría externa, 2026-08-15). CLSI C28-A3 exige particionar antes de aplicar Tukey |
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

---

## 2026-07-28 — Loop B: Data Preparation → Data Understanding

**El loop que el PRD anticipaba, disparado por un motivo distinto al previsto.**

El §9.1 del PRD define el Loop B así: *"al analizar outliers en Fase 3 se
descubre algún valor clínicamente imposible → se necesita más EDA"*. Se
verificó esa hipótesis y **resultó falsa**: al contrastar los 84 valores
atípicos de las cuatro variables de T8 contra límites biológicos duros
(`TP` 2–12, `ALB` 0.5–6.5, `A/G Ratio` 0.1–5.0, `Sgot` 1–20000), **cero
quedan fuera del rango posible**. Todos son *outliers estadísticos*, no
erróneos.

Los dos casos extremos se revisaron paciente por paciente y son internamente
coherentes: el de `Sgot` = 4929 tiene los tres ejes hepáticos alterados a la
vez (bilirrubina 9× el techo normal, transaminasas en miles, albúmina en
2.4); el de `TP` = 2.7 tiene albúmina en 0.9. Son cuadros clínicos completos,
no errores de captura. La única violación dura del dataset sigue siendo la de
`DB > TB` ya detectada en Loop C.

**Pero el loop se disparó igual, por otro hallazgo.**

**Disparador real:** al repetir el conteo de outliers estratificado (F3-R15 y
el mandato de Loop C sobre edad), el umbral único de Tukey resultó **no ser
neutro respecto a sexo ni a edad** — y el sesgo **no tiene una sola
dirección**.

| Variable | Subgrupo | Umbral global | Umbral propio |
|---|---|---|---|
| `Sgot` | Mujeres (n=142) | 10 (**7.0%**) | 15 (**10.6%**) |
| `A/G Ratio` | Mujeres (n=142) | 3 (**2.1%**) | 7 (**4.9%**) |
| `Alkphos` | Menores (n=25) | 4 (**16.0%**) | 2 (**8.0%**) |
| `ALB` | Menores (n=25) | **0 (0.0%)** | **4 (16.0%)** |
| `TP` | Menores (n=25) | **0 (0.0%)** | **3 (12.0%)** |

**Hallazgo 1 — subdetección en mujeres.** Los cuartiles globales se calculan
sobre una muestra que es 75.6% masculina, así que están calibrados
esencialmente con hombres. Una mujer con un valor elevado *para una mujer*
puede quedar dentro de un límite fijado por la distribución masculina. **Es
el mismo mecanismo que Loop C documentó en los umbrales de ALT**, ahora
reapareciendo en una etapa distinta del pipeline: no solo el umbral
diagnóstico, también el tratamiento de outliers.

**Hallazgo 2 — el sesgo etario se invierte según la variable.** Loop C
predijo que el umbral global **sobredetectaría** en menores por la ALP
elevada del crecimiento óseo, y se confirma (16.0% vs 8.0%). Lo **no
anticipado** es que en `ALB` y `TP` ocurre lo contrario: el umbral global
marca **cero** menores, mientras que dentro de su propia banda saldrían 4 y
3. Es decir, el umbral único también **subdetecta** donde la distribución
pediátrica es más estrecha que la adulta.

Conclusión general: **el umbral único no describe bien a ningún subgrupo en
particular**, y el error que comete depende de la variable.

**Hallazgo 3 — los duplicados no son neutros.** Las 26 filas implicadas en
duplicación son **84.6% hombres**, frente al 75.6% de la muestra completa. La
diferencia es pequeña y sobre 26 filas no admite lectura estadística, pero se
registra: cualquier decisión sobre duplicados afecta más a un sexo que al
otro.

**Decisiones tomadas:**

1. **No se altera el conteo obligatorio.** El número que responde a la
   rúbrica (F3-R11) es el global, calculado sobre las 583 filas, y así queda
   reportado. La estratificación **se suma**, nunca sustituye (§0.4 del PRD).
2. **No se elimina ningún valor atípico.** Al no haber ninguno erróneo,
   eliminarlos borraría exactamente los pacientes más graves — los que un
   cribado debe detectar. Se conservan las 583 filas.
3. **Se traspasa a la Fase 5** como material concreto: el tratamiento de
   outliers es un punto del pipeline donde el sesgo por sexo puede entrar sin
   que nadie lo decida explícitamente.

**Impacto:** ninguno sobre el alcance de las Fases 0–3. Refuerza con
evidencia nueva la hipótesis de sub-diagnóstico registrada en Loop A, y añade
un mecanismo candidato que no estaba en la lista: **el preprocesamiento
mismo**.

---

## 2026-08-15 — Loop D: auditoría metodológica externa

**Disparador:** una revisión metodológica independiente del proyecto completo
(no del equipo, un revisor externo) encontró 8 hallazgos, ejecutando el
código y recalculando cifras en vez de solo leer el informe. Dos se
calificaron de gravedad crítica.

**Hallazgos y decisiones (todas, salvo la nº4, aprobadas explícitamente por
el usuario antes de ejecutarse — ver detalle en `AGENTS.md`, checkpoint
2026-08-15):**

1. **La brecha de diagnóstico por sexo (8.7 pp) nunca se probó
   estadísticamente.** Con n=142 mujeres, el test exacto de Fisher da
   **p = 0.055** (IC 95% de la diferencia: −0.2 a +17.6 pp, cruza el cero) —
   en el límite de la significancia convencional. **Decisión:** agregar la
   prueba y suavizar el lenguaje de "hallazgo principal" a "brecha
   observada, y un mecanismo candidato" en `informe_act1.md`/`.tex` y
   `02_eda.ipynb`. No invalida el análisis de sensibilidad de ALT, que sigue
   siendo un mecanismo evaluable de forma independiente.
2. **El "cluster de severidad" de `02c_eda_clustering.ipynb` es
   parcialmente circular.** Se etiqueta usando la mediana de `TB`, que ya es
   la 2.ª variable más correlacionada con `Selector` de las 9. **Decisión:**
   se agregó un chequeo directo — ordenar solo por `TB` reproduce 54–81% del
   mismo cluster (índice de Jaccard) y, en 3 de 5 estratos, casi la misma
   tasa de diagnóstico. La tabla de honestidad metodológica del notebook se
   corrigió para reflejarlo en vez de presentar el hallazgo como estructura
   multivariada nueva sin matiz.
3. **Los límites biológicos de T8/F3-R12 no tenían fuente citada**
   (`TP` 2.0–12.0, `ALB` 0.5–6.5, `A/G Ratio` 0.1–5.0, `Sgot` 1–20 000).
   **Decisión:** se investigó (`docs/fuentes/Consulta_5.md`). Para el techo
   de `Sgot` existe fuente real y verificable — Chang et al. (2007), rango
   clínicamente reportable, 7 446 U/L, más estricto que el anterior sin
   fuente — adoptada en `src/config.py`. Para `TP`, `ALB` y `A/G Ratio` no se
   encontró una fuente publicada verificable con la confianza que exige el
   protocolo del proyecto: se declaran explícitamente como estimación
   razonada del equipo, no como cita clínica. *(Corrección durante la
   implementación: el primer intento también adoptó el **piso** de Chang et
   al., 24 U/L, para `Sgot` — error, porque ese piso es el mínimo verificado
   por esa prueba de calibración de instrumento, no un piso biológico; un
   AST de 10 U/L es sano y normal, y 124 pacientes del ILPD quedaban
   marcados como "erróneos" por ese piso mal aplicado. Corregido: el piso de
   `Sgot` vuelve a ser 1, sin necesidad de cita.)*
4. **El índice de Whipple (Q9) se aplica a una cohorte hospitalaria sin
   discutir que fue diseñado para datos censales.** **Decisión:** párrafo de
   limitación explícito en `informe_act1.md`/`.tex` y `data_dictionary.md`.
   No se retracta el valor 163.3 (se reverificó, es correcto) ni la
   conclusión de que las edades están redondeadas — se declara la
   incertidumbre sobre la transferencia de la escala de interpretación de la
   ONU a un contexto no censal. Se registró además, como pregunta abierta
   sin investigar a fondo, un posible artefacto adicional de *heaping* o
   *top-coding* en `Age = 60` (34 pacientes, frente a 14 en 58 y 5 en 61).
5. **Atribución de fuente inflada:** el rango de ALP pediátrica (129–468
   U/L) se citaba como "CALIPER, Zierk 2017", pero la mayoría de las filas
   de la tabla real vienen de una hoja de laboratorio hospitalario
   (Children's Minnesota Lab), no de esos papers revisados por pares.
   **Decisión:** corregir la cita en `02_eda.ipynb` y aquí mismo (§ Loop C,
   arriba) para que refleje la fuente real de cada número.
6. **Las estadísticas oficiales de T3 (`TB`/`DB`) incluyen 2 valores que T2
   (Q8) ya calificó de "bioquímicamente imposibles".** **Decisión:** nota de
   sensibilidad (con/sin esas 3 filas) en `02_eda.ipynb`, `act1_anexo.ipynb`
   e `informe_act1.md`/`.tex`, con el mismo criterio ya usado para los 13
   duplicados en §3.2. **Los números oficiales de T3 no se modifican** —
   afectan a un criterio de la rúbrica (C2) y ya están en el PDF entregado;
   se agregó transparencia, no un recálculo.
7. **Los 6 nulos sin imputar (3 `TB` + 3 `DB`) se propagan en silencio a los
   tres CSV de `data/processed/`**, incluidos los dos "escalados" — sklearn
   no falla con `NaN`, lo deja pasar. **Decisión:** nota explícita en
   `data_dictionary.md` § 4.1 para que la futura Fase 4 no los descubra por
   accidente.
8. **El caveat aritmético del umbral de ALT** ("la ventana femenina es más
   ancha") no estaba cuantificado. **Decisión:** se agregó un chequeo de
   densidad por unidad de ancho — las mujeres concentran ~40% más densidad
   que los hombres en su respectiva franja, incluso ajustando por el ancho —
   como corroboración independiente, en `02_eda.ipynb` e
   `informe_act1.md`/`.tex`.

**Verificación técnica.** Los 4 notebooks modificados
(`02_eda.ipynb`, `02c_eda_clustering.ipynb`, `03_preprocessing.ipynb`,
`act1_anexo.ipynb`) se re-ejecutaron *restart & run all* tras cada cambio —
**0 errores** en los cuatro. `reports/informe_act1.pdf` se recompiló.

**Impacto:** ninguna de las 8 correcciones cambia una conclusión central del
proyecto (todas eran matices de rigor, no errores de cómputo), pero sí
cambia cuánta certeza transmite el lenguaje en dos puntos (la brecha de
sexo, el cluster de severidad) y corrige trazabilidad de fuentes en otros
dos. Se registra íntegro por transparencia, siguiendo la misma regla de oro
que el proyecto aplica a sus propios *loops*: todo hallazgo que obliga a
revisar una fase anterior se documenta con fecha, disparador y decisión.

## 2026-08-15 — Loop E: revisión del repositorio y un error de intervalo

**Disparador:** una revisión del repositorio completo (estado de git,
reproducibilidad, coherencia entre documentos) encontró que el entorno no era
ejecutable en una máquina nueva y que varias afirmaciones de la documentación
no coincidían con el estado real. Al re-ejecutar todo para verificar las
cifras publicadas, apareció además **un error numérico real** en una de las
correcciones del Loop D.

### El error de intervalo (lo relevante para el análisis)

El Hallazgo 8 del Loop D añadió un chequeo de **densidad de ALT por unidad de
ancho** para cuantificar cuánto del exceso femenino era artefacto del ancho de
la ventana. Ese chequeo contaba la franja como `[ULN_sexo, 40)` — cerrada por
abajo, abierta por arriba.

**Esos límites no coinciden con la regla de reclasificación que el chequeo
debe explicar.** La regla es `anormal ⟺ Sgpt > ULN`, luego se reclasifica
exactamente quien cae en `(ULN_sexo, 40]`. Con los límites viejos:

| | Franja `[ULN, 40)` (vieja) | Franja `(ULN, 40]` (correcta) | Reclasificados según la tabla oficial |
|---|---|---|---|
| Mujeres | 70 | 70 | **70** ✅ |
| Hombres | **74** ❌ | 69 | **69** |

El conteo masculino incluía **12 hombres con ALT exactamente 30** (que *no*
cambian de categoría) y excluía **7 con ALT exactamente 40** (que *sí*):
74 − 12 + 7 = 69. En mujeres el error se cancelaba por casualidad (2 casos en
cada extremo), razón por la cual pasó desapercibido: la cifra femenina, que es
la que se citaba, era correcta.

**Efecto sobre el resultado:** la densidad masculina pasa de `0.0168` a
`0.0156` personas por U/L, y el exceso femenino sube de "~40% mayor" a
**exactamente 50% mayor**. La dirección del hallazgo **no cambia: se
refuerza**. Se registra igual, porque el error existió y porque el criterio
del proyecto no es publicar solo lo que conviene.

**Prevención:** la celda incorpora ahora un `assert` que falla si el conteo de
la franja vuelve a divergir del número de reclasificados de la tabla de
umbrales. Un chequeo de coherencia que se ejecuta solo, en vez de una nota que
hay que recordar leer.

### Verificación independiente de todas las cifras

Se recalcularon **41 cifras publicadas** desde `data/raw/` usando `src/`, sin
leer los notebooks, y se compararon una a una contra lo que afirman el informe
y el README: estructura (583×11, 441/142, 13 duplicados, 4 nulos), brecha de
diagnóstico (73.47% / 64.79% / 8.7 pp), Fisher (p = 0.055), las ocho celdas de
la tabla de umbrales de ALT, los porcentajes entre no-hepáticos (68.0% /
41.0%), el sesgo de outliers (`Sgot` y `A/G Ratio` en mujeres), el índice de
Whipple (163.3) y los tres CSV procesados (583×11, 6 nulos cada uno).

**Resultado: 39 de 41 coincidían exactamente. Las 2 que no eran las dos caras
del error de intervalo descrito arriba.** Corregidas, las 41 coinciden.

### Correcciones de reproducibilidad y documentación

| # | Problema | Corrección |
|---|---|---|
| 1 | El `venv/` **sí** viaja por OneDrive (`.gitignore` no lo impide) y llega apuntando al intérprete de otra máquina | Recreado con Python 3.13.3; `AGENTS.md` y `README.md` corregidos — antes afirmaban lo contrario, que era la razón de que nadie borrara el roto |
| 2 | Rutas locales filtradas en 3 notebooks pese al commit `437aae2` que decía haberlas eliminado | Causa raíz eliminada: `matplotlib.use("Agg")` forzaba un backend no interactivo cuyo aviso imprimía la ruta temporal. El commit anterior había borrado las *salidas* pero no la *causa*, así que reaparecieron al re-ejecutar en el Loop D. Además `03_preprocessing` imprimía la ruta absoluta del equipo: ahora imprime `data/processed/` |
| 3 | Efecto colateral favorable del punto 2 | Al quitar el backend `Agg`, las figuras **se renderizan dentro de los notebooks** (10 en total). Antes solo existían como PNG sueltos en `reports/figures/` |
| 4 | `AGENTS.md` se contradecía: la Fase E figuraba como "pendiente de merge" y dos filas más abajo como ya mezclada | Verificado con `git branch --merged`: mezclada. Fila corregida |
| 5 | El PRD, "fuente de verdad" declarada, vive fuera del repo y sin versionar | Se ratifica dejarlo fuera y se documentan las consecuencias en `docs/adr/0005-prd-fuera-del-repositorio.md` |
| 6 | El índice de `docs/adr/README.md` no listaba el ADR 0004 | Añadidos 0004 y 0005 |
| 7 | Tres copias idénticas del dataset (`act1/`, `act2/`, `data/raw/`) | Borradas las dos de fuera del repo; la de `data/raw/` es la canónica y está versionada |
| 8 | 7 ramas `feature/*` ya mezcladas seguían vivas local y en el remoto | Borradas en ambos sitios |
| 9 | El README describía el umbral de ALT como "el hallazgo principal", en un tono más seguro que el informe tras el Loop D | Reescrito: ahora abre con la brecha y su `p = 0.055`, y separa explícitamente lo que puede afirmarse de lo que no |

### Qué queda abierto

- ✅ **`reports/informe_act1.pdf` recompilado** (resuelto el mismo día, en una
  segunda ronda). `pdflatex` **sí estaba en el PATH** de la máquina; la razón de
  que pareciera ausente es que `AGENTS.md` traía la ruta absoluta de otra
  máquina en vez de resolverla desde el entorno. Verificado: **10 páginas
  exactas, 0 errores**, y las tres cifras corregidas presentes en el PDF
  (`0.0156`, `exactamente un 50 % mayor`, `Cinco iteraciones`), sin rastro de
  las viejas.

  > 🔴 **Trampa descubierta y registrada.** La primera compilación se colgó
  > indefinidamente: faltaba `caption.sty`, MiKTeX intentó descargarlo de un
  > mirror de CTAN que cerró la conexión, y `pdflatex` quedó esperando con el
  > socket en `CloseWait` — **sin consumir CPU y sin escribir al log**, de modo
  > que parecía estar trabajando. La señal que lo delató fue el tiempo de CPU
  > congelado. Desde ahora se compila con `--disable-installer`, que falla al
  > instante y nombra el `.sty` que falta en vez de colgarse. Documentado en
  > `AGENTS.md`.

- ⏸️ **Hallazgo 4 del Loop D** (índice de Whipple aplicado fuera de su dominio
  censal) sigue sin la confirmación explícita del usuario, tal como lo dejó
  registrado el propio Loop D.

**Verificación técnica.** Los 5 notebooks se re-ejecutaron *restart & run all*
con el entorno nuevo — **0 errores**. Ninguna ruta local queda en el
repositorio. Las 41 cifras publicadas coinciden con el recálculo
independiente.

---

## 2026-08-15 — Loop F: Modeling → Data Preparation (métrica de optimización mal elegida)

**Disparador:** al empezar la Fase 6 (ajuste de hiperparámetros de los 5
algoritmos), antes de tocar el notebook se contrastó `params.yaml` contra el
propio dataset. `metrics.optimize_for: "f1"` con `pos_label: 1` —
declarado en la Fase 4/§11.2 del PRD, **antes de que existiera ningún
modelo entrenado** — asume implícitamente que la clase positiva es la
minoritaria (el caso típico para el que F1 se recomienda). En este dataset
no lo es: `Selector = 1` (enfermo) es la clase **mayoritaria** (71.27% en
train).

**Hallazgo.** Medido sobre train (456 filas), el clasificador nulo que
responde "todos enfermos" sin mirar ningún dato de entrada obtiene:

| Métrica | Valor |
|---|---|
| accuracy | 0.7127 |
| recall | 1.0000 |
| precision | 0.7127 |
| **F1** | **0.8323** |
| balanced_accuracy | 0.5000 |

(Sobre test, congelado desde la Fase 4 y no tocado en esta fase: accuracy
0.7105, recall 1.0000, precision 0.7105, F1 0.8308 — cifras de referencia
del PRD, recalculadas aquí solo como verificación, nunca usadas para
ajustar nada.)

Optimizar `GridSearchCV` por F1 (o por accuracy) premia exactamente al
modelo que no aprendió nada: cualquier configuración que se acerque a
"predecir siempre enfermo" saca un F1 cercano a 0.83 sin distinguir un
paciente de otro. Verificado empíricamente ajustando los 5 algoritmos
exigidos: optimizados por `balanced_accuracy` (la métrica corregida), su F1
de validación cruzada cae entre 0.62 y 0.72 — **por debajo** del suelo
nulo — precisamente porque dejan de sobre-predecir la clase mayoritaria
para también acertar en la clase sana. `balanced_accuracy`, en cambio, sube
de su propio suelo nulo (0.50, azar) a 0.67–0.72 en los 5 modelos: la
métrica corregida sí distingue "aprendió algo" de "no aprendió nada";
la original no.

Consecuencia adicional, específica de este proyecto: un modelo que predice
"enfermo" casi siempre tiene FNR ≈ 0 en ambos sexos, así que la auditoría
de equidad de la Fase 9 no habría encontrado ninguna brecha — no porque no
exista, sino porque un modelo que no distingue a nadie tampoco puede
discriminar a nadie.

**Decisión:** `metrics.optimize_for` en `params.yaml` pasa de `"f1"` a
`"balanced_accuracy"` — la media de la sensibilidad (recall) en cada clase,
con suelo nulo en 0.50 en cualquier split, en vez de un suelo inflado por
el desbalance de clase. Documentado en
`docs/adr/0012-metrica-de-optimizacion-balanced-accuracy.md`. `accuracy`,
precisión, recall y F1 se **siguen reportando** tal como exige T6/T7 del
enunciado (`tests/test_fase6_modelos.py` los calcula y compara contra este
mismo suelo) — lo que cambia es el criterio de **selección** de
hiperparámetros, no lo que se informa.

**Impacto:** ningún cálculo de las Fases 4–5 se modifica — el hallazgo es
sobre una configuración declarada para una fase que todavía no se había
ejecutado, no sobre datos ya procesados. Sí cambia el comportamiento de
`GridSearchCV` en la Fase 6 y en todo lo que la reutilice (Fases 7–9): los
modelos seleccionados van a tener F1/accuracy más bajos que si se hubiera
optimizado por F1, y **eso es la corrección funcionando, no una
regresión** — queda fijado en
`tests/test_fase6_modelos.py::test_scoring_supera_el_suelo_nulo` para que
una sesión futura no lo lea como un error.
