<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# CONTEXTO: Estoy limpiando el ILPD. Encontré 3 filas donde la bilirrubina

directa excede a la bilirrubina total (ej. TB=1.8 y DB=9.0), lo cual es
bioquímicamente imposible porque la directa es una fracción de la total.
También detecté "age heaping": 16.8% de las edades son múltiplos de 10 y
16.0% múltiplos de 5 (esperable ~10% y ~10%).

CONSULTA:

1. ¿Cuál es el fundamento analítico por el que la bilirrubina directa no
puede exceder a la total? ¿Existe algún escenario de laboratorio (error de
método, interferencia, hemólisis) donde se reporte legítimamente DB > TB?
2. ¿Cuál es la práctica recomendada en control de calidad de datos clínicos
para valores que violan una restricción bioquímica dura: corregir,
marcar como faltante, o eliminar la fila? ¿Qué recomiendan las guías de
data quality en investigación clínica (ej. marcos de Kahn et al., OHDSI)?
3. ¿Qué indica el "age heaping" sobre la calidad de la recolección de datos?
¿Qué índices existen para cuantificarlo (Whipple, Myers) y qué umbrales
se consideran problemáticos?

SALIDA DESEADA:

- Respuesta separada por cada uno de los 3 puntos, con citas
- Para el punto 2, una recomendación accionable y justificada
- Para el punto 3, la fórmula del índice de Whipple y su escala de
interpretación
- Fuentes: bioquímica clínica, guías de data quality, demografía

Tengo suficiente evidencia para responder directamente sin generar un reporte, dado que es una consulta técnica puntual con tres subpreguntas específicas.

## 1. Fundamento bioquímico: ¿por qué DB no puede exceder TB?

La bilirrubina total (BT) es la suma matemática de la fracción conjugada (directa, BD) y la no conjugada (indirecta): BT = BD + BI. Por definición, un componente de una suma no puede ser mayor que el total, salvo que exista un error de medición, ya que ambas fracciones se cuantifican por métodos ópticos distintos (reacción con ácido sulfanílico diazotado) y no se derivan realmente de un balance de masas perfecto en el instrumento.[^1]

Existen escenarios de laboratorio bien documentados donde BD > BT se reporta "legítimamente" en el sentido de que el instrumento lo arroja, pero siempre por interferencia analítica, nunca por fisiología real:

- **Interferencia por paraproteínas (gammapatías monoclonales):** en mieloma múltiple u otras discrasias de células plasmáticas, concentraciones altas de inmunoglobulina monoclonal (IgG o IgM) interfieren específicamente con el ensayo de bilirrubina directa, aumentándola falsamente sin afectar igual al ensayo de bilirrubina total, produciendo BD > BT.[^2][^3][^4][^1]
- **Hemólisis y lipemia:** a concentraciones altas pueden interferir con los métodos diazo, con sesgos que dependen del método (algunos subestiman BT, otros la sobreestiman), generando discordancias entre ambas fracciones.[^5][^1]
- **Problemas de calibración y reactivo:** encuestas del College of American Pathologists muestran que el ensayo de bilirrubina directa (DBIL) tiene alta variabilidad entre laboratorios por problemas de calibración, ausencia de blanco de suero, concentración inadecuada de HCl en el reactivo, y uso de agentes tensioactivos — problemas de *bias* metodológico, no de imprecisión aleatoria.[^6]
- **Gelificación de proteína monoclonal:** reportada como causa específica de interferencia en el ensayo de bilirrubina total, generando resultados espurios y no reproducibles.[^3]

Es decir, no hay ningún mecanismo fisiológico real donde la fracción directa supere a la total; todo caso reportado en la literatura es catalogado como artefacto analítico o error de laboratorio, casi siempre ligado a proteínas anómalas en sangre.[^4][^1]

## 2. Práctica recomendada en control de calidad de datos clínicos

Los marcos de referencia como el de **Kahn et al. (2016)**, adoptado por OHDSI, clasifican este tipo de error como una violación de **plausibilidad** (específicamente "atemporal", del tipo "unexpected fact" o "implausible numerical value"), distinta de errores de conformidad o completitud. La Data Quality Dashboard (DQD) de OHDSI implementa checks de "plausibleValueHigh/Low" precisamente para detectar valores fuera de rango biológico esperado.[^7][^8][^9][^10]

La guía operativa de OHDSI es explícita: cuando un valor es implausible pero explicable (por ejemplo, por un error de ETL conocido), debe corregirse en la fuente; cuando es inexplicable, la fila del evento/medición puede eliminarse del análisis para no contaminar aguas abajo, pero **siempre documentando la decisión**. Un ejemplo práctico de un miembro activo de la comunidad OHDSI: en vez de borrar el registro completo, convierten el valor implausible en un código centinela (p. ej. -9) que indica "medición realizada pero valor inválido", preservando la fila para no perder información sobre la ocurrencia del evento, evitando a la vez sesgar estadísticas descriptivas como la media.[^9][^11]

**Recomendación accionable para tu caso (BD > BT, n=3):**

1. **No elimines las filas completas.** Dado que son solo 3 de un dataset, el riesgo de sesgo por eliminación es bajo, pero perderías información válida de otras variables (edad, otras enzimas, variable objetivo) que sí pueden ser útiles para el modelo.
2. **Marca ambos valores de bilirrubina (BD y BT) de esas 3 filas como faltantes (missing/NA)**, no los "corrijas" arbitrariamente (por ejemplo, intercambiando BD y BT), porque no tienes evidencia de cuál valor es el correcto ni si ambos están comprometidos por la misma interferencia.
3. Documenta la regla de negocio aplicada (BD > BT es una restricción dura violada) en tu diccionario de datos o script de limpieza, siguiendo el principio de plausibilidad atemporal de Kahn, de forma que sea auditable y reproducible.[^7][^9]
4. Si usas modelos que toleran NA (árboles, LightGBM/XGBoost) puedes dejar el missing tal cual; si necesitas imputación, usa un método que no dependa de las otras bilirrubinas para evitar propagar el artefacto (p. ej. imputación por mediana condicionada a otras variables hepáticas, o modelo de imputación múltiple).

Esta jerarquía —corregir solo si hay evidencia clara del valor correcto, marcar como faltante si la violación es dura y no explicable, eliminar la fila completa solo como último recurso— es consistente con cómo Kahn y OHDSI tratan los checks de plausibilidad frente a los de conformidad (donde sí se prefiere eliminar/rechazar el registro por incumplir un esquema).[^8][^12][^7]

## 3. Age heaping: qué indica y cómo cuantificarlo

El "age heaping" (acumulación en edades terminadas en 0 o 5) es un indicador clásico de **mala calidad en la recolección de la edad**, típicamente producido por redondeo del entrevistado o del encuestador cuando no se verifica la edad exacta contra un documento (cédula, partida de nacimiento), sino que se estima de memoria. Es una señal indirecta de que otros campos capturados de forma similar (sin verificación documental) también podrían tener errores de medición no aleatorios.[^13][^14]

**Índice de Whipple:**

$$
IW = \frac{P_{25}+P_{30}+P_{35}+P_{40}+P_{45}+P_{50}+P_{55}+P_{60}}{\frac{1}{5}\sum_{x=23}^{62} P_x} \times 100
$$

donde $P_x$ es la población que reporta edad exactamente $x$. El índice usa el rango 23–62 años porque ahí se asume que la mortalidad y la estructura etaria no distorsionan el resultado.[^15][^14][^7]

**Escala de interpretación (Naciones Unidas / Demographic Yearbook):**


| Valor del Índice de Whipple | Calidad de los datos |
| :-- | :-- |
| < 105 | Muy exacta (desviación <5%) [^14][^15] |
| 105 – 109.9 | Bastante exacta (5–9.99%) [^14][^15] |
| 110 – 124.9 | Aproximada (10–24.99%) [^14][^15] |
| 125 – 174.9 | Tosca / "rough" (25–74.99%) [^14][^15] |
| ≥ 175 | Muy tosca (≥75%) [^14][^15] |

Un valor de 100 indica ausencia total de preferencia por dígitos 0/5 (distribución perfectamente uniforme), mientras que 500 es el máximo teórico (todas las edades reportadas terminan en 0 o 5).[^14][^15]

Con tu dato de **16.8% en múltiplos de 10 y 16.0% en múltiplos de 5** (≈32.8% combinado en 0/5, frente al ~20% esperado bajo distribución uniforme para dos dígitos de 10 posibles), el sesgo es claramente perceptible pero moderado — consistente con un índice de Whipple probablemente en el rango "aproximado" a "tosco" (110–175), no catastrófico. Como referencia empírica, un estudio poblacional indio con edades terminadas en 0/5 en 23–24% obtuvo un IW de 116–118, catalogado como "aproximado", por lo que tu proporción más alta sugiere que el ILPD probablemente se ubique en la banda "tosca" (125–174.9).[^16]

Como complemento, el **Índice Myers Blended** evalúa preferencia por *todos* los dígitos (0-9), no solo 0 y 5, sumando poblaciones ponderadas por dígito terminal, calculando el porcentaje de cada dígito sobre el total "blended", tomando la desviación absoluta de cada uno respecto al 10% esperado, sumando esas desviaciones y dividiendo por 2. Va de 0 (sin heaping) a 90 (heaping máximo); un valor <10% se considera aceptable, y valores entre 20-40 se consideran inexactos, y >40 muy inexactos según algunas convenciones regionales.[^17][^15][^16]
<span style="display:none">[^18][^19][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^30]</span>

<div align="center">⁂</div>

[^1]: https://academic.oup.com/clinchem/article/61/6/889/5611408

[^2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9107917/

[^3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8047781/

[^4]: https://doaj.org/article/48ade07541b6449ca5e2eaa71c2ab64e

[^5]: https://acutecaretesting.org/en/articles/useful-tips-to-avoid-preanalytical-errors-in-blood-gas-testing-neonatal-total-bilirubin

[^6]: https://academic.oup.com/clinchem/article-abstract/39/4/641/5646938

[^7]: https://www.ohdsi.org/wp-content/uploads/2019/09/2-Plenary-1-OMOP-DQ-Clair-Andrew.pdf

[^8]: https://www.ohdsi.org/wp-content/uploads/2020/10/Clair-Blacketer-BLACKETER_DATA_QUALITY_NETWORK_2020Symposium.pdf

[^9]: https://ohdsi.github.io/DataQualityDashboard/articles/checks/plausibleValueHigh.html

[^10]: https://www.ohdsi.org/wp-content/uploads/2019/09/ohdsi_tutorial_presentation_20190918.pdf

[^11]: https://forums.ohdsi.org/t/defining-implausible-values-from-emr-data/6089/3

[^12]: https://ohdsi.github.io/TheBookOfOhdsi/DataQuality.html

[^13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2963876/

[^14]: https://unstats.un.org/Unsd/demographic/products/dyb/DYBcensus/V1_Notes1c.pdf

[^15]: https://www.scribd.com/document/683795471/Acs-411-Lecture-2

[^16]: https://journals.lww.com/ijcm/fulltext/2022/47010/quantifying_age_heaping_and_age_misreporting_in_a.21.aspx

[^17]: https://repository.unja.ac.id/263/1/teknik demografi (pertemuan 4).pptx

[^18]: https://www.ohdsi.org/wp-content/uploads/2024/10/128-Wieand-ExecutableSSDQAFramework-Kaleigh-Wieand.pdf

[^19]: https://wwwn.cdc.gov/nchs/data/nhanes/public/2021/labmethods/BIOPRO-L-MET-Total-Bilirubin-508.pdf

[^20]: https://www.mayoclinic.org/tests-procedures/bilirubin/about/pac-20393041

[^21]: https://my.clevelandclinic.org/health/diagnostics/17845-bilirubin

[^22]: https://mithrashealth.com/journal/data-quality-health-research/

[^23]: https://www.hsc.wvu.edu/media/5107/bilirubin-summary.pdf

[^24]: https://metricgate.com/docs/myers-blended-index/

[^25]: https://citeseerx.ist.psu.edu/document?repid=rep1\&type=pdf\&doi=e7a1042bc4dbebd4fc16ade567a33b229bd8b057

[^26]: https://askfilo.com/user-question-answers-smart-solutions/differentiate-whipplers-index-and-myers-blended-index-by-3435393638303030

[^27]: https://unstats.un.org/unsd/demographic/meetings/wshops/Myanmar/2014/docs/s06.pdf

[^28]: https://paa2014.populationassociation.org/papers/140099

[^29]: https://www.scribd.com/document/868045312/Demography-Ch-3

[^30]: https://www.ohdsi.org/wp-content/uploads/2019/09/DataQualityDashboard-Tutorial.pdf

