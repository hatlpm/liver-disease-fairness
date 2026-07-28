## Resumen directo

Sí, es un error metodológico analizar conjuntamente población pediátrica y adulta en un mismo estudio de biomarcadores hepáticos sin estratificar: la fosfatasa alcalina (FA) varía varias veces su magnitud con la edad debido a la contribución de la isoenzima ósea durante el crecimiento, y las guías CLSI/IFCC (C28-A3) y proyectos como CALIPER exigen partición por edad y sexo como requisito mínimo. Aplicar la regla de Tukey (1.5·IQR) sobre los 583 pacientes del ILPD sin excluir o estratificar a los 25 menores de 18 años inevitablemente marcará como "atípicos" valores óseos fisiológicamente normales, generando falsos positivos por confusión de covariable (edad) con patología.[^1][^2]

## Cómo varía la FA con la edad: evidencia cuantitativa

Los estudios de referencia pediátrica muestran un patrón bifásico característico de la FA: un primer pico en el primer año de vida, una meseta hasta el inicio de la pubertad, y un segundo pico marcado durante el estirón puberal (antes en niñas, después en niños), seguido de un descenso continuo hasta valores adultos alrededor de los 18-20 años. El estudio de Zierk et al. (CALIPER/percentiles LMS, más de 100,000 muestras) confirma que los valores de FA no pueden representarse con un intervalo único de 0 a 18 años por esta progresión ondulante, y recomienda tablas de percentiles específicas por edad y sexo. En el estudio brasileño de transferencia CALIPER-Fontes, el rango de FA en niños de 1-9 años (149-301 U/L) es notablemente más alto que en adultos.[^3][^4][^5]

La siguiente tabla resume intervalos de referencia representativos de FA reportados por distintos proyectos, consistentes con la mediana de 320 U/L observada en los menores del dataset ILPD:

| Franja etaria | Sexo | Intervalo de referencia FA | Fuente |
|---|---|---|---|
| 0-14 días | Ambos | 83-248 U/L | Children's Minnesota Lab (basado en normas pediátricas)[^6] |
| 15 días-<1 año | Ambos | 122-469 U/L | Children's Minnesota Lab[^6] |
| 1-<10 años | Ambos | 142-335 U/L | Children's Minnesota Lab[^6] |
| 1-9 años | Ambos | 149-301 U/L | CALIPER-Fontes (Brasil)[^3] |
| 10-<13 años | Ambos | 129-417 U/L | Children's Minnesota Lab[^6] |
| 10-12 años | Ambos | 127-326 U/L | CALIPER-Fontes[^3] |
| 13-<15 años | Varones | 116-468 U/L | Children's Minnesota Lab[^6] |
| 13-<15 años | Mujeres | 57-254 U/L | Children's Minnesota Lab[^6] |
| Pico puberal (niñas ~11 años) | Mujeres | Máximo de la serie (~11 años) | Estudio pediátrico continuo (BINASSS)[^5] |
| Pico puberal (niños ~13 años) | Varones | Máximo de la serie, ~19.7% mayor que niñas | Estudio pediátrico continuo (BINASSS)[^5] |
| 15-<17 años | Varones | 82-331 U/L | Children's Minnesota Lab[^6] |
| 15-<17 años | Mujeres | 50-117 U/L | Children's Minnesota Lab[^6] |
| 17-<19 años | Varones | 55-149 U/L | Children's Minnesota Lab[^6] |
| >17 años | Mujeres | 35-104 U/L | Children's Minnesota Lab[^6] |
| >19 años (adulto) | Ambos | 40-129 U/L | Children's Minnesota Lab[^6] |
| 6-39 años (adulto joven) | Ambos | ~200 U/L rango típico (percentiles CHMS) | Colaboración CHMS-CALIPER[^7] |
| 40-80 años | Ambos | Rango ligeramente distinto al adulto joven | Colaboración CHMS-CALIPER[^7] |

Nótese la coherencia con el caso descrito: los menores (mediana 320 U/L) caen dentro de rangos pediátricos normales (129-468 U/L según edad/sexo), mientras que el "rango adulto" de referencia (~200-215 U/L en el ILPD) corresponde a las franjas de >17-19 años, cuando la FA ya descendió tras el cierre de las placas de crecimiento.[^6][^7]

## Base fisiológica: isoenzima ósea y crecimiento

La FA sérica total es la suma de isoenzimas de distintos tejidos, principalmente hígado, hueso, intestino y placenta; en adultos sanos, la isoenzima ósea (bone ALP) constituye aproximadamente el 40% del total, pero durante el crecimiento activo su contribución se dispara. La FA ósea es producida por osteoblastos para generar concentraciones altas de fosfato inorgánico en la superficie celular durante la mineralización ósea, y actúa como marcador directo de formación ósea. Durante el estirón puberal, los niveles de FA ósea pueden alcanzar 4-5 veces el rango de referencia adulto, explicando por qué la FA total se eleva marcadamente en adolescentes sanos sin ninguna patología hepática.[^8]

Histológicamente, la enzima se localiza en la membrana celular de los osteoblastos y en el frente de mineralización del hueso normal, con actividad particularmente intensa en la zona hipertrófica de la placa de crecimiento (growth plate), la región cartilaginosa donde ocurre el crecimiento longitudinal óseo. Una revisión de Frontiers in Endocrinology confirma que la FA sérica y sus isoenzimas reflejan directamente el metabolismo óseo, siendo clínicamente utilizada para el seguimiento del raquitismo y otras condiciones del crecimiento en pediatría. El proyecto CALIPER resume esto explícitamente: "los niños no son adultos pequeños" — el rápido crecimiento fisiológico y desarrollo durante la infancia y adolescencia influye en la bioquímica, requiriendo interpretación específica por edad y sexo, citando la FA como ejemplo paradigmático de esta variación.[^9][^10][^11][^1]

## Recomendaciones de proyectos de intervalos pediátricos

El proyecto CALIPER (Canadian Laboratory Initiative on Pediatric Reference Intervals), con una biobanco de ~8,640 muestras de niños de 0 a 18 años, ha establecido intervalos pediátricos particionados por edad y sexo para más de 40 marcadores bioquímicos, incluyendo FA. Su metodología estadística sigue estrictamente CLSI C28-A3: examina visualmente los datos para determinar particiones apropiadas de edad y sexo, evalúa estadísticamente las particiones con el método de Harris y Boyd, y solo entonces elimina outliers dentro de cada partición usando el método de Tukey o Tukey ajustado. Esto es clave: CALIPER aplica Tukey *después* de particionar por edad/sexo, no sobre la población mixta completa, precisamente para evitar el problema que el usuario anticipa.[^12][^1]

Otros estudios convergen en la misma conclusión. Zierk et al. (2017) crearon percentiles continuos de FA por edad y sexo desde el nacimiento hasta los 18 años, señalando que son necesarios para incorporación en sistemas de información de laboratorio. Un estudio chino con 6,322 niños y adolescentes (1 a <18 años) encontró diferencias de edad y sexo en prácticamente todos los analitos hepáticos evaluados (FA, ALT, AST, GGT, bilirrubina, albúmina), con hasta 11 particiones necesarias para algunos analitos, concluyendo explícitamente que "deben establecerse intervalos de referencia pediátricos específicos por edad y sexo para ayudar a un diagnóstico preciso". La colaboración CHMS-CALIPER, que combinó ~12,000 canadienses de 3 a 79 años, encontró que la mayoría de los biomarcadores requieren partición por edad y/o sexo a lo largo de toda la vida, con dos o más particiones de edad necesarias para casi todos los analitos excepto el bicarbonato.[^4][^13][^7]

## ¿Es válido combinar pediatría y adultos en un mismo estudio?

Según la guía CLSI/IFCC C28-A3 —el estándar de referencia global para establecer intervalos de laboratorio— la estratificación por edad y sexo es el "prerrequisito mínimo" cuando existe evidencia biológica de diferencia entre subgrupos, y recomienda un mínimo de 120 sujetos por partición para límites de confianza estables. La guía introduce el criterio del "SDR" (standard deviation ratio): un SDR mayor a 0.3 entre subgrupos es indicación estadística de que se debe particionar la variable, en lugar de tratar la muestra como una sola población. Dado que la FA muestra diferencias de varias veces su magnitud entre niños prepúberes, adolescentes en pico puberal y adultos, el SDR entre estos grupos supera ampliamente ese umbral, lo cual convierte la falta de partición en una violación directa del estándar metodológico.[^5][^2][^14]

Por tanto, mezclar pediatría y adultos en un mismo análisis de outliers o en un mismo modelo de biomarcadores hepáticos sin partición por edad no es simplemente subóptimo: se considera un error de diseño según el marco CLSI/CALIPER, porque introduce una variable de confusión (etapa de crecimiento óseo) que domina la variabilidad observada de la FA, más que la variable de interés (función hepática). La recomendación metodológica explícita es: (1) excluir o analizar por separado a los sujetos menores de 18-20 años cuando se estudian enzimas hepáticas en adultos, o (2) si se incluyen, aplicar intervalos de referencia y detección de outliers estratificados por edad (y sexo, cuando aplique) siguiendo CLSI C28-A3/EP28-A3c, replicando el enfoque CALIPER de particionar antes de aplicar la regla de Tukey.[^2][^14][^7][^1]

## Implicación práctica para el pipeline con el dataset ILPD

Para los 25 menores del ILPD, aplicar el filtro de Tukey (1.5·IQR) sobre la población completa de 583 pacientes mezclará el umbral estadístico de "atípico" calculado sobre una distribución dominada por adultos, cuando en realidad los valores de FA de ~320 U/L corresponden al rango fisiológico normal pediátrico/puberal documentado en múltiples fuentes (129-468 U/L según edad y sexo). La recomendación técnica es tratar la edad pediátrica como variable de partición previa al cálculo de IQR, o excluir explícitamente a los menores de 18 años del análisis de outliers de biomarcadores hepáticos adultos, documentando esta decisión como control de calidad de datos antes del modelado.[^3][^5][^6]

---

## References

1. [Healthcare Professionals](https://caliperproject.ca/caliper/healthcare-professionals/) - CALIPER has established paediatric intervals, partitioned according to age and sex, for a number of ...

2. [Reference intervals: current status, recent developments and ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC4783089/) - por Y Ozarda · 2016 · Mencionado por 375 — The selection of reference individuals using a sample que...

3. [reference interval transference from CALIPER to a pediatric ...](https://www.scielo.br/j/jbpml/a/Vd3QtVMNcMg6X7wBWwxTHBs/) - por R Fontes · 2018 · Mencionado por 7 — Results: The ALP RI results (IU/l) obtained were: 149-301 f...

4. [Pediatric reference intervals for alkaline phosphatase](https://pubmed.ncbi.nlm.nih.gov/27505090/) - por J Zierk · 2017 · Mencionado por 149 — We created percentile charts for alkaline phosphatase acti...

5. [[PDF] Pediatric reference values of alkaline phosphatase - BINASSS](https://www.binasss.sa.cr/set23/15.pdf)

6. [Alkaline Phosphatase Total and Isoenzymes](https://www.childrensmn.org/references/lab/chemistry/alkaline-phosphatase-total-and-isoenzymes.pdf)

7. [Biochemical marker reference values across pediatric, ...](https://pubmed.ncbi.nlm.nih.gov/26044506/) - Understanding the fluctuations in biochemical markers over a wide age range provides important insig...

8. [Bone-specific Alkaline Phosphatase](https://pathlabs.rlbuht.nhs.uk/bone_specific_alp.pdf) - It is produced by osteoblasts to provide a high phosphate concentration at the osteoblast cell surfa...

9. [Alkaline phosphatase in clinical practice in childhood: Focus on rickets](https://www.frontiersin.org/journals/endocrinology/articles/10.3389/fendo.2023.1111445/full) - Serum alkaline phosphatase (ALP) and its isoenzymes reflect bone metabolism: ALP increases the ratio...

10. [Enzyme histochemical localisation of alkaline phosphatase ...](https://pubmed.ncbi.nlm.nih.gov/16182549/) - por P Sarathchandra · 2005 · Mencionado por 21 — This confirms the findings of the previous report o...

11. [The growth plate: a physiologic overview - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC7484711/) - The growth plate is the cartilaginous portion of long bones where the longitudinal growth of the bon...

12. [Closing the Gaps in Pediatric Laboratory Reference Intervals: A CALIPER Database of 40 Biochemical Markers in a Healthy and Multiethnic Population of Children](https://academic.oup.com/clinchem/article/58/5/854/5620695?login=false) - AbstractBACKGROUND. Pediatric healthcare is critically dependent on the availability of accurate and...

13. [Establishment of age‐ and sex‐specific reference intervals for serum liver function tests in pediatric population aged 1–<18 years: A prospective study](https://onlinelibrary.wiley.com/doi/10.1002/jcla.23708) - ## Abstract

## Background

The diagnosis, treatment, and prognosis of pediatric diseases rely on th...

14. [Defining, Establishing, and Verifying Reference Intervals in ...](https://webstore.ansi.org/preview-pages/CLSI/preview_CLSI+C28-A3.pdf) - A guideline for global application developed through the Clinical and Laboratory. Standards Institut...

