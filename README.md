# Liver Disease Fairness

Análisis exploratorio y preparación de datos del **ILPD** (*Indian Liver Patient Dataset*, 583 pacientes) con **auditoría de equidad por sexo**, siguiendo CRISP-DM.

## El problema

El hígado no duele. La enfermedad hepática crónica avanza durante años sin síntomas, y cuando aparecen los signos visibles el daño suele ser irreversible. El diagnóstico temprano depende casi por completo de un análisis de sangre barato, lo que lo convierte en buen candidato para un modelo de cribado.

Pero este dataset tiene **441 hombres y 142 mujeres**, y Straw & Wu (2022) ya documentaron que los modelos entrenados sobre él producen **tasas de falsos negativos hasta 24 puntos peores en mujeres**. El proyecto no se limita a un EDA genérico: cada análisis se hace también estratificado por sexo.

## Una brecha observada, y un mecanismo candidato

El dataset presenta una brecha en la tasa de diagnóstico positivo: **73.47% en hombres frente a 64.79% en mujeres**, 8.7 puntos porcentuales. Con solo 142 mujeres, esa diferencia **no alcanza la significancia convencional**: test exacto de Fisher, **p = 0.055**, con el intervalo de confianza cruzando el cero. No puede descartarse que sea ruido muestral.

Describir la brecha es fácil. Lo que aporta este trabajo es **medir un mecanismo concreto** que, si la brecha es real, contribuiría a producirla.

### El mecanismo: el umbral de ALT

Durante décadas los laboratorios usaron **un solo umbral de ALT (40 U/L)** para ambos sexos. Prati et al. (2002) establecieron que las mujeres sanas tienen valores naturalmente más bajos: el límite real es **30 U/L en hombres y 19 U/L en mujeres**. Un umbral único de 40 queda calibrado, sin proponérselo, mucho más cerca del techo masculino.

Al aplicar ambos criterios a esta cohorte:

| | Umbral unisex (40) | Umbral por sexo | Cambio |
|---|---|---|---|
| **Mujeres** (n=142) | 27.5% anormales | **76.8%** | **+49.3 pp** |
| **Hombres** (n=441) | 47.2% anormales | 62.8% | +15.6 pp |

Cambiar el criterio reclasifica a **la mitad de las mujeres** y a **una sexta parte de los hombres**: un impacto tres veces mayor sobre ellas, que además **invierte la ordenación entre sexos**.

Parte de esa asimetría es aritmética —la ventana femenina (19–40] es más ancha que la masculina (30–40]—, así que se midió cuánto queda tras descontar el ancho: la densidad femenina en su franja (0.0235 personas por U/L) es **exactamente un 50% mayor** que la masculina (0.0156). Una ventana ancha solo atrapa gente si hay gente dentro.

### El mismo mecanismo, en otra etapa del pipeline

Reaparece en el tratamiento de valores atípicos: el umbral global de Tukey detecta el **7.0%** de las mujeres en `Sgot` frente al **10.6%** que detectaría su propio grupo. Los cuartiles globales se calculan sobre una muestra 75.6% masculina, de modo que están calibrados esencialmente con hombres.

### Qué se puede y qué no se puede concluir

**Puede afirmarse** que la población femenina está desproporcionadamente concentrada en el rango donde la elección del umbral decide el resultado, y que **un pipeline de datos aparentemente neutro puede introducir sesgo por sexo en al menos dos puntos independientes**, ambos medibles **antes de entrenar modelo alguno**.

**No puede afirmarse** que los clínicos usaran el umbral de 40, que esas mujeres estén mal diagnosticadas, ni que esto explique la brecha. Es un **análisis de sensibilidad**: no prueba una causa, mide cuánto depende un resultado de una decisión metodológica.

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 0 — Setup | Estructura, entorno, `config.py` | ✅ |
| 1 — Problem Framing | Contexto, preguntas de investigación, limitaciones | ✅ |
| 2 — EDA | Tareas 1–5: estructura, calidad, descriptivos, distribuciones, correlación | ✅ |
| 2b — EDA clínico | Lectura por eje funcional, diccionario de datos, sensibilidad de umbrales | ✅ |
| 2c — Clustering | Clustering jerárquico edad × sexo, análisis de sensibilidad | ✅ |
| 3 — Preprocessing | Tareas 6–8: imputación, escalado, valores atípicos | ✅ |
| E — Entregables | Informe y notebook anexo | ✅ |

Las fases 4–7 (modelado, auditoría de *fairness*, pipeline, dashboard) quedan para un PRD posterior.

## Estructura

```
liver-disease-fairness/
├── data/
│   ├── raw/                      dataset original (INMUTABLE)
│   └── processed/                datasets tratados (se regeneran al ejecutar)
├── notebooks/
│   ├── 01_problem_framing.ipynb
│   ├── 02_eda.ipynb              Tareas 1-5 + lectura clínica
│   ├── 02c_eda_clustering.ipynb  clustering exploratorio (valor agregado)
│   ├── 03_preprocessing.ipynb    Tareas 6-8
│   └── act1_anexo.ipynb          ← ENTREGABLE: código en orden T1→T8
├── reports/
│   ├── informe_act1.md           ← ENTREGABLE: informe sin código
│   └── figures/                  PNG a 300 dpi
├── docs/
│   ├── data_dictionary.md        significado clínico y rangos de referencia
│   ├── CHANGELOG_iteraciones.md  iteraciones CRISP-DM (Loops A–E)
│   ├── adr/                      decisiones de ingeniería (5 ADR)
│   └── fuentes/                  investigación bibliográfica versionada
├── src/                          config.py, utils.py
└── requirements.txt
```

> La especificación del proyecto (`PRD_liver_disease_fases_0-3.md`) es material del curso y **vive fuera de este repositorio**, sin versionar. Decisión deliberada y sus consecuencias en [`docs/adr/0005`](docs/adr/0005-prd-fuera-del-repositorio.md). El repo está escrito para poder leerse sin ella.

## Cómo reproducir

```bash
rm -rf venv                    # ver aviso abajo
py -3.13 -m venv venv          # 3.12 y 3.13 verificados
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

> ⚠️ **Si el proyecto llegó por una carpeta sincronizada (OneDrive, Drive), borra el `venv/` antes de recrearlo.** `.gitignore` impide que git lo rastree, pero no impide que el sincronizador copie la carpeta: llega un `venv/` de aspecto normal cuyo intérprete apunta a la máquina de origen, y todo falla con `did not find executable at 'C:\Users\<otro-usuario>\...'`.

Verificación rápida de que quedó bien: `load_raw_data()` debe devolver un DataFrame de **583 × 11**.

Para ejecutar un notebook de principio a fin y verificar que no falla:

```bash
./venv/Scripts/python.exe -m nbconvert --to notebook --execute notebooks/act1_anexo.ipynb
```

> El entorno usa `ipykernel` + `nbconvert` en lugar del metapaquete `jupyter`, que no se puede instalar por el límite de rutas largas de Windows agravado por la ruta de OneDrive. Ver `docs/adr/0002-entorno-notebooks-sin-jupyterlab.md`.

Los cinco notebooks pasan *restart & run all* sin errores, sin rutas absolutas y con `RANDOM_STATE = 42` fijo. Verificado por última vez el **2026-08-15**, en una máquina distinta de aquella en que se escribieron.

## Rigor metodológico

- **Ningún número inventado.** Cada valor del informe procede de una celda ejecutada.
- **Ninguna afirmación clínica sin fuente.** Umbrales y criterios proceden de fuentes primarias versionadas en `docs/fuentes/` (guías ACG/AASLD, estudios de intervalos de referencia, proyecto CALIPER, marco de calidad de datos de Kahn et al.).
- **Iteraciones documentadas.** Cinco *loops* CRISP-DM registrados con fecha, disparador y decisión — incluidos los dos que nacieron de revisiones externas al equipo.
- **Los errores propios se publican, no se borran.** Cuando una revisión encuentra un fallo, la corrección queda registrada junto al número que la motivó. El Loop E, por ejemplo, corrigió los límites de un intervalo que hacían que un conteo no coincidiera con la tabla del propio informe; el resultado se movió en la dirección contraria a la conveniente para la tesis del trabajo, y así se reporta.
- **Conclusiones atenuadas cuando la evidencia no alcanza.** Varias hipótesis se reportan como "sugerente, no probatorio" tras contrastarlas con las fuentes.

## Referencias principales

- Prati, D. et al. (2002). *Updated definitions of healthy ranges for serum alanine aminotransferase levels.* Ann Intern Med, 137(1). https://doi.org/10.7326/0003-4819-137-1-200207020-00006
- Straw, I. & Wu, H. (2022). *Investigating for bias in healthcare algorithms: a sex-stratified analysis of supervised machine learning models in liver disease prediction.* BMJ Health Care Inform, 29(1). https://doi.org/10.1136/bmjhci-2021-100457
- Ramana, B. V. & Venkateswarlu, N. B. *ILPD.* UCI Machine Learning Repository. https://doi.org/10.24432/C5D02C
