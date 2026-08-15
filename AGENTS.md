# AGENTS.md — contexto operativo para agentes de IA

Este archivo existe para que cualquier agente (Claude Code u otro) que abra este
repo — desde esta máquina o desde otra, vía OneDrive — recupere el contexto
operativo sin tener que re-derivarlo desde cero. No reemplaza al PRD ni al
CHANGELOG: es un mapa rápido de "cómo trabajamos aquí" y "en qué vamos".

## ⚠️ Primer paso obligatorio para cualquier sesión nueva (agregado 2026-07-26)

El historial de chat **no viaja entre computadoras** — solo este repo lo hace,
vía OneDrive. Por eso, antes de tocar cualquier archivo o continuar el
trabajo, toda sesión/chat nuevo que abra este repo — sobre todo si es en otra
máquina — debe **reconstruir el contexto operativo primero**, en este orden:

1. Leer este archivo completo y el PRD (`../PRD_liver_disease_fases_0-3.md`).
2. Correr `git log --oneline --all --graph` y `git status` para confirmar en
   qué rama y commit está parado el repo, y si hay cambios sin commitear.
3. Revisar la tabla "Estado actual" (más abajo) para saber qué fase está
   cerrada, cuál está en curso y cuál no ha empezado.
4. Si algo del estado real del repo no coincide con lo que dice esta tabla
   (una rama que no aparece aquí, commits más recientes que la última fecha
   registrada, etc.), **decirlo explícitamente al usuario antes de seguir**
   — no asumir ni rellenar huecos de información por cuenta propia.

## Qué es este proyecto

Predicción de enfermedad hepática (ILPD) con auditoría de *fairness* por sexo,
siguiendo CRISP-DM. Hay **dos PRD**, ambos un nivel arriba de este repo, en
`MIU data science/`. Son la fuente de verdad; no se resumen aquí para evitar
que los documentos diverjan — **lee el que corresponda antes de tocar código**:

| PRD | Cubre | Estado |
|---|---|---|
| `../PRD_liver_disease_fases_0-3.md` | **Actividad 1** — Fases 0–3 (EDA y preparación), tareas T1–T8 del enunciado 1 | ✅ Completada y entregada |
| `../PRD_liver_disease_act2_fases_4-9.md` | **Actividad 2** — Fases 4–9 (modelado y evaluación), tareas T1–T8 del enunciado 2 | 📋 Escrito 2026-08-15, **sin empezar** |

⚠️ **No confundir la numeración de tareas.** Cada actividad tiene su propio
T1–T8. "T6" significa cosas distintas en cada PRD (imputación en la Act 1,
elección de métricas en la Act 2). Citar siempre la actividad.

> ⚠️ **El PRD vive FUERA del repositorio y no está versionado.** Decisión
> deliberada y vigente (ratificada 2026-08-15) — ver
> `docs/adr/0005-prd-fuera-del-repositorio.md`. Consecuencias que hay que
> tener presentes: **(a)** quien clone desde GitHub **no recibe la fuente de
> verdad** del proyecto; **(b)** su única copia depende de la sincronización
> de OneDrive, sin historial ni posibilidad de revertir; **(c)** si el PRD
> cambia, nada lo registra. Un agente que abra este repo sin acceso al
> directorio padre debe **decirlo explícitamente** en vez de inferir los
> requisitos desde el código.

## Convenciones de trabajo (acordadas con el usuario)

- **Gitflow por fase.** Cada fase del PRD vive en su propia rama
  `feature/fase-N-nombre`, creada desde `main`. Al cerrar una fase con el visto
  bueno del usuario, se mezcla a `main`. **Una vez mezclada, la rama se borra**
  (local y en el remoto): el historial ya vive en `main`, y dejarlas vivas solo
  ensucia la vista del repo — la tabla "Estado actual" de este archivo es el
  registro de qué fase existió, no la lista de ramas. (Añadido 2026-08-15, tras
  acumular 7 ramas muertas.)
- **Nunca commitear ni mezclar ramas sin permiso explícito.** El agente
  prepara el *working tree* (staging) y propone el mensaje; el usuario da la
  orden de commitear en cada checkpoint. Esto aplica también a los merges de
  rama, salvo mezclas mecánicas necesarias para que la siguiente rama de fase
  arranque con los archivos correctos (se explican igual antes de ejecutarlas).
- **Checkpoint por fase o subfase.** El usuario quiere participar en la
  interpretación y corrección de cada fase — no se avanza a la siguiente sin
  presentar resultados y esperar confirmación.
- **No adelantar fases.** Si una tarea de las Fases 0–3 sugiere entrenar un
  modelo o calcular métricas de clasificación, hay que detenerse: está fuera
  de alcance (ver §2.3 del PRD). Solo dejar notas escritas sobre implicaciones
  futuras.
- **Sin *magic numbers*, sin paths absolutos.** Todo vía `src/config.py` +
  `pathlib`. `RANDOM_STATE = 42` fijo en toda operación estocástica.
- **Ningún número inventado.** Todo lo que se cite en `reports/informe_act1.md`
  debe salir de una celda ejecutada en un notebook.

## Entorno técnico — nota importante

⚠️ **El `venv/` SÍ viaja por OneDrive, y llega roto.** (Corregido 2026-08-15;
antes este archivo afirmaba lo contrario.) `.gitignore` impide que *git* lo
rastree, pero **no impide que OneDrive sincronice la carpeta** — son dos
mecanismos independientes. El resultado es peor que si no viajara: aparece un
`venv/` de aspecto normal cuyo `pyvenv.cfg` apunta al intérprete de **otra**
máquina, y cualquier comando falla con
`did not find executable at 'C:\Users\<otro-usuario>\...'`.

**Primer paso en cada máquina nueva: borrarlo y recrearlo.** No basta con
crearlo "si no existe" — hay que borrar el que llegó por OneDrive:

```bash
rm -rf venv
py -3.13 -m venv venv
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

Verificación rápida de que quedó bien: `load_raw_data()` debe devolver un
DataFrame de **583 × 11** (DoD de la Fase 0).

**Compatibilidad de versión de Python (verificado 2026-07-28):** el entorno
se probó con **3.12 y 3.13**, ambos instalan `requirements.txt` sin
conflictos y pasan la verificación de 583×11. Si una máquina nueva no tiene
3.12 disponible (`py --list` para chequear), usar `py -3.13` sin problema —
no hace falta forzar la versión exacta.

`jupyter` (el metapaquete completo, con Jupyter Lab) **no se pudo instalar**
por el límite de rutas largas de Windows, agravado por la ruta anidada de
OneDrive. En su lugar el entorno usa `ipykernel` + `nbconvert`: suficiente
para editar/ejecutar notebooks desde VS Code y para verificar
*restart & run all* vía `jupyter nbconvert --to notebook --execute`. Ver
`docs/adr/0002-entorno-notebooks-sin-jupyterlab.md` para el detalle y cómo
revertir esta decisión si hace falta Jupyter Lab standalone.

## Estado actual

| Fase | Rama | Estado |
|------|------|--------|
| 0 — Setup | `feature/fase-0-setup` → mezclada a `main` | ✅ Cerrada |
| 1 — Problem Framing | `feature/fase-1-problem-framing` → mezclada a `main` | ✅ Cerrada |
| 2 — EDA | `feature/fase-2-eda` → mezclada a `main` | ✅ Cerrada (T1–T5, Loop A) |
| 2b — EDA clínico | `feature/fase-2-eda-clinico` → mezclada a `main` (2026-07-28) | ✅ Cerrada. Loop C: lectura clínica del EDA, diccionario de datos, 2 problemas de calidad nuevos (Q8 `DB>TB`, Q9 age heaping), Q7 resuelto, y análisis de sensibilidad de umbrales de ALT por sexo. Aprobada por el usuario sin observaciones. |
| 2c — EDA clustering/PCA | `feature/fase-2c-eda-clustering` → mezclada a `main` (2026-07-28) | ✅ Cerrada. Clustering jerárquico edad × sexo, revisado tras observaciones del usuario: se corrigieron los clusters de 1-4 personas (causa: distancias euclídeas sobre variables crudas con asimetría 10) y una contradicción interna. Exp 1 se reporta en dos variantes (crudo vs `log1p`) como análisis de sensibilidad. Incluye control de sesgo del propio análisis (ALT es 4.ª de 9 en separación). Ver CHANGELOG § 2026-07-28. |
| 3 — Preprocessing | `feature/fase-3-preprocessing` → mezclada a `main` (2026-07-28) | ✅ Cerrada. T6, T7 y T8 completos: 39 celdas, 0 errores, 13/13 requisitos (9 `[M]` + 4 `[V]`). Loop B registrado. Aprobada por el usuario. |
| E — Entregables | `feature/fase-e-entregables` → mezclada a `main` (2026-07-28) | ✅ Cerrada. `act1_anexo.ipynb` (E2, 32 celdas, 0 errores) e `informe_act1` en Markdown, LaTeX y **PDF compilado de 10 páginas exactas**. README de portafolio actualizado. |
| Auditoría externa | (sin rama propia; `feature/fase-e-entregables` ya estaba mezclada a `main`, correcciones aplicadas directamente sobre `main`) | ✅ **Aplicada (2026-08-15).** Auditoría metodológica independiente encontró 8 hallazgos; los 8 se corrigieron con decisiones estadísticas/metodológicas aprobadas explícitamente por el usuario. Ver checkpoint más abajo y `docs/CHANGELOG_iteraciones.md` § Loop D. |
| Loop E — revisión de repo | (sin rama propia; aplicado sobre `main`) | ✅ **Aplicado (2026-08-15).** Entorno reproducible reparado, fugas de rutas eliminadas de raíz, 41 cifras publicadas re-verificadas contra recálculo independiente, y **un error numérico real corregido** (límites de intervalo en el chequeo de densidad de ALT del Loop D). Entorno LaTeX resuelto y **PDF recompilado con las cifras corregidas: 10 páginas exactas, 0 errores.** Ver `docs/CHANGELOG_iteraciones.md` § Loop E. |

### Actividad 2 — Fases 4–9 (modelado y evaluación)

| Fase | Contenido | Estado |
|------|-----------|--------|
| 4 — Contrato de datos y *split* | T1, T3: deduplicación, reconstrucción de `A/G Ratio`, split estratificado | ✅ Cerrada (2026-08-15). `feature/fase-4-split-datos` → mezclada a `main` (fast-forward) y borrada. `params.yaml`, `src/data.py`, `src/splitting.py`, `notebooks/04_split.ipynb` (37 celdas, 0 errores, emparejado con `.py` vía jupytext), 12 tests verdes (`test_fase4_split.py` + `test_global.py`), ADR 0006 y 0007. Split congelado en `data/processed/split_indices.json` (no versionado, regenerable). |
| 5 — Pipeline y balanceo | T1, T2: `imblearn.Pipeline`, SMOTE solo en train | ✅ Cerrada (2026-08-15). `feature/fase-5-pipeline-balanceo`: `src/pipelines.py` (`build_pipeline`, fábrica de las 4 combinaciones {minmax,zscore}×{con,sin SMOTE}), `notebooks/05_pipeline.ipynb` (35 celdas, 0 errores, emparejado con `.py`), 8 tests nuevos (`test_fase5_pipeline.py`, 24 tests totales en verde), ADR-0011 (indicador de nulos con `MissingIndicator(features="all")`, no `add_indicator=True`, para blindar el ancho de columnas de cara a la Fase 9). SMOTE verificado: train 456→650 (325/131→325/325, 194 sintéticos), test siempre en 114. **F5-R8 — hallazgo central de la fase:** la comparación correcta no es train completo antes/después (24.56%→24.31%, diluida por las 456 filas reales que no cambian) sino minoría de train vs. sintéticos que la replican: **29.77% de mujeres en la minoría → 23.71% en los 194 sintéticos, brecha −6.06 pp**, descompuesta en dos causas independientes — truncamiento `int64` de `imbalanced-learn` hacia `Male=0` (−3.09 pp; 46 mujeres sintéticas reales vs. 52 con redondeo neutro) y geometría de los `k` vecinos de SMOTE, que no preserva la proporción de sexos aunque no hubiera ningún error de tipo de dato (−2.97 pp). SMOTE subrepresenta mujeres entre los sintéticos por mecánica propia, no por una decisión explícita — fijado en `test_smote_subrepresenta_mujeres_en_sinteticos` y anotado para que la Fase 9 lo retome al comparar la brecha de FNR con/sin SMOTE (F9-R4). `pyproject.toml` añade `ignore = ["RUF001", "RUF002"]` (ruido de puntuación en español); baseline de deuda heredada recalculado de 35 a 24 (`src/utils.py` bajó de 11 a 1). Pendiente de confirmación del usuario antes de mezclar. |
| 6A — Selección de variables | `[V]` Selección **dentro** del `Pipeline`; decisión sobre `Gender` | ⬜ Sin empezar |
| 6 — Algoritmos e hiperparámetros | T4, T5: los 5 algoritmos + Grid Search con rejillas de `params.yaml` | ⬜ Sin empezar |
| 7 — Evaluación | T6, T7: métricas justificadas + tabla comparativa | ⬜ Sin empezar |
| 8 — Experimento factorial | T8: {MinMax, Z-Score} × {SMOTE, sin SMOTE} × 5 modelos — **vale el 25%** | ⬜ Sin empezar |
| 9 — Auditoría de equidad | Valor agregado: FNR por sexo vía CV repetida | ⬜ Sin empezar |
| E2 — Entregables | Informe ≤15 págs + `act2_anexo.ipynb` | ⬜ Sin empezar |
| P — Producción (MLOps) | `src/` + `tests/` + CI/CD + tablero Streamlit | ⬜ Sin empezar |

### ⚠️ Deuda técnica conocida — `ruff check .` (registrada 2026-08-15, Fase 4; actualizada Fase 5)

La Fase 4 configuró `ruff` en `pyproject.toml`
(`extend-select = ["I", "B", "RUF"]`). La Fase 5 añadió
`ignore = ["RUF001", "RUF002"]` (esas dos reglas marcaban la raya larga "–"
y otro puntuación típica del español en strings/docstrings como "carácter
ambiguo" — ruido, no señal, en un proyecto escrito en español; `RUF003`,
la misma regla pero para comentarios `#`, se dejó activa a propósito). Con
la configuración vigente, `ruff check .` da **24 errores preexistentes**,
ninguno introducido en la Fase 4 ni en la Fase 5 (el código nuevo de ambas
está limpio) — **no se tocan hasta que se monte CI** (Fase P), porque
corregirlos en los notebooks obligaría a re-ejecutarlos y volver a
verificar cifras ya entregadas:

| Archivo | Errores | Reglas típicas |
|---|---|---|
| `notebooks/03_preprocessing.ipynb` | 7 | mixto |
| `notebooks/02_eda.ipynb` | 6 | mixto (incluye `F401` import sin usar) |
| `notebooks/act1_anexo.ipynb` | 5 | mixto |
| `notebooks/02c_eda_clustering.ipynb` | 5 | mixto (incluye `I001` imports desordenados) |
| `src/utils.py` | 1 | `B905` (`zip()` sin `strict=`) |

`src/utils.py` bajó de 11 a 1 con el `ignore` de la Fase 5 (casi toda su
deuda era `RUF002` en docstrings). No es un notebook — su deuda solo se hizo
visible al activar `B`/`RUF` en la Fase 4 (antes no había `pyproject.toml`
con reglas extendidas). Verificar el conteo vigente con
`ruff check . --statistics` antes de asumir que sigue en 24.

**Decisiones ya tomadas por el usuario (2026-08-15), antes de empezar:**

- **Nulos.** Los 4 de `A/G Ratio` se **reconstruyen** con `ALB/(TP−ALB)`
  (4.9× más preciso que la media; error de redondeo declarado). Las 3 filas con
  `DB>TB` se **conservan e imputan dentro del `Pipeline`** — por mediana, con
  `add_indicator=True`, y con análisis de sensibilidad obligatorio. §5 del PRD.
- 🔴 **Duplicados: se deduplica ANTES del split (583 → 570).** Divergencia
  deliberada respecto de la Act 1, donde conservarlos era correcto porque no se
  entrenaba nada. Motivo: con `random_state=42`, **5 de los 13 pares idénticos
  caen a ambos lados del split** y el modelo memoriza la fila. Nunca es cero en
  ninguna semilla probada. §5.3 del PRD.
- **Fase 9 (equidad) se incluye**, aunque el enunciado no la pida.
- **Sin feature engineering.** Verificado contra el enunciado: ninguna de las 8
  tareas lo pide. **Sí hay selección de variables** (Fase 6A), que es cosa
  distinta y va *después* del engineering en el orden canónico. §9 del PRD.

> ⚠️ **Antes de tocar nada en la Act 2, correr `pytest tests/ -v`.** El PRD
> exige tests por fase precisamente para que una sesión nueva sepa qué está
> realmente cerrado sin fiarse de la documentación. Protocolo en §0.1 del PRD.

> Actualiza estas tablas al cerrar cada fase. Son lo primero que debe leer un
> agente nuevo para saber dónde retomar.

## 🔖 Checkpoint — Fase E completa, proyecto entregable (2026-07-28)

**Las 8 tareas del enunciado están respondidas y los dos entregables existen.**

### Entregables

| ID | Archivo | Estado |
|---|---|---|
| **E1** | `reports/informe_act1.pdf` | **10 páginas exactas**, 0 errores de compilación |
| E1 (fuentes) | `reports/informe_act1.md` y `.tex` | Mismo contenido y mismos números |
| **E2** | `notebooks/act1_anexo.ipynb` | 32 celdas, 0 errores, orden literal T1→T8 |

### Entorno LaTeX (agregado 2026-07-28; ✅ resuelto y verificado 2026-08-15)

**MiKTeX instalado en modo usuario**, con `AutoInstall=1` para descargar
paquetes de CTAN al vuelo. **No escribas la ruta a mano** — la versión anterior
de este archivo tenía la ruta de otra máquina, con el nombre de usuario
incrustado, y por eso pareció durante un tiempo que MiKTeX no estaba instalado.
Resuélvela siempre desde el entorno:

```powershell
# pdflatex ya suele estar en el PATH del usuario; comprobar primero:
(Get-Command pdflatex -ErrorAction SilentlyContinue).Source
# si no aparece, la carpeta es:
"$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
```

Compilación (dos pasadas, por las referencias cruzadas):

```powershell
cd reports
pdflatex -interaction=nonstopmode --disable-installer informe_act1.tex
pdflatex -interaction=nonstopmode --disable-installer informe_act1.tex
```

> 🔴 **Usa siempre `--disable-installer` en sesiones automatizadas.** Sin esa
> bandera, si falta un paquete MiKTeX intenta descargarlo de un mirror de CTAN;
> si el mirror falla, `pdflatex` **se queda colgado para siempre** con la
> conexión en `CloseWait`, sin consumir CPU y sin escribir nada al log — parece
> que trabaja, pero está muerto. Pasó el 2026-08-15 y costó un rato
> diagnosticarlo. Con la bandera puesta, falla al instante y **dice qué archivo
> `.sty` falta**, que es lo que quieres saber. Si falta alguno, instálalo
> aparte (`mpm --install=<paquete>`) y vuelve a compilar.

**Verificación obligatoria tras cada compilación** — el informe debe tener
**10 páginas exactas**:

```powershell
Select-String informe_act1.log -Pattern 'Output written.*pages'
```

Los auxiliares (`.aux`, `.log`, `.out`, `.toc`) están en `.gitignore`; el
`.tex` y el `.pdf` sí se versionan. MiKTeX también trae `pdftotext` (para
verificar que una cifra corregida llegó al PDF) y `pdftoppm` (para renderizar
a PNG y revisarlo visualmente).

> ⚠️ **Lección registrada:** la estimación de páginas desde el Markdown (11.3)
> resultó **muy optimista** — el PDF real salió en **14 páginas**. No estimar
> la extensión sin compilar. Se llegó a 10 con: 10pt, márgenes 1.95 cm,
> `titlesec`, tablas en `\footnotesize` y figuras al 66–88% del ancho. Los
> mandos están agrupados y comentados en el preámbulo del `.tex`.

### Qué queda

- Fases 4–7 (modelado, auditoría de *fairness*, pipeline, dashboard):
  requieren un PRD nuevo. Las notas de traspaso están en la última celda de
  cada notebook modular.

## 🔖 Checkpoint — Auditoría externa aplicada (2026-08-15)

Una revisión metodológica independiente auditó el proyecto completo y
encontró 8 hallazgos (2 críticos, 2 altos, 3 medios, 1 bajo). Los 8 se
corrigieron. Las decisiones con juicio estadístico/metodológico real se
presentaron al usuario en español simple, sin ambigüedad, con los archivos
que se iban a tocar, y **se ejecutaron solo tras su aprobación explícita**.

| # | Hallazgo | Decisión aprobada | Archivos tocados |
|---|---|---|---|
| 1 | Brecha de 8.7 pp sin prueba de significancia | Agregar test exacto de Fisher (p=0.055, IC cruza 0) + suavizar "hallazgo principal" | `02_eda.ipynb`, `informe_act1.md/.tex`, `CHANGELOG` |
| 2 | Circularidad en el "cluster de severidad" (02c) | Agregar chequeo de solape cluster-vs-TB (Jaccard 0.54–0.81) + corregir tabla de honestidad | `02c_eda_clustering.ipynb` |
| 3 | Límites biológicos de T8 sin fuente | Sgot: fuente real (Chang et al. 2007); TP/ALB/A-G Ratio: declarados como estimación del equipo | `src/config.py`, `03_preprocessing.ipynb`, `act1_anexo.ipynb`, `Consulta_5.md`, `informe_act1.md/.tex` |
| 4 | Whipple aplicado fuera de su dominio censal | Párrafo de limitación explícito + nota abierta sobre posible heaping en `Age=60` | `data_dictionary.md`, `informe_act1.md/.tex` |
| 5 | Atribución de fuente inflada (ALP pediátrica) | Corregir cita: mayoría de filas son de un sitio de hospital, no de CALIPER/Zierk | `02_eda.ipynb`, `CHANGELOG`, `data_dictionary.md` |
| 6 | T3 incluye 2 valores ya calificados "imposibles" | Nota de sensibilidad (con/sin esas filas), sin tocar los números oficiales | `02_eda.ipynb`, `act1_anexo.ipynb`, `informe_act1.md/.tex` |
| 7 | Nulos residuales sin flag en `data/processed/` | Nota explícita en el diccionario de datos | `data_dictionary.md` |
| 8 | Caveat de ALT sin cuantificar | Chequeo de densidad por unidad de ancho (mujeres ~40% más concentradas) | `02_eda.ipynb`, `informe_act1.md/.tex` |

✅ **Nota sobre el Hallazgo 4 — cerrado (2026-08-15).** Se le pasó al usuario
por alto en la ronda de aprobación explícita (error del agente) y se resolvió
de la forma de bajo riesgo ya anunciada (solo texto de limitación, ninguna
conclusión ni número existente se retracta). **El usuario revisó los tres
fragmentos exactos y los aprobó explícitamente después del hecho.** No quedan
cambios del Loop D sin aprobar.

Los 4 notebooks modificados (`02_eda.ipynb`, `02c_eda_clustering.ipynb`,
`03_preprocessing.ipynb`, `act1_anexo.ipynb`) se re-ejecutaron *restart & run
all* después de cada cambio — **0 errores**. El PDF se recompiló.

### Advertencias permanentes

- ⚠️ **El usuario no tiene formación clínica.** Explicar desde cero, con
  redundancia, antes de concluir. Ver el protocolo al final de este archivo.
- ⚠️ **No commitear ni mezclar sin orden explícita** (convención del proyecto).

## Dónde está cada tipo de decisión documentada

- **Decisiones de ingeniería/tooling** (con consecuencias duraderas, ej. por
  qué gitflow por fase, por qué este entorno de notebooks) → `docs/adr/`.
  A hoy hay cuatro: `0001-gitflow-por-fase`, `0002-entorno-notebooks-sin-jupyterlab`,
  `0003-sin-holdout-en-fases-0-3` (por qué no se aparta un holdout ahora y de
  dónde saldrá el *golden set*) y `0004-umbrales-referencia-sexo-especificos`
  (qué rangos clínicos se adoptan y por qué).
- **Hallazgos de datos que disparan un loop CRISP-DM** (ej. el EDA revela algo
  que obliga a revisar Fase 1 o Fase 3) → `docs/CHANGELOG_iteraciones.md`.
- **Significado clínico de cada variable**, unidades, rangos de referencia y
  tabla consolidada de problemas de calidad → `docs/data_dictionary.md`.
- **Resultados e interpretación académica** (las 8 tareas del enunciado) →
  `reports/informe_act1.md`.

## Protocolo de fundamentación clínica (agregado 2026-07-27, Loop C)

El equipo **no tiene formación en salud**. Por eso ninguna afirmación clínica,
umbral ni criterio metodológico entra al proyecto sin fuente citable. El método
acordado:

1. Formular una consulta de investigación profunda con **CONSULTA + CONTEXTO**
   explícitos (qué se pregunta, sobre qué datos, y qué salida se necesita).
2. Ejecutarla contra fuentes primarias (guías de sociedades médicas, estudios
   de intervalos de referencia, marcos de calidad de datos).
3. **Versionar la respuesta completa con sus referencias en `docs/fuentes/`.**

Las cinco consultas realizadas cubren: rangos de referencia por sexo · ALP
dependiente de edad · cociente De Ritis · restricciones bioquímicas y calidad
de captura · límites biológicamente posibles para T8/F3-R12 (`Consulta_5.md`,
añadida 2026-08-15 tras una auditoría externa que señaló que esos límites no
tenían fuente citada — ver checkpoint más abajo). **Este protocolo es parte
del valor del proyecto, no un andamio temporal** — hace cada número auditable,
incluido frente a revisiones externas.

> ⚠️ **Al explicar hallazgos al usuario:** no asumir conocimiento clínico.
> Explicar primero qué mide la variable, su unidad y su rango de referencia;
> después la conclusión; y siempre separar lo que el dato **prueba** de lo que
> solo **sugiere**.
