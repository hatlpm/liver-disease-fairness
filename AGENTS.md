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
siguiendo CRISP-DM. Especificación completa en
`../PRD_liver_disease_fases_0-3.md` (un nivel arriba de este repo, en
`MIU data science/`). Ese PRD es la fuente de verdad para las Fases 0–3; no
se resume aquí para evitar que ambos documentos diverjan — léelo antes de
tocar código.

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
| Loop E — revisión de repo | (sin rama propia; aplicado sobre `main`) | ✅ **Aplicado (2026-08-15).** Entorno reproducible reparado, fugas de rutas eliminadas de raíz, 41 cifras publicadas re-verificadas contra recálculo independiente, y **un error numérico real corregido** (límites de intervalo en el chequeo de densidad de ALT del Loop D). Ver `docs/CHANGELOG_iteraciones.md` § Loop E. ⏸️ Deja el PDF desactualizado — ver aviso de LaTeX. |

> Actualiza esta tabla al cerrar cada fase. Es lo primero que debe leer un
> agente nuevo para saber dónde retomar.

## 🔖 Checkpoint — Fase E completa, proyecto entregable (2026-07-28)

**Las 8 tareas del enunciado están respondidas y los dos entregables existen.**

### Entregables

| ID | Archivo | Estado |
|---|---|---|
| **E1** | `reports/informe_act1.pdf` | **10 páginas exactas**, 0 errores de compilación |
| E1 (fuentes) | `reports/informe_act1.md` y `.tex` | Mismo contenido y mismos números |
| **E2** | `notebooks/act1_anexo.ipynb` | 32 celdas, 0 errores, orden literal T1→T8 |

### Entorno LaTeX (agregado 2026-07-28; ⏸️ pendiente de revisar 2026-08-15)

**MiKTeX 25.12 instalado** vía `winget` en modo usuario, con `AutoInstall=1`
para que descargue paquetes de CTAN al vuelo. `pdflatex` **no está en el PATH
del sistema**: hay que añadir su carpeta `miktex\bin\x64` en cada sesión antes
de compilar.

> ⏸️ **PENDIENTE — no resuelto a 2026-08-15.** La ruta que estaba escrita aquí
> (`C:\Users\<otro-usuario>\AppData\Local\Programs\MiKTeX\...`) era de **otra máquina** y
> no existe en la actual. El usuario confirma que MiKTeX **sí está instalado**
> en este equipo, pero se decidió **no abordarlo en esta ronda**. Queda por
> hacer: localizar `pdflatex.exe` en esta máquina y sustituir la ruta por una
> forma que no dependa del nombre de usuario (p. ej. resolver la carpeta desde
> `$env:LOCALAPPDATA`). **Hasta entonces, el PDF no se puede recompilar aquí.**
>
> 🔴 **CONSECUENCIA ACTIVA: `reports/informe_act1.pdf` está DESACTUALIZADO.**
> El Loop E (2026-08-15) corrigió dos cifras en `informe_act1.md` y `.tex` que
> el PDF **todavía muestra con el valor viejo**:
>
> | Dónde | PDF (viejo) | Fuentes `.md`/`.tex` (correcto) |
> |---|---|---|
> | §10.1, densidad masculina de ALT | `0.0168/U·L` | **`0.0156/U·L`** |
> | §10.1, exceso de densidad femenina | `~40% mayor` | **`exactamente 50% mayor`** |
> | §10.4, iteraciones CRISP-DM | `Cuatro` | **`Cinco`** |
>
> **El `.md` y el `.tex` son la versión correcta; el PDF no.** Es un cambio de
> texto corrido, así que no debería alterar la paginación, pero **hay que
> verificar que sigan siendo 10 páginas exactas** al recompilar. Hasta
> entonces, **no entregar ni publicar el PDF**.

MiKTeX también trae `pdftoppm`, útil para renderizar el PDF a PNG y revisarlo
visualmente.

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

⚠️ **Nota sobre el Hallazgo 4:** se le pasó al usuario por alto en la ronda de
aprobación explícita (error del agente) y se resolvió de la forma de bajo
riesgo ya anunciada (solo texto de limitación, ninguna conclusión ni número
existente se retracta). Queda señalado para que el usuario lo confirme
después del hecho.

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
