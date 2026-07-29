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

## Convenciones de trabajo (acordadas con el usuario)

- **Gitflow por fase.** Cada fase del PRD vive en su propia rama
  `feature/fase-N-nombre`, creada desde `main`. Al cerrar una fase con el visto
  bueno del usuario, se mezcla a `main`.
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

**El `venv/` no viaja por OneDrive** (está en `.gitignore`, como debe ser). En
cada máquina nueva hay que recrearlo antes de ejecutar cualquier notebook:

```bash
py -3.12 -m venv venv
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
| E — Entregables | `feature/fase-e-entregables` | 🔄 **En curso.** Falta `act1_anexo.ipynb` (E2) e `informe_act1.md` (E1). Con la Fase 3 cerrada, **las 8 tareas del enunciado ya están respondidas** — la Fase E solo consolida y redacta. |

> Actualiza esta tabla al cerrar cada fase. Es lo primero que debe leer un
> agente nuevo para saber dónde retomar.

## 🔖 Checkpoint abierto — Fase 3 escrita, pendiente de revisión (2026-07-28)

`notebooks/03_preprocessing.ipynb` está **completo y ejecutado** (39 celdas,
0 errores) en `feature/fase-3-preprocessing`. **No mezclado a `main`** —
espera la revisión del usuario.

### Las tres decisiones de diseño, ya acordadas con el usuario

| # | Decisión | Elegida |
|---|---|---|
| 1 | 13 duplicados exactos (F3-R13) | **Conservar**, con análisis de sensibilidad |
| 2 | Los 6 nulos que crea marcar `DB > TB` | **Dejar como faltantes**, no imputar |
| 3 | `log1p` en T7 | **Tercera columna**, después del comparativo obligatorio |

### Resultados principales

- **T6:** la imputación por media falla en 3 de 4 filas (error de hasta 34%
  contra el valor determinista). *Variance shrinkage* medido: −0.687%.
  Eliminar los 13 duplicados movería las medias solo +0.70% / +0.77%.
- **T7:** con MinMax, el **95.9%** de los pacientes de `Sgot` queda bajo 0.1
  y la mediana en el 0.65% del rango. Conclusión: **Z-Score**.
- **T8:** 66 atípicos en `Sgot`, 8 en `TP`, **0 en `ALB`**, 10 en
  `A/G Ratio`. **Ninguno es biológicamente imposible** → se conservan todos.
- **Loop B registrado** en el CHANGELOG: el umbral único de Tukey subdetecta
  en mujeres (`Sgot` 7.0% global vs 10.6% propio) y el sesgo etario **se
  invierte** según la variable.

### Salidas

3 datasets en `data/processed/` (gitignored, se regeneran al ejecutar):
`ilpd_procesado.csv`, `ilpd_zscore.csv`, `ilpd_log_zscore.csv`.
2 figuras nuevas: `fase3_t7_escalado.png`, `fase3_t8_boxplots.png`.

### Qué sigue si el usuario aprueba

Merge a `main` y arrancar la **Fase E** (entregables): `act1_anexo.ipynb`
consolidado en orden T1→T8, e `informe_act1.md` (≤10 páginas, sin código,
con números reales de celdas ejecutadas).

### Reutilizable en `src/`

`utils.py`: `age_band_label`, `assign_age_sex_stratum`,
`hierarchical_cluster_cut`, `cluster_sizes`, `cliffs_delta`,
`separation_by_variable`, `whipple_index`, `de_ritis_ratio`,
`flag_biochemical_violations`, `alt_threshold_comparison`.
`config.py`: umbrales de ALT, rangos de referencia, `SKEWED_COLS`,
`IQR_MULTIPLIER`, paleta de figuras validada.

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

Las cuatro consultas realizadas cubren: rangos de referencia por sexo · ALP
dependiente de edad · cociente De Ritis · restricciones bioquímicas y calidad
de captura. **Este protocolo es parte del valor del proyecto, no un andamio
temporal** — hace cada número auditable.

> ⚠️ **Al explicar hallazgos al usuario:** no asumir conocimiento clínico.
> Explicar primero qué mide la variable, su unidad y su rango de referencia;
> después la conclusión; y siempre separar lo que el dato **prueba** de lo que
> solo **sugiere**.
