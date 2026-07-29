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
| 2c — EDA clustering/PCA | `feature/fase-2c-eda-clustering` (creada desde `main` en `52d2477`) | 🔄 **Revisada tras observaciones del usuario (2026-07-28).** Se corrigieron dos fallos metodológicos: clusters de 1-4 personas (causa: clustering euclídeo sobre variables crudas con asimetría 10) y una contradicción interna sobre `60-120 · Female`. Ahora el Exp 1 se corre en dos variantes (crudo vs `log1p`) como análisis de sensibilidad. Ver CHANGELOG § 2026-07-28. Pendiente de tu visto bueno y merge. |
| 3 — Preprocessing | `feature/fase-3-preprocessing` (creada desde `main`) | 🔄 **En planificación.** Rama abierta, sin código todavía — pendiente de acordar el diseño de imputación (ver checkpoint abajo) antes de escribir la primera celda. |
| E — Entregables | — | ⏳ No iniciada |

> Actualiza esta tabla al cerrar cada fase. Es lo primero que debe leer un
> agente nuevo para saber dónde retomar.

## 🔖 Checkpoint abierto — traspaso del 2026-07-28 (noche), cambio de máquina

**El usuario cierra esta sesión acá y la retoma en otra computadora, vía
OneDrive.** Este commit es un **checkpoint de trabajo en progreso**: el
contenido de `02c_eda_clustering.ipynb` **corre sin errores pero todavía no
fue revisado a fondo por el usuario** — no tratarlo como aprobado. La sesión
nueva debe esperar sus observaciones antes de seguir, igual que se hizo con
Loop C.

**Rama activa:** `feature/fase-2c-eda-clustering` (creada desde `main` en
`52d2477`). `feature/fase-3-preprocessing` existe en paralelo, mismo punto
de partida, todavía sin código — el usuario decidió empezar por la sub-fase
2c primero.

**Qué hay en `02c_eda_clustering.ipynb` (19 celdas, ejecuta de punta a punta,
`nbconvert --execute` verificado):**

1. Exclusión documentada de 7 filas con problemas de calidad ya conocidos
   (4 `A/G Ratio` faltante, 3 `DB>TB`) → n=576 para este notebook exploratorio
   únicamente (no toca la imputación formal de Fase 3).
2. Estratificación edad×sexo: 7 grupos (`assign_age_sex_stratum` en
   `src/utils.py`), banda 0-17 fusionada por tamaño chico
   (`MIN_STRATUM_SIZE_FOR_SEX_SPLIT=30` en `src/config.py`).
3. **Experimento 1** (clustering jerárquico, Ward, sobre las 9 variables):
   encuentra un cluster chico de "severidad multivariada" en las 3 bandas
   adultas — `TB` muy elevada, ~100% diagnosticado. Dendrogramas en
   `reports/figures/fase2c_exp1_dendrogramas.png`.
4. **Experimento 2** (clustering solo sobre `Sgpt`/`Sgot`, crudo y `log1p`,
   pedido explícito del usuario): confirma que reducir a 2 variables no
   resuelve un corte fino comparable a la literatura. Conclusión reportada:
   el ILPD es cohorte hospitalaria (71% ya diagnosticados), sin la franja de
   sanos que un método data-driven necesita para encontrar el mismo límite
   que Prati/ACG/AASLD calibraron con muestras poblacionales — **refuerza,
   con evidencia empírica directa, la decisión ya tomada en el ADR 0004** de
   no derivar umbrales propios. Dendrogramas en
   `reports/figures/fase2c_exp2_dendrogramas.png`.
5. Valor agregado: De Ritis por cluster — diferencia grande (casi el doble)
   solo en 2 de 7 estratos (`40-59 · Male`, `60-120 · Female`), reportado
   como hipótesis abierta y específica, no como patrón general.
6. Tabla de honestidad metodológica + notas para Fase 4+/Fase 5.

**Funciones nuevas reutilizables en `src/utils.py`:** `age_band_label`,
`assign_age_sex_stratum`, `hierarchical_cluster_cut` (corta el dendrograma
en el mayor salto de distancia de fusión — sin fijar `k` a mano).
Constantes nuevas en `src/config.py`: `CLUSTER_LINKAGE_METHOD`,
`MIN_STRATUM_SIZE_FOR_SEX_SPLIT`.

**Lo primero que debe hacer la sesión en la otra máquina:**

1. Recrear el `venv` (ver "Entorno técnico" arriba) — se borró a propósito
   antes de cerrar esta sesión, como siempre. 3.12 o 3.13 funcionan bien,
   ambos verificados.
2. Abrir `notebooks/02c_eda_clustering.ipynb` y leer con el usuario — él va
   a traer preguntas de una revisión propia, probablemente sobre el
   Experimento 2, la explicación del sesgo de cohorte hospitalaria, o el
   hallazgo de De Ritis en los 2 estratos.
3. **No avanzar a construir más análisis de clustering, ni tocar
   `feature/fase-3-preprocessing`, hasta que el usuario dé su visto bueno o
   pida cambios concretos sobre este notebook.**

**Advertencias para el agente de la sesión nueva:**

- ⚠️ **El usuario no tiene formación clínica.** Explicar desde cero, con
  redundancia, antes de concluir. Ver el protocolo al final de este archivo.
- ⚠️ **No commitear ni mezclar sin orden explícita** (convención del
  proyecto) — este checkpoint es la excepción ya autorizada para cerrar la
  sesión de forma trazable, no un cambio de política.

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
