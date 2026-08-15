# ADR — Architecture / Engineering Decision Records

Registro de decisiones de **ingeniería y tooling** con consecuencias
duraderas para el proyecto (estructura del repo, flujo de git, elección de
entorno, etc.).

**Qué NO va aquí:** hallazgos sobre los datos que disparan un loop CRISP-DM
(eso va en `../CHANGELOG_iteraciones.md`) ni resultados/interpretación
académica (eso va en `../../reports/informe_act1.md`).

## Formato

Un archivo por decisión, numerado secuencialmente: `000N-titulo-corto.md`.

```markdown
# N. Título de la decisión

**Fecha:** AAAA-MM-DD
**Estado:** Aceptada | Superada por ADR-000X

## Contexto
Qué problema o disyuntiva obligó a decidir.

## Decisión
Qué se decidió, en una o dos frases.

## Consecuencias
Qué gana el proyecto y qué costo/limitación acepta a cambio.
```

Los ADR son **inmutables una vez aceptados**: si una decisión cambia, se crea
un ADR nuevo que marca al anterior como "Superada por ADR-000X" — no se
edita el original.

## Índice

| ADR | Título |
|-----|--------|
| [0001](0001-gitflow-por-fase.md) | Gitflow por fase del PRD, con rama `feature/fase-N-*` desde `main` |
| [0002](0002-entorno-notebooks-sin-jupyterlab.md) | Entorno de notebooks sin Jupyter Lab completo (límite de rutas largas de Windows) |
| [0003](0003-sin-holdout-en-fases-0-3.md) | Sin *holdout* de validación en las Fases 0–3; *golden set* se extrae más adelante del raw inmutable |
| [0004](0004-umbrales-referencia-sexo-especificos.md) | Umbrales de referencia sexo-específicos (Prati/ACG/AASLD) como criterio del proyecto |
| [0005](0005-prd-fuera-del-repositorio.md) | El PRD permanece fuera del repositorio, sin versionar |
| [0006](0006-deduplicar-antes-del-split.md) | Deduplicar antes del split, no después (fuga de datos crítica de la Actividad 2) |
| [0007](0007-tratamiento-de-nulos.md) | Tratamiento de nulos: reconstrucción determinista de `A/G Ratio` + imputación con indicador dentro del `Pipeline` |
| [0011](0011-indicador-nulos-columntransformer-dos-ramas.md) | Indicador de nulos vía `ColumnTransformer` de dos ramas (`MissingIndicator(features="all")`), no `SimpleImputer(add_indicator=True)` |
