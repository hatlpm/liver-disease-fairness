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
