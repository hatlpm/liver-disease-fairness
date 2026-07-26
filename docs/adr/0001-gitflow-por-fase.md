# 1. Gitflow por fase del PRD, con rama `feature/fase-N-*` desde `main`

**Fecha:** 2026-07-26
**Estado:** Aceptada

## Contexto

El desarrollo cubre cuatro fases del PRD (0–3) más el ensamblado de
entregables (Fase E), y el usuario pidió explícitamente poder revisar e
interpretar/corregir el trabajo de cada fase antes de que se dé por cerrada.
Trabajar todo directamente sobre `main` no deja un punto de retorno claro si
una fase necesita reabrirse (por ejemplo, un Loop A/B de CRISP-DM que obliga
a revisar una fase anterior).

## Decisión

Cada fase del PRD se desarrolla en su propia rama `feature/fase-N-nombre`,
creada desde `main`. El agente no commitea ni mezcla ramas sin la aprobación
explícita del usuario en cada checkpoint. Al cerrar una fase, se mezcla a
`main` (mezcla local, nunca se publica automáticamente a ningún remoto).

## Consecuencias

- Cada fase queda aislada en su propia rama: si algo hay que corregir, se
  corrige antes del merge sin tocar el trabajo ya cerrado de fases previas.
- El historial de `main` queda como una secuencia legible de fases cerradas,
  útil para el portafolio.
- Costo: para un usuario nuevo en git, cada fase implica un paso extra
  (crear rama, luego mezclar) en vez de commitear directo — se documenta en
  `AGENTS.md` y se explican los comandos en cada checkpoint.
