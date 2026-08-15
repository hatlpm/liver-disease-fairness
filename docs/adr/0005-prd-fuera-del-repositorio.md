# 5. El PRD permanece fuera del repositorio, sin versionar

**Fecha:** 2026-08-15
**Estado:** Aceptada

## Contexto

`PRD_liver_disease_fases_0-3.md` es la **fuente de verdad declarada** del
proyecto: `AGENTS.md` instruye explícitamente a leerlo antes de tocar código,
y no se resume en ningún otro sitio para evitar que dos documentos diverjan.
Los requisitos numerados que el trabajo dice cumplir (`F3-R12`, `§2.3`,
`§7.3`, la rúbrica de las 8 tareas) solo existen ahí.

Sin embargo vive **un nivel por encima de la raíz del repo**, en la carpeta
`MIU data science/`, junto a los enunciados originales de las actividades del
máster (`act1/`, `act2/`). No está en git.

Una revisión del repositorio (2026-08-15) señaló la asimetría: el proyecto
aplica un estándar de trazabilidad estricto a todo lo demás — ningún número
sin celda ejecutada, ninguna afirmación clínica sin fuente versionada en
`docs/fuentes/`, cuatro loops CRISP-DM registrados — mientras que el
documento que define qué hay que hacer no tiene historial, ni diffs, ni
posibilidad de revertir.

Se evaluaron tres caminos:

1. **Mover el PRD dentro del repo.** Resuelve la trazabilidad, pero mezcla
   material del enunciado del máster con el repositorio público de
   portafolio, y separa el PRD de los enunciados originales (`act1.docx`,
   `act2.docx`) con los que se lee en conjunto.
2. **Copiarlo dentro del repo.** Crea exactamente la divergencia entre dos
   copias que `AGENTS.md` decía querer evitar.
3. **Dejarlo fuera y documentar la consecuencia.**

## Decisión

Se mantiene **fuera del repositorio y sin versionar**, por ahora. El PRD
pertenece al material de curso del máster, no al artefacto de portafolio, y
se lee junto a los enunciados de las actividades que viven a su lado.

A cambio, la limitación queda **declarada de forma visible** en `AGENTS.md`,
en vez de quedar implícita en una ruta relativa `../`.

## Consecuencias

- **Quien clone desde GitHub no recibe la fuente de verdad.** El repo debe
  poder leerse sin ella: `README.md`, `AGENTS.md`, los ADR y el
  `CHANGELOG` cargan esa responsabilidad.
- **La única copia depende de la sincronización de OneDrive**, sin historial,
  sin diffs y sin posibilidad de revertir un cambio accidental. Si el PRD se
  corrompe o se borra, no hay recuperación desde el proyecto.
- **Un cambio en el PRD no deja rastro.** Si los requisitos se modifican a
  mitad de una fase, nada lo registra y el `CHANGELOG` no puede referenciar
  "antes/después".
- Un agente que abra el repo sin acceso al directorio padre **debe decirlo
  explícitamente** en vez de inferir los requisitos leyendo el código — lo
  cual invertiría la relación entre especificación e implementación.
- **Revisar al abrir las Fases 4–7.** Esas fases requieren un PRD nuevo; ese
  es el momento natural para reconsiderar esta decisión, y bastaría un ADR
  que marque este como superado.
