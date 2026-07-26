# CHANGELOG de iteraciones CRISP-DM

Registro de los *loops* de retroalimentación entre fases (§9.1 del PRD): cada
vez que un hallazgo de datos obliga a revisar una fase anterior, se anota
aquí con fecha, disparador y decisión. Esto es distinto de `docs/adr/`
(decisiones de ingeniería/tooling) y de `reports/informe_act1.md`
(resultados académicos).

## 2026-07-26 — Loop A: EDA → Business Understanding

**Disparador:** en `02_eda.ipynb` (Fase 2), el análisis estratificado por
`Gender` mostró dos hallazgos no anticipados en el detalle de la Fase 1:

1. La tasa de diagnóstico positivo (`Selector == 1`) es **73.47% en hombres
   vs. 64.79% en mujeres** — una brecha de ~9 puntos porcentuales que no se
   puede atribuir, solo con EDA, a una población femenina más sana o a un
   patrón de sub-diagnóstico.
2. `Alkphos` (ALP) es la **única** enzima con media más alta en mujeres
   (302.3) que en hombres (286.8), rompiendo el patrón de todas las demás
   enzimas/bilirrubinas (más altas en hombres). Coincide con el hallazgo de
   Straw & Wu (2022) de que ALP y sexo ganan importancia al corregir la
   subrepresentación femenina.

**Decisión:** no se reformulan las preguntas de investigación de
`01_problem_framing.ipynb` (siguen siendo válidas tal como están), pero se
deja este hallazgo registrado como evidencia concreta que **refuerza** la
pregunta de investigación 3 (¿el desbalance de sexo puede inducir sesgo?) y
sienta una hipótesis específica y verificable para la futura Fase 5
(auditoría de *fairness*, fuera de este PRD): *"la brecha en tasa de
diagnóstico por sexo observada aquí es coherente con un patrón de
sub-diagnóstico en mujeres, no solo con una diferencia real de severidad."*
No se puede confirmar ni descartar con EDA — requiere el modelo y las
métricas de la Actividad 2.

**Impacto:** ninguno sobre el alcance de las Fases 0–3; el hallazgo queda
documentado para informar el diseño de la Fase 5 cuando se especifique.
