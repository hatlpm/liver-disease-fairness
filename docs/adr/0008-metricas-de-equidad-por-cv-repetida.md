# 8. Las métricas de equidad se calculan por CV repetida sobre train, no sobre el split único

**Fecha:** 2026-08-16
**Estado:** Aceptada

## Contexto

La Fase 9 audita si el sesgo por sexo que la Actividad 1 midió *antes* de
entrenar llega al modelo. La métrica central es la **tasa de falsos negativos
(FNR)** por sexo: de cada paciente enferma, ¿el modelo la mandó a casa como
sana?

El camino evidente —evaluar sobre el conjunto de prueba congelado en la Fase 4,
igual que hizo la Fase 7— **no funciona aquí**, y la razón es puramente de
tamaño muestral. La FNR se calcula sobre los *enfermos* de cada grupo, y ese
denominador es minúsculo en el test:

| Origen | Mujeres enfermas | Hombres enfermos | IC95 de la brecha |
|---|---|---|---|
| Test (split único, 114 filas) | **18** | 63 | **±26.1 pp** |
| Train (456 filas) | **73** | 252 | **±13.0 pp** |

Con 18 mujeres enfermas, cualquier brecha menor de ~26 pp queda dentro del
ruido. **La brecha realmente observada fue de +19.08 pp**: sobre el split único
no habría sido distinguible del azar, y la fase habría concluido "no hay
evidencia" por falta de resolución, no por ausencia de sesgo.

Es el mismo cálculo que el `ADR-0003` hizo en la Actividad 1 para rechazar un
*holdout* de 29 filas: con n=7 mujeres, una sola predicción distinta movía la
métrica 14–37 pp. El problema no cambió; cambió la fase en la que aparece.

Usar el dataset completo (570 filas) para ganar mujeres tampoco es opción:
metería el test en el entrenamiento de algunos pliegues y emborronaría la
estimación insesgada que la Fase 7 obtuvo gastando el test una sola vez.

## Decisión

Las métricas de equidad se obtienen por **`RepeatedStratifiedKFold(n_splits=5,
n_repeats=10)` sobre TRAIN**, agregando predicciones *out-of-fold*. Así entran
las **73 mujeres enfermas** de train (4.1× el denominador del test) en vez de 18.

Tres precisiones que forman parte de la decisión:

1. **Agregación por voto mayoritario.** Cada paciente recibe 10 predicciones OOF
   (una por repetición); se reducen a una por voto mayoritario, con desempate
   hacia `enfermo` (clase mayoritaria). El denominador de cualquier métrica
   posterior sigue siendo el número real de pacientes — nunca
   `n_repeats × n_pacientes`, que fingiría 10× más gente de la que hay.
2. **El intervalo de confianza se obtiene remuestreando PACIENTES**, jamás
   repeticiones. Las repeticiones reducen la varianza *del procedimiento* de
   validación cruzada —qué pliegue le tocó a quién—, no la incertidumbre de que
   solo haya 73 mujeres enfermas. Un IC calculado sobre la dispersión entre
   repeticiones saldría absurdamente estrecho y sería falso.
3. **La resolución se declara ANTES de calcular la brecha.** `declared_resolution()`
   se llama en el notebook antes de la celda que mide, y un test verifica ese
   orden. Declarar la resolución después de ver el resultado es racionalizarlo.

## Consecuencias

- **La auditoría se hace sobre train, no sobre datos nunca vistos.** Es el
  precio de no gastar el test dos veces, y hay que declararlo como limitación
  en el informe: la brecha está medida en validación cruzada, no en un conjunto
  independiente.
- El modelo auditado se ajusta 50 veces (5 pliegues × 10 repeticiones) por cada
  variante; con 4 variantes (`Gender` × SMOTE) son 200 ajustes. Es asumible con
  456 filas, pero deja de serlo si el dataset crece.
- **La conclusión de la fase depende de esta decisión.** Con el split único, los
  +19.08 pp (p = 0.0043) habrían quedado dentro de un IC de ±26 pp. Quien lea el
  informe debe poder ver que la elección se tomó por resolución estadística
  declarada de antemano, no porque diera el resultado deseado — de ahí el
  requisito 3.
- Se mantiene el estándar del Loop D: prueba de significancia e intervalo de
  confianza antes de llamar hallazgo a nada. Si la brecha no hubiera alcanzado,
  la conclusión honesta era "no hay evidencia suficiente con n=73".
