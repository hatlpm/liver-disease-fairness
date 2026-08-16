"""Auditoría de equidad por sexo del mejor modelo (Fase 9, F9-R1..R6).

Responde la pregunta que la Actividad 1 dejó abierta (`informe_act1.md`
§10.1): una brecha de diagnóstico de 8.7 pp entre sexos, con p=0.055 (no
significativa). Esta fase mide si esa brecha, observada **antes** de
entrenar nada, llega al modelo ya entrenado (`logistic_regression`, F7-R4).

**El test NUNCA se usa aquí** (Trampa 1 del encargo): todas las funciones de
este módulo trabajan sobre TRAIN, agregando predicciones *out-of-fold* (OOF)
de una validación cruzada repetida (`RepeatedStratifiedKFold`, F9-R2) -- así
entran las 73 mujeres enfermas de train, no las ~20 de un split único.

Organización del módulo:

- **Blindaje del selector (Trampa 4).** :func:`build_shielded_selector`
  saca las columnas del indicador de nulos (`TB`/`DB`) de la competencia de
  `SelectKBest` -- nunca pueden quedar constantes-y-descartadas-en-silencio
  en un pliegue sin las 3 filas nulas. Es local a este módulo (vía el mismo
  parámetro `selector=` que ya acepta `build_pipeline`): `src/pipelines.py`
  no cambia una línea, así que las Fases 5-8 quedan intactas. Ver ADR-0016.
- **OOF de la CV repetida (Trampa 2, F9-R2).** `cross_val_predict` de
  sklearn no admite `RepeatedStratifiedKFold` (exige que `cv.split()` sea
  una partición; con `n_repeats=10` cada fila aparece 10 veces). Por eso
  :func:`oof_predictions_repeated_cv` itera las 10 repeticiones a mano y
  :func:`aggregate_oof_by_majority_vote` reduce las 10 predicciones por
  paciente a una sola por voto mayoritario.
- **Resolución declarada ANTES del resultado (Trampa 3).**
  :func:`declared_resolution` calcula, con la aritmética de una proporción
  binomial, qué tamaño de brecha es distinguible del ruido dado n=73
  mujeres enfermas -- se llama en el notebook antes de calcular la brecha
  real observada.
- **Métricas e incertidumbre (F9-R1, F9-R6).** :func:`metrics_by_group` /
  :func:`fnr_gap` calculan FNR/FPR/balanced_accuracy por sexo y la brecha.
  :func:`bootstrap_fnr_gap_ci` remuestrea PACIENTES (nunca repeticiones,
  Trampa 2) reutilizando :func:`src.evaluate.bootstrap_indices`.
  :func:`fnr_gap_significance` aplica el mismo test exacto de Fisher que
  Loop D ya usó para la brecha de diagnóstico de la Actividad 1.
- **Orquestación (F9-R3..R5).** :func:`run_fairness_variant` corre una
  configuración completa (una combinación de `include_gender`/`use_smote`)
  de principio a fin; :func:`compare_variants` corre las 4 combinaciones de
  F9-R4/F9-R5 en una sola tabla.
- **Controles de robustez.** El selector blindado, con el `k` congelado de
  la Fase 6/7, audita un modelo de 11 variables -- no las 10 que
  `GridSearchCV` eligió originalmente compitiendo con el indicador
  (documentado en :func:`_split_frozen_params`). `run_fairness_variant`
  acepta un ``selector_factory`` opcional para reproducir, sin blindar, el
  modelo real de 10 variables como control. :func:`fnr_gap_per_repeat`
  calcula la brecha con cada una de las 10 repeticiones por separado, sin
  agregar, para verificar que no depende de una partición concreta.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, norm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src.config import NUMERIC_COLS
from src.data import GENDER_MAP
from src.evaluate import bootstrap_indices
from src.models import build_estimator, fit_grid_search
from src.pipelines import build_pipeline

# Prefijo de las columnas del indicador de nulos TB/DB tal como las nombra
# el ColumnTransformer de build_pipeline (verificado empíricamente:
# "tb_db_indicator__missingindicator_TB"/"...DB"). No es una coincidencia
# de nombres frágil: el paso se llama "tb_db_indicator" en src/pipelines.py
# y ese nombre es el prefijo que ColumnTransformer antepone bajo
# set_output(transform="pandas").
_INDICATOR_COLUMN_PREFIX = "tb_db_indicator__"

GROUP_FEMALE = GENDER_MAP["Female"]
GROUP_MALE = GENDER_MAP["Male"]
_GROUP_LABELS = {GROUP_FEMALE: "mujeres", GROUP_MALE: "hombres"}

_ESTIMATOR_PARAM_PREFIX = "estimator__"
_SELECTOR_K_PARAM = "selector__k"


# ---------------------------------------------------------------------------
# Trampa 4 -- selector blindado (no toca src/pipelines.py)
# ---------------------------------------------------------------------------


def build_shielded_selector(k: int) -> ColumnTransformer:
    """`SelectKBest` que nunca compite con las columnas del indicador TB/DB (Trampa 4).

    `f_classif` calcula un F-estadístico basado en varianza entre grupos: si
    un pliegue de entrenamiento no contiene ninguna de las 3 filas con
    `TB`/`DB` nulos, la columna del indicador queda constante (varianza 0),
    el cálculo da `0/0` y `SelectKBest` la descarta con un score `NaN`, en
    silencio (riesgo heredado de la Fase 6, medido: 1 de los 50 pliegues de
    `RepeatedStratifiedKFold(5, n_repeats=10)` sobre train). Este selector
    elimina el problema de raíz: separa las columnas en dos ramas de un
    `ColumnTransformer` -- `SelectKBest(f_classif, k)` compite solo entre
    las columnas que NO son del indicador, y las 2 columnas del indicador
    pasan siempre, sin competir nunca por varianza.

    Se pasa como el argumento ``selector=`` de
    :func:`src.pipelines.build_pipeline`, el mismo mecanismo genérico que ya
    usan las Fases 6A-8 -- por eso este blindaje es exclusivo de la Fase 9 y
    no requiere ningún cambio en `src/pipelines.py` ni en las fases
    anteriores (ADR-0016).

    Parameters
    ----------
    k : int
        Número de columnas a elegir de la rama NO indicadora (mismo
        significado que el ``k`` de `SelectKBest`).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Con salida en formato pandas (mismo patrón que el ``"preprocess"``
        de `build_pipeline`), columnas: las `k` elegidas por `SelectKBest` +
        las 2 del indicador, siempre presentes.
    """
    selector = ColumnTransformer(
        [
            (
                "kbest",
                SelectKBest(f_classif, k=k),
                make_column_selector(pattern=f"^(?!{re.escape(_INDICATOR_COLUMN_PREFIX)})"),
            ),
            (
                "indicator_passthrough",
                "passthrough",
                make_column_selector(pattern=f"^{re.escape(_INDICATOR_COLUMN_PREFIX)}"),
            ),
        ]
    )
    selector.set_output(transform="pandas")
    return selector


def indicator_columns_present(transformed: pd.DataFrame) -> bool:
    """Verifica que las 2 columnas del indicador TB/DB sobrevivieron a la selección.

    Busca `_INDICATOR_COLUMN_PREFIX` como subcadena, no como prefijo
    estricto: el selector blindado antepone su propio nombre de paso
    (``"indicator_passthrough__"``) delante del nombre original de
    `ColumnTransformer` (p. ej.
    ``"indicator_passthrough__tb_db_indicator__missingindicator_TB"``), a
    diferencia de la salida plana de `build_pipeline` donde el prefijo del
    indicador queda al principio.

    Parameters
    ----------
    transformed : pandas.DataFrame
        Salida de un selector (blindado o no) ya ajustado y aplicado.

    Returns
    -------
    bool
        ``True`` si ambas columnas del indicador están presentes.
    """
    indicator_cols = [c for c in transformed.columns if _INDICATOR_COLUMN_PREFIX in c]
    return len(indicator_cols) == 2


# ---------------------------------------------------------------------------
# Trampa 2 -- OOF de la CV repetida (F9-R2)
# ---------------------------------------------------------------------------


def build_fairness_cv(params: dict) -> RepeatedStratifiedKFold:
    """Construye la CV repetida de equidad (F9-R2): `RepeatedStratifiedKFold(5, n_repeats=10)`.

    Lee ``params["cv"]["fairness"]`` y ``params["seed"]`` -- nunca literal.

    Parameters
    ----------
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    sklearn.model_selection.RepeatedStratifiedKFold
    """
    cv_cfg = params["cv"]["fairness"]
    return RepeatedStratifiedKFold(n_splits=cv_cfg["n_splits"], n_repeats=cv_cfg["n_repeats"], random_state=params["seed"])


def oof_predictions_repeated_cv(pipeline, X: pd.DataFrame, y: pd.Series, params: dict) -> np.ndarray:
    """Predicciones *out-of-fold* de las 10 repeticiones, una fila por repetición (F9-R2, Trampa 2).

    `sklearn.model_selection.cross_val_predict` no admite
    `RepeatedStratifiedKFold` directamente: exige que `cv.split()` sea una
    partición (cada fila predicha exactamente una vez), y con
    ``n_repeats=10`` cada fila del dataset aparece 10 veces. Esta función
    hace a mano lo que `cross_val_predict` haría para una partición simple,
    repetido 10 veces: `RepeatedStratifiedKFold.split()` entrega los 50
    splits en orden -- los primeros ``n_splits`` pertenecen a la repetición
    0 (una partición completa), los siguientes ``n_splits`` a la repetición
    1, etc. (comportamiento interno estable de `sklearn`) -- así que basta
    agrupar los splits de 5 en 5.

    `pipeline` se clona en cada pliegue (nunca se reutiliza el estado
    ajustado de un pliegue anterior, igual que hace `cross_val_predict`
    internamente) y **no se reajustan hiperparámetros aquí**: `pipeline` ya
    debe traer los `best_params_` congelados de la Fase 6/7 aplicados
    (mismo patrón que ADR-0015 en la Fase 8).

    Parameters
    ----------
    pipeline : sklearn estimator
        Pipeline sin ajustar (o ajustado -- se clona igual), con los
        hiperparámetros del ganador ya fijados vía `set_params`.
    X : pandas.DataFrame
        Variables de entrada de TRAIN (nunca test -- esta función no lo
        conoce ni lo recibe).
    y : pandas.Series
        Variable objetivo de TRAIN, mismo índice y mismo orden de filas
        que `X`.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    numpy.ndarray
        Forma ``(n_repeats, n_muestras)``. La fila `r`, columna `i` es la
        predicción de la fila `i`-ésima de `X` (orden posicional, no el
        índice de pandas) en la repetición `r`.
    """
    cv = build_fairness_cv(params)
    n_splits = params["cv"]["fairness"]["n_splits"]
    n = len(X)

    splits = list(cv.split(X, y))
    n_repeats = len(splits) // n_splits
    oof = np.full((n_repeats, n), fill_value=-1, dtype=int)

    for i, (train_idx, test_idx) in enumerate(splits):
        repeat = i // n_splits
        fold_pipe = clone(pipeline)
        fold_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[repeat, test_idx] = fold_pipe.predict(X.iloc[test_idx])

    return oof


def aggregate_oof_by_majority_vote(oof_matrix: np.ndarray, tie_break_label: int = 1) -> np.ndarray:
    """Reduce las 10 predicciones OOF por paciente a una sola, por voto mayoritario.

    Las repeticiones reducen la varianza DEL PROCEDIMIENTO de validación
    cruzada -- qué pliegue le tocó a cada paciente -- no la incertidumbre de
    que en train solo haya 73 mujeres enfermas (Trampa 2). El voto
    mayoritario aprovecha esa reducción de ruido para la estimación puntual
    sin fingir que hay `n_repeats * n_muestras` pacientes independientes: el
    denominador de cualquier métrica calculada después sigue siendo el
    número real de pacientes.

    Parameters
    ----------
    oof_matrix : numpy.ndarray
        Forma ``(n_repeats, n_muestras)``, salida de
        :func:`oof_predictions_repeated_cv`.
    tie_break_label : int, opcional
        Etiqueta a asignar en caso de empate exacto (posible con
        ``n_repeats`` par). Por defecto ``1`` (enfermo, F4-R5) -- que
        además es la clase mayoritaria de train (71.27%), así que el
        desempate coincide con la regla "empate -> clase mayoritaria".

    Returns
    -------
    numpy.ndarray
        Vector de largo ``n_muestras``, una etiqueta agregada por paciente.
    """
    n_repeats = oof_matrix.shape[0]
    votes_for_one = oof_matrix.sum(axis=0)
    aggregated = np.full(oof_matrix.shape[1], fill_value=tie_break_label, dtype=int)
    aggregated[votes_for_one > n_repeats / 2] = 1
    aggregated[votes_for_one < n_repeats / 2] = 0
    return aggregated


# ---------------------------------------------------------------------------
# Trampa 3 -- resolución declarada ANTES del resultado (F9-R6)
# ---------------------------------------------------------------------------


def declared_resolution(
    n_sick_group_a: int, n_sick_group_b: int, assumed_fnr: float, confidence_level: float
) -> dict:
    """Qué brecha de FNR es distinguible del ruido, calculado ANTES de mirar el resultado real (Trampa 3).

    Usa la aproximación normal del error estándar de una proporción binomial
    (``sqrt(p(1-p)/n)``) para cada grupo, y la combina para la brecha
    (varianzas independientes: ``sqrt(se_a² + se_b²)``). Con
    ``assumed_fnr≈0.45`` (coherente con el recall≈0.55 del ganador,
    Fase 8) y los tamaños reales de train (73 mujeres enfermas, 252
    hombres enfermos), el ancho del IC95% de la brecha es del orden de
    ±13 pp -- una brecha menor no será distinguible del ruido con este
    tamaño muestral, sin importar qué tan cuidadosa sea la validación
    cruzada. Debe llamarse ANTES de calcular la brecha observada real
    (verificado por
    `tests/test_fase9_fairness.py::test_resolucion_declarada`, que
    comprueba el orden en el notebook).

    Parameters
    ----------
    n_sick_group_a, n_sick_group_b : int
        Número de enfermos de cada grupo (p. ej. mujeres, hombres) --
        el denominador real de la FNR de cada uno.
    assumed_fnr : float
        FNR supuesta para el cálculo del error estándar, en (0, 1). No es
        la FNR observada -- es un supuesto declarado de antemano para fijar
        la resolución del experimento antes de ejecutarlo.
    confidence_level : float
        Nivel de confianza, p. ej. 0.95.

    Returns
    -------
    dict
        ``"se_group_a"``, ``"se_group_b"``, ``"se_gap"``,
        ``"ci_half_width_group_a_pp"``, ``"ci_half_width_group_b_pp"``,
        ``"ci_half_width_gap_pp"`` (en puntos porcentuales), ``"z_value"``,
        ``"assumed_fnr"``.
    """
    alpha = 1 - confidence_level
    z_value = norm.ppf(1 - alpha / 2)

    se_a = np.sqrt(assumed_fnr * (1 - assumed_fnr) / n_sick_group_a)
    se_b = np.sqrt(assumed_fnr * (1 - assumed_fnr) / n_sick_group_b)
    se_gap = np.sqrt(se_a**2 + se_b**2)

    return {
        "assumed_fnr": assumed_fnr,
        "z_value": float(z_value),
        "se_group_a": float(se_a),
        "se_group_b": float(se_b),
        "se_gap": float(se_gap),
        "ci_half_width_group_a_pp": float(z_value * se_a * 100),
        "ci_half_width_group_b_pp": float(z_value * se_b * 100),
        "ci_half_width_gap_pp": float(z_value * se_gap * 100),
    }


# ---------------------------------------------------------------------------
# F9-R1, F9-R3, F9-R6 -- métricas por sexo, brecha, incertidumbre
# ---------------------------------------------------------------------------


def metrics_by_group(y_true, y_pred, group, params: dict) -> pd.DataFrame:
    """FNR, FPR y balanced accuracy por grupo (F9-R1), con foco en la FNR.

    La FNR se calcula sobre los ENFERMOS de cada grupo (F4-R5:
    `pos_label=1`): de cada paciente enfermo, ¿el modelo lo mandó a casa
    como sano? La FPR, sobre los sanos: ¿el modelo le hizo una prueba de
    más a alguien sano?

    Parameters
    ----------
    y_true, y_pred : array-like
        Etiquetas verdaderas y predichas, en {0, 1}, alineadas
        posicionalmente con `group`.
    group : array-like
        Valor de grupo por fila (p. ej. `Gender`, en {0, 1}).
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    pandas.DataFrame
        Indexado por valor de grupo. Columnas: ``n``, ``n_enfermos``,
        ``n_sanos``, ``FN``, ``TP``, ``FP``, ``TN``, ``FNR``, ``FPR``,
        ``balanced_accuracy``.
    """
    pos_label = params["metrics"]["pos_label"]
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)

    rows = {}
    for g in np.unique(group):
        mask = group == g
        yt, yp = y_true[mask], y_pred[mask]
        sick = yt == pos_label
        healthy = ~sick

        tp = int(np.sum(sick & (yp == pos_label)))
        fn = int(np.sum(sick & (yp != pos_label)))
        fp = int(np.sum(healthy & (yp == pos_label)))
        tn = int(np.sum(healthy & (yp != pos_label)))

        rows[g] = {
            "n": int(mask.sum()),
            "n_enfermos": int(sick.sum()),
            "n_sanos": int(healthy.sum()),
            "FN": fn,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FNR": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
            "balanced_accuracy": balanced_accuracy_score(yt, yp),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def fnr_gap(metrics_df: pd.DataFrame, group_a: int = GROUP_FEMALE, group_b: int = GROUP_MALE) -> dict:
    """Brecha de FNR entre dos grupos, con signo (F9-R3: compara magnitud Y dirección).

    Por defecto, mujeres menos hombres (`GROUP_FEMALE`/`GROUP_MALE`, según
    :data:`src.data.GENDER_MAP`) -- signo negativo significa que a las
    mujeres se les escapan MÁS diagnósticos, el mismo sentido que el
    hallazgo de Straw & Wu (2022): −24.07 pp.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Salida de :func:`metrics_by_group`.
    group_a, group_b : int, opcional
        Valores de grupo a comparar (`group_a - group_b`). Por defecto,
        mujeres − hombres.

    Returns
    -------
    dict
        ``"fnr_group_a"``, ``"fnr_group_b"``, ``"gap"`` (fracción),
        ``"gap_pp"`` (puntos porcentuales).
    """
    fnr_a = metrics_df.loc[group_a, "FNR"]
    fnr_b = metrics_df.loc[group_b, "FNR"]
    gap = fnr_a - fnr_b
    return {"fnr_group_a": fnr_a, "fnr_group_b": fnr_b, "gap": gap, "gap_pp": gap * 100}


def bootstrap_fnr_gap_ci(
    y_true, y_pred, group, params: dict, n_iterations: int | None = None, seed: int | None = None
) -> dict:
    """IC bootstrap de la FNR por grupo y de su brecha, remuestreando PACIENTES (Trampa 2, F9-R6).

    Remuestrea con reemplazo los índices de paciente (nunca las
    repeticiones de la CV -- ver el módulo) reutilizando
    :func:`src.evaluate.bootstrap_indices`, el mismo mecanismo que la
    Fase 7 ya usa para el IC del test (F7-R8), aplicado aquí a las
    predicciones OOF agregadas de train. En cada remuestreo se recalculan
    FNR/FPR/balanced_accuracy por grupo y la brecha sobre las etiquetas YA
    predichas (ningún modelo se reajusta dentro del bootstrap: el bootstrap
    mide incertidumbre muestral de qué pacientes componen cada grupo, no
    incertidumbre de ajuste).

    Parameters
    ----------
    y_true, y_pred : array-like
        Etiquetas verdaderas y predicción OOF agregada (salida de
        :func:`aggregate_oof_by_majority_vote`).
    group : array-like
        Valor de grupo por fila.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    n_iterations : int o None, opcional
        Por defecto, ``params["evaluation"]["bootstrap"]["n_iterations"]``.
    seed : int o None, opcional
        Por defecto, ``params["seed"]``.

    Returns
    -------
    dict
        ``"by_group"``: `dict` de `dict` (una entrada por valor de grupo,
        con ``point_estimate``/``ci_lower``/``ci_upper`` de FNR). ``"gap"``:
        `dict` con las mismas claves para la brecha (`group_a` − `group_b`,
        mujeres − hombres por defecto). ``"n_iterations"``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)

    if n_iterations is None:
        n_iterations = params["evaluation"]["bootstrap"]["n_iterations"]
    if seed is None:
        seed = params["seed"]
    indices = bootstrap_indices(len(y_true), n_iterations, seed)

    point_metrics = metrics_by_group(y_true, y_pred, group, params)
    point_gap = fnr_gap(point_metrics)

    fnr_samples = {g: [] for g in point_metrics.index}
    gap_samples = []
    for idx in indices:
        resampled_metrics = metrics_by_group(y_true[idx], y_pred[idx], group[idx], params)
        for g in point_metrics.index:
            fnr_samples[g].append(resampled_metrics.loc[g, "FNR"] if g in resampled_metrics.index else np.nan)
        gap_samples.append(fnr_gap(resampled_metrics)["gap"] if set(point_metrics.index) <= set(resampled_metrics.index) else np.nan)

    confidence_level = params["evaluation"]["bootstrap"]["confidence_level"]
    alpha = 1 - confidence_level

    by_group = {}
    for g in point_metrics.index:
        samples = np.array(fnr_samples[g], dtype=float)
        ci_lower, ci_upper = np.nanquantile(samples, [alpha / 2, 1 - alpha / 2])
        by_group[int(g)] = {
            "label": _GROUP_LABELS.get(g, str(g)),
            "point_estimate": point_metrics.loc[g, "FNR"],
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    gap_arr = np.array(gap_samples, dtype=float)
    gap_ci_lower, gap_ci_upper = np.nanquantile(gap_arr, [alpha / 2, 1 - alpha / 2])
    return {
        "by_group": by_group,
        "gap": {
            "point_estimate": point_gap["gap"],
            "ci_lower": float(gap_ci_lower),
            "ci_upper": float(gap_ci_upper),
        },
        "n_iterations": len(indices),
    }


def fnr_gap_significance(y_true, y_pred, group, pos_label: int, group_a: int = GROUP_FEMALE, group_b: int = GROUP_MALE) -> dict:
    """Test exacto de Fisher sobre la tabla 2x2 (grupo x {FN, TP} entre enfermos), Trampa 5/F9-R6.

    Mismo test que Loop D ya aplicó a la brecha de diagnóstico de la
    Actividad 1 (`informe_act1.md` §10.1, p=0.055) -- usar el mismo
    estándar estadístico en ambas fases permite comparar directamente si
    la brecha de FNR pasa el mismo umbral que la brecha de diagnóstico.

    Parameters
    ----------
    y_true, y_pred : array-like
        Etiquetas verdaderas y predicción OOF agregada.
    group : array-like
        Valor de grupo por fila.
    pos_label : int
        Clase positiva (enfermo, F4-R5).
    group_a, group_b : int, opcional
        Grupos a comparar. Por defecto, mujeres/hombres.

    Returns
    -------
    dict
        ``"table"`` (`numpy.ndarray` 2x2: filas = [FN, TP], columnas =
        [group_a, group_b]), ``"p_value"``, ``"odds_ratio"``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group = np.asarray(group)
    sick = y_true == pos_label

    def _fn_tp(g):
        mask = sick & (group == g)
        fn = int(np.sum(y_pred[mask] != pos_label))
        tp = int(np.sum(y_pred[mask] == pos_label))
        return fn, tp

    fn_a, tp_a = _fn_tp(group_a)
    fn_b, tp_b = _fn_tp(group_b)
    table = np.array([[fn_a, fn_b], [tp_a, tp_b]])

    odds_ratio, p_value = fisher_exact(table)
    return {"table": table, "p_value": float(p_value), "odds_ratio": float(odds_ratio)}


def fnr_gap_per_repeat(oof_matrix: np.ndarray, y_true, group, params: dict) -> np.ndarray:
    """Brecha de FNR calculada por separado en CADA repetición (control de robustez).

    :func:`aggregate_oof_by_majority_vote` colapsa las 10 repeticiones en
    una sola estimación puntual -- útil para el resultado oficial, pero no
    dice si la brecha observada depende de una partición particular. Esta
    función NO agrega: para cada una de las `n_repeats` filas de
    `oof_matrix` (una partición completa de los 456 pacientes, ver
    :func:`oof_predictions_repeated_cv`), calcula la brecha de FNR con esa
    única repetición. Si la brecha aparece en todas las repeticiones y en
    un rango estrecho, no es un artefacto de qué pliegue le tocó a quién.

    Parameters
    ----------
    oof_matrix : numpy.ndarray
        Forma ``(n_repeats, n_muestras)``, salida de
        :func:`oof_predictions_repeated_cv` (antes de agregar).
    y_true : array-like
        Etiquetas verdaderas, alineadas posicionalmente con las columnas de
        `oof_matrix`.
    group : array-like
        Valor de grupo por fila, misma alineación.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    numpy.ndarray
        Un valor de ``gap_pp`` (puntos porcentuales) por repetición, largo
        ``n_repeats``.
    """
    y_true = np.asarray(y_true)
    group = np.asarray(group)
    gaps_pp = np.empty(oof_matrix.shape[0])
    for repeat in range(oof_matrix.shape[0]):
        metrics_df = metrics_by_group(y_true, oof_matrix[repeat], group, params)
        gaps_pp[repeat] = fnr_gap(metrics_df)["gap_pp"]
    return gaps_pp


# ---------------------------------------------------------------------------
# F9-R4, F9-R5 -- orquestación de las 4 variantes
# ---------------------------------------------------------------------------


def _split_frozen_params(best_params: dict) -> tuple[int, dict]:
    """Traduce `best_params_` de un `GridSearchCV` (selector plano) al selector blindado.

    `best_params_["selector__k"]` asume un `SelectKBest` plano bajo el paso
    ``"selector"``; con :func:`build_shielded_selector`, ese paso es un
    `ColumnTransformer` cuyo `SelectKBest` vive anidado bajo
    ``"selector__kbest__k"``. Esta función separa `k` (para reconstruir el
    selector) del resto de hiperparámetros del estimador, que sí se aplican
    tal cual con `set_params`.

    Parameters
    ----------
    best_params : dict
        `best_params_` de un `GridSearchCV` ajustado con
        ``selector=SelectKBest(f_classif)`` (no blindado).

    Returns
    -------
    tuple
        ``(k, estimator_params)`` -- `k` **sin acotar** (tal como lo eligió
        `GridSearchCV` compitiendo entre TODAS las columnas, indicador
        incluido), listo para :func:`build_shielded_selector` una vez
        acotado por el llamador (ver :func:`run_fairness_variant`), o para
        un `SelectKBest` plano sin acotar (el selector de control, sin
        blindaje). `estimator_params` va tal cual a
        `pipe.set_params(**estimator_params)`.

    Notes
    -----
    Bajo el selector blindado, la rama de `SelectKBest` compite solo entre
    las columnas NO indicadoras (9 en la variante sin `Gender`, 10 en la
    variante con `Gender`) -- un `k` congelado igual o mayor a esa cantidad
    (como el `k=10` de la variante oficial, sin `Gender`, competido
    originalmente entre 11 columnas) selecciona TODAS las columnas
    no-indicadoras: el modelo auditado bajo el selector blindado tiene 11
    variables (9 + las 2 del indicador, siempre presentes), no las 10 que
    `GridSearchCV` eligió originalmente entre 11 candidatas para el ganador
    de la Fase 7. Es una consecuencia real del blindaje (ADR-0016), no un
    error -- documentada explícitamente en `notebooks/09_fairness.ipynb`,
    con un selector de control SIN blindar que reproduce el modelo de 10
    variables de la Fase 7 para verificar que la conclusión no depende de
    esta diferencia.
    """
    k = best_params[_SELECTOR_K_PARAM]
    estimator_params = {key: value for key, value in best_params.items() if key.startswith(_ESTIMATOR_PARAM_PREFIX)}
    return k, estimator_params


def _default_shielded_selector(k: int, n_non_indicator_cols: int) -> ColumnTransformer:
    """Selector oficial de la Fase 9: blindado (Trampa 4), con `k` acotado a las columnas disponibles.

    El `k` congelado por `GridSearchCV` compitió entre TODAS las columnas
    (indicador incluido); bajo el blindaje, `SelectKBest` compite solo entre
    las `n_non_indicator_cols` no-indicadoras. Sin acotar, un `k` mayor
    (p. ej. 10 sobre 9 disponibles en la variante sin `Gender`) hace que
    `sklearn` seleccione todas de todos modos, pero emitiendo un
    `UserWarning` en cada uno de los 50 ajustes de la CV repetida -- 100
    avisos idénticos por variante que podrían tapar uno real en una fase
    futura. Acotar `k = min(k, n_non_indicator_cols)` produce exactamente el
    mismo resultado (seleccionar todas) sin el aviso.

    Parameters
    ----------
    k : int
        `k` congelado, sin acotar (salida de :func:`_split_frozen_params`).
    n_non_indicator_cols : int
        Columnas disponibles en la rama no-indicadora (9 sin `Gender`, 10
        con `Gender` -- `len(NUMERIC_COLS) + int(Gender presente)`).

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Salida de :func:`build_shielded_selector` con `k` acotado.
    """
    return build_shielded_selector(min(k, n_non_indicator_cols))


def run_fairness_variant(
    X_variant: pd.DataFrame,
    y_train: pd.Series,
    gender_train,
    params: dict,
    *,
    use_smote: bool,
    include_gender: bool,
    best_params: dict,
    selector_factory=None,
) -> dict:
    """Corre UNA configuración completa de la auditoría de equidad (F9-R1..R6).

    Construye el pipeline congelado del ganador (`logistic_regression`,
    `params["tuning_baseline"]["scaler"]`, selector blindado de la Trampa
    4 por defecto), obtiene las predicciones OOF de las 10 repeticiones
    (:func:`oof_predictions_repeated_cv`), las agrega por voto mayoritario,
    y calcula métricas por sexo, brecha, IC bootstrap y significancia.

    Parameters
    ----------
    X_variant : pandas.DataFrame
        Variables de entrada de TRAIN para esta variante (con o sin
        `Gender`, según `include_gender`).
    y_train : pandas.Series
        Variable objetivo de TRAIN, mismo orden de filas que `X_variant`.
    gender_train : array-like
        Valor de `Gender` de cada fila de TRAIN (0/1), mismo orden que
        `X_variant` -- se usa para desagregar métricas por sexo incluso en
        la variante SIN `Gender` como variable de entrada (F6A-R4: quitar
        `Gender` del modelo no significa dejar de auditar por sexo).
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    use_smote : bool
        Si esta variante balancea con SMOTE (F9-R4).
    include_gender : bool
        Si `Gender` es columna de entrada en esta variante (F9-R5) --
        metadato de la fila resultante, no cambia el cálculo (ya está
        reflejado en `X_variant`).
    best_params : dict
        `best_params_` de `GridSearchCV` para `logistic_regression` bajo
        esta variante de `Gender` (de
        ``fit_grid_search("logistic_regression", X_variant, y_train, params,
        selector=SelectKBest(f_classif))``), congelados -- nunca se
        reajustan dentro de esta función (mismo patrón que ADR-0015).
    selector_factory : callable o None, opcional
        ``k -> transformer sklearn`` para construir el paso `"selector"`.
        Por defecto (``None``), el selector blindado oficial con `k`
        acotado (:func:`_default_shielded_selector`) -- el modelo de 11
        variables auditado en esta fase (F9-R1..R6). Para el control de
        robustez del notebook (¿la conclusión depende de blindar?), pasar
        ``lambda k: SelectKBest(f_classif, k=k)`` reproduce, sin acotar, el
        selector plano de las Fases 6A-8 -- el modelo de 10 variables
        declarado ganador en la Fase 7 (con el riesgo medido del indicador
        constante, no blindado).

    Returns
    -------
    dict
        ``"use_smote"``, ``"include_gender"``, ``"metrics_by_group"``
        (`pandas.DataFrame`), ``"gap"``, ``"bootstrap"``, ``"significance"``,
        ``"oof_matrix"`` (forma ``(n_repeats, n_muestras)``, para controles
        de robustez como :func:`fnr_gap_per_repeat`).
    """
    k, estimator_params = _split_frozen_params(best_params)
    if selector_factory is None:
        n_non_indicator_cols = len(NUMERIC_COLS) + int(include_gender)
        selector = _default_shielded_selector(k, n_non_indicator_cols)
    else:
        selector = selector_factory(k)

    pipe = build_pipeline(
        scaler=params["tuning_baseline"]["scaler"],
        use_smote=use_smote,
        estimator=build_estimator("logistic_regression", params),
        params=params,
        selector=selector,
    )
    pipe.set_params(**estimator_params)

    oof_matrix = oof_predictions_repeated_cv(pipe, X_variant, y_train, params)
    y_pred_agg = aggregate_oof_by_majority_vote(oof_matrix)
    y_true_arr = y_train.to_numpy()
    group_arr = np.asarray(gender_train)

    metrics_df = metrics_by_group(y_true_arr, y_pred_agg, group_arr, params)
    gap = fnr_gap(metrics_df)
    bootstrap = bootstrap_fnr_gap_ci(y_true_arr, y_pred_agg, group_arr, params)
    significance = fnr_gap_significance(y_true_arr, y_pred_agg, group_arr, params["metrics"]["pos_label"])

    return {
        "use_smote": use_smote,
        "include_gender": include_gender,
        "metrics_by_group": metrics_df,
        "gap": gap,
        "bootstrap": bootstrap,
        "significance": significance,
        "oof_matrix": oof_matrix,
    }


def compare_variants(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> pd.DataFrame:
    """Las 4 combinaciones {con/sin `Gender`} x {con/sin SMOTE} en una tabla (F9-R4, F9-R5).

    Los hiperparámetros de `logistic_regression` se ajustan UNA vez por
    variante de `Gender` (2 `GridSearchCV`, bajo `params["tuning_baseline"]`
    -- Z-Score + SMOTE, la configuración con la que se declaró ganador en
    la Fase 6/7) y se congelan para las dos variantes de SMOTE de esa misma
    columna de `Gender` (mismo patrón que ADR-0015 en la Fase 8): `Gender`
    cambia el conjunto de variables de entrada (hay que reajustar), SMOTE
    es un factor de preprocesamiento que se aísla sin reajustar.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Variables de entrada de TRAIN, CON `Gender` (variante completa;
        esta función deriva la variante sin `Gender` internamente).
    y_train : pandas.Series
        Variable objetivo de TRAIN.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    pandas.DataFrame
        4 filas: ``include_gender``, ``use_smote``, ``fnr_mujeres``,
        ``fnr_hombres``, ``gap_pp``, ``gap_ci_lower_pp``,
        ``gap_ci_upper_pp``, ``p_value``.
    """
    gender_train = X_train["Gender"]
    X_by_variant = {True: X_train, False: X_train.drop(columns=["Gender"])}

    best_params_by_gender = {
        include_gender: fit_grid_search(
            "logistic_regression", X_by_variant[include_gender], y_train, params, selector=SelectKBest(f_classif)
        ).best_params_
        for include_gender in (True, False)
    }

    rows = []
    for include_gender in (True, False):
        for use_smote in (True, False):
            result = run_fairness_variant(
                X_by_variant[include_gender],
                y_train,
                gender_train,
                params,
                use_smote=use_smote,
                include_gender=include_gender,
                best_params=best_params_by_gender[include_gender],
            )
            rows.append(
                {
                    "include_gender": include_gender,
                    "use_smote": use_smote,
                    "fnr_mujeres": result["metrics_by_group"].loc[GROUP_FEMALE, "FNR"],
                    "fnr_hombres": result["metrics_by_group"].loc[GROUP_MALE, "FNR"],
                    "gap_pp": result["gap"]["gap_pp"],
                    "gap_ci_lower_pp": result["bootstrap"]["gap"]["ci_lower"] * 100,
                    "gap_ci_upper_pp": result["bootstrap"]["gap"]["ci_upper"] * 100,
                    "p_value": result["significance"]["p_value"],
                }
            )
    return pd.DataFrame(rows)
