"""Experimento factorial {MinMax, Z-Score} x {sin SMOTE, SMOTE} x 5 modelos (Fase 8, F8-R1..R7).

Responde T8: qué impacto tuvo la normalización y el balanceo en las métricas
de rendimiento. Las 20 celdas se comparan por validación cruzada estratificada
sobre TRAIN (F8-R2, Trampa 1 del encargo) -- **ninguna función de este módulo
recibe ni referencia el conjunto de prueba**, ni siquiera como parámetro
opcional, para que la restricción sea verificable por lectura del código y no
solo por convención (`tests/test_fase8_factorial.py::test_factorial_evaluado_en_cv_no_en_test`).

Los hiperparámetros (incluido `selector__k`) se heredan de la Fase 6
(`best_params_` de `GridSearchCV` bajo `params["tuning_baseline"]`) y se
aplican con `pipe.set_params(...)` a cada una de las 20 combinaciones -- este
módulo nunca vuelve a llamar `GridSearchCV` (Trampa 2: reajustar por celda
confundiría el efecto del preprocesamiento con el efecto de volver a ajustar,
que es justo lo que T8 pregunta por separado).

Reutiliza `build_pipeline` (`src/pipelines.py`), `build_cv`/`build_estimator`
(`src/models.py`) y `fit_all_models`/`compute_metrics`/`confusion_matrix_narrative`
(`src/evaluate.py`) -- ninguna construcción de pipeline ni cálculo de métricas
se reimplementa aquí.
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.neighbors import NearestNeighbors

from src.config import SKEWED_COLS, SKLEARN_ZERO_DIVISION
from src.evaluate import confusion_matrix_narrative, fit_all_models
from src.models import MODEL_KEYS, build_cv, build_estimator
from src.pipelines import build_pipeline

_METRIC_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]


def _cv_scoring(params: dict) -> dict:
    """Diccionario de `scoring=` para `cross_validate`, con `average`/`pos_label` de `params`.

    `accuracy` y `balanced_accuracy` no distinguen clase positiva (nombres de
    `sklearn` bastan); `precision`/`recall`/`f1` se atan explícitamente a
    ``params["metrics"]["average"]``/``["pos_label"]`` (F7-R2, mismo criterio
    que :func:`src.evaluate.compute_metrics`) en vez de confiar en que el
    default de `sklearn` coincida por casualidad.

    Parameters
    ----------
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    dict
        Listo para pasar como ``scoring=`` a `cross_validate`.
    """
    average = params["metrics"]["average"]
    pos_label = params["metrics"]["pos_label"]
    return {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": make_scorer(
            precision_score, average=average, pos_label=pos_label, zero_division=SKLEARN_ZERO_DIVISION
        ),
        "recall": make_scorer(
            recall_score, average=average, pos_label=pos_label, zero_division=SKLEARN_ZERO_DIVISION
        ),
        "f1": make_scorer(f1_score, average=average, pos_label=pos_label, zero_division=SKLEARN_ZERO_DIVISION),
    }


def _validate_balancing_method(balancing_method: str, params: dict) -> bool:
    """Traduce ``balancing_method`` (valor de ``params["balancing"]["methods"]``) a `use_smote`.

    Parameters
    ----------
    balancing_method : str
        Debe estar en ``params["balancing"]["methods"]`` (hoy: ``"none"`` o ``"smote"``).
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    bool
        ``True`` si ``balancing_method == "smote"``.

    Raises
    ------
    ValueError
        Si `balancing_method` no está declarado en `params.yaml`.
    """
    valid_methods = params["balancing"]["methods"]
    if balancing_method not in valid_methods:
        raise ValueError(f"balancing_method={balancing_method!r} no está en params.balancing.methods={valid_methods}")
    return balancing_method == "smote"


# ---------------------------------------------------------------------------
# (a) Hiperparámetros congelados de la Fase 6 -- se recalculan una vez, nunca
#     dentro de una celda del factorial (Trampa 2, ADR-0015)
# ---------------------------------------------------------------------------


def collect_frozen_best_params(X_train, y_train, params: dict, *, selector=None) -> dict:
    """`best_params_` de los 5 modelos bajo `params["tuning_baseline"]`, recalculados una vez.

    Reutiliza :func:`src.evaluate.fit_all_models`, que ya ajusta bajo la
    configuración fija de la Fase 6 (Z-Score + SMOTE) -- el mismo patrón que
    `07_evaluacion.ipynb` usó para ser autocontenido, en vez de perseguir el
    estado de un notebook anterior. Ningún hiperparámetro se copia a mano
    (ADR-0015): estos `best_params_` son los que las 20 celdas del factorial
    aplican vía `pipe.set_params(...)`, sin volver a llamar `GridSearchCV`.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Variables de entrada de TRAIN (variante oficial: sin `Gender`).
    y_train : pandas.Series
        Variable objetivo de TRAIN.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    selector : sklearn transformer o None, opcional
        Selector de variables sin ajustar (p. ej. ``SelectKBest(f_classif)``).

    Returns
    -------
    dict
        ``{nombre_modelo: best_params_}``, una entrada por :data:`src.models.MODEL_KEYS`.
    """
    searches = fit_all_models(X_train, y_train, params, selector=selector)
    return {name: search.best_params_ for name, search in searches.items()}


# ---------------------------------------------------------------------------
# (b)-(c) Las 20 celdas -- F8-R1/F8-R2
# ---------------------------------------------------------------------------


def evaluate_cell(
    name: str,
    scaler: str,
    balancing_method: str,
    best_params: dict,
    X_train,
    y_train,
    params: dict,
    *,
    selector=None,
) -> dict:
    """UNA celda del factorial: pipeline congelado + `cross_validate` sobre TRAIN.

    Construye ``build_pipeline(scaler, use_smote, build_estimator(name, params),
    params, selector)`` y le aplica `best_params` con `pipe.set_params(...)` --
    **sin `GridSearchCV` en esta función**: `best_params` es un argumento que
    ya trae los hiperparámetros congelados de la Fase 6
    (:func:`collect_frozen_best_params`), nunca se recalcula aquí (Trampa 2).
    Evalúa con `cross_validate(cv=build_cv(params), ...)` -- CV estratificada
    sobre `X_train`/`y_train`, jamás el conjunto de prueba (F8-R2).

    Parameters
    ----------
    name : str
        Uno de :data:`src.models.MODEL_KEYS`.
    scaler : str
        Uno de ``params["preprocessing"]["scalers"]``.
    balancing_method : str
        Uno de ``params["balancing"]["methods"]`` (``"none"`` o ``"smote"``).
    best_params : dict
        `best_params_` de `GridSearchCV` para `name` (con prefijos
        ``estimator__``/``selector__``), tal como los devuelve
        :func:`collect_frozen_best_params`.
    X_train, y_train : pandas.DataFrame / pandas.Series
        Datos de TRAIN. Nunca el conjunto de prueba -- esta función no lo
        conoce ni lo recibe.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    selector : sklearn transformer o None, opcional
        Selector de variables sin ajustar, insertado en el `Pipeline` antes
        de aplicar `best_params` (para que ``"selector__k"`` tenga a qué
        aplicarse).

    Returns
    -------
    dict
        ``{"modelo", "escalador", "balanceo"}`` + una clave ``f"{métrica}_cv"``
        por cada una de :data:`_METRIC_KEYS`, con la media sobre los pliegues.
    """
    use_smote = _validate_balancing_method(balancing_method, params)
    pipe = build_pipeline(scaler, use_smote, build_estimator(name, params), params, selector=selector)
    pipe.set_params(**best_params)

    scores = cross_validate(pipe, X_train, y_train, cv=build_cv(params), scoring=_cv_scoring(params), n_jobs=-1)
    row = {"modelo": name, "escalador": scaler, "balanceo": balancing_method}
    row.update({f"{metric}_cv": scores[f"test_{metric}"].mean() for metric in _METRIC_KEYS})
    return row


def run_factorial(X_train, y_train, params: dict, best_params_by_model: dict, *, selector=None) -> pd.DataFrame:
    """Las 20 configuraciones {escaladores} x {balanceo} x 5 modelos (F8-R1).

    Recorre ``params["preprocessing"]["scalers"]`` x ``params["balancing"]["methods"]``
    x :data:`src.models.MODEL_KEYS` -- nunca un literal ``20``: el número de
    filas de la tabla resultante es
    ``len(scalers) * len(methods) * len(MODEL_KEYS)``, lo que sea que
    `params.yaml` declare.

    Parameters
    ----------
    X_train, y_train : pandas.DataFrame / pandas.Series
        Datos de TRAIN (variante oficial: sin `Gender`).
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    best_params_by_model : dict
        Salida de :func:`collect_frozen_best_params`.
    selector : sklearn transformer o None, opcional
        Selector de variables sin ajustar, igual en las 20 celdas.

    Returns
    -------
    pandas.DataFrame
        Una fila por celda, columnas ``modelo``, ``escalador``, ``balanceo`` +
        una por métrica de :data:`_METRIC_KEYS` (sufijo ``_cv``).
    """
    rows = [
        evaluate_cell(name, scaler, balancing_method, best_params_by_model[name], X_train, y_train, params, selector=selector)
        for scaler in params["preprocessing"]["scalers"]
        for balancing_method in params["balancing"]["methods"]
        for name in MODEL_KEYS
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (d) Trampa 4 -- celdas degeneradas (predicción de una sola clase)
# ---------------------------------------------------------------------------


def detect_degenerate_cells(
    X_train, y_train, params: dict, best_params_by_model: dict, *, selector=None
) -> pd.DataFrame:
    """Marca las celdas donde el modelo predice UNA SOLA CLASE para todo TRAIN (Trampa 4).

    `balanced_accuracy = 0.5000` exacto es compatible con "aprendió algo raro
    que promedia a 0.5" y con "no predijo más que una clase" -- esta función
    no infiere, **verifica**: usa `cross_val_predict(method="predict")` (no
    `predict_proba` + umbral -- la Fase 8 no toca el umbral de decisión, eso
    es exclusivo de la Fase 7/ADR-0014) para obtener las predicciones reales
    de cada celda y cuenta cuántas clases distintas aparecen. La narrativa en
    pacientes reutiliza :func:`src.evaluate.confusion_matrix_narrative`.

    Parameters
    ----------
    X_train, y_train : pandas.DataFrame / pandas.Series
        Datos de TRAIN.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    best_params_by_model : dict
        Salida de :func:`collect_frozen_best_params`.
    selector : sklearn transformer o None, opcional
        Igual que en :func:`run_factorial`.

    Returns
    -------
    pandas.DataFrame
        Una fila por celda: ``modelo``, ``escalador``, ``balanceo``,
        ``n_clases_predichas``, ``clase_unica_predicha`` (o ``None`` si no
        degeneró) y las 4 claves de
        :func:`src.evaluate.confusion_matrix_narrative`.
    """
    rows = []
    for scaler in params["preprocessing"]["scalers"]:
        for balancing_method in params["balancing"]["methods"]:
            use_smote = _validate_balancing_method(balancing_method, params)
            for name in MODEL_KEYS:
                pipe = build_pipeline(scaler, use_smote, build_estimator(name, params), params, selector=selector)
                pipe.set_params(**best_params_by_model[name])
                y_pred = cross_val_predict(pipe, X_train, y_train, cv=build_cv(params), method="predict")

                clases_predichas = np.unique(y_pred)
                narrativa = confusion_matrix_narrative(y_train, y_pred, pos_label=params["metrics"]["pos_label"])
                rows.append(
                    {
                        "modelo": name,
                        "escalador": scaler,
                        "balanceo": balancing_method,
                        "n_clases_predichas": len(clases_predichas),
                        "clase_unica_predicha": int(clases_predichas[0]) if len(clases_predichas) == 1 else None,
                        **narrativa,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (e) Trampa 3 -- por qué SMOTE rompe la invariancia al escalado que el
#     estimador sí tiene (verificación empírica del mecanismo, no de manual)
# ---------------------------------------------------------------------------


def smote_neighbor_agreement_across_scalers(X_train, y_train, params: dict) -> pd.DataFrame:
    """Prueba geométrica de Trampa 3: ¿SMOTE ve los mismos vecinos bajo MinMax y Z-Score?

    Ajusta el paso ``"preprocess"`` (sin selector, sin SMOTE) bajo cada
    escalador, restringe a la clase minoritaria (la que SMOTE remuestrea) y,
    con `NearestNeighbors(k_neighbors)` (el mismo `k` que usa `SMOTE` en
    `params["balancing"]["smote"]["k_neighbors"]`), compara el conjunto de
    vecinos de cada paciente minoritario entre ambos escalados con el índice
    de Jaccard. Si MinMax y Z-Score fueran solo un cambio de unidades sin
    consecuencia geométrica, el conjunto de vecinos sería idéntico (Jaccard=1)
    para todos; si la escala relativa entre variables importa (como predice
    la distancia euclídea), no lo será.

    Parameters
    ----------
    X_train, y_train : pandas.DataFrame / pandas.Series
        Datos de TRAIN.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    pandas.DataFrame
        Una fila por paciente de la clase minoritaria: ``jaccard`` (0-1) y los
        conjuntos de vecinos bajo cada escalador.
    """
    pos_label = params["metrics"]["pos_label"]
    k = params["balancing"]["smote"]["k_neighbors"]
    minority_mask = y_train == pos_label

    neighbor_sets = {}
    for scaler in params["preprocessing"]["scalers"]:
        pipe = build_pipeline(scaler, False, DummyClassifier(), params, selector=None)
        transformed = pipe.named_steps["preprocess"].fit_transform(X_train)
        minority_transformed = transformed.loc[minority_mask.to_numpy()]

        nn = NearestNeighbors(n_neighbors=k + 1).fit(minority_transformed)
        _, indices = nn.kneighbors(minority_transformed)
        neighbor_sets[scaler] = [frozenset(row[1:]) for row in indices]  # excluye el propio punto (vecino 0)

    scalers = params["preprocessing"]["scalers"]
    rows = []
    for i in range(len(neighbor_sets[scalers[0]])):
        set_a, set_b = neighbor_sets[scalers[0]][i], neighbor_sets[scalers[1]][i]
        jaccard = len(set_a & set_b) / len(set_a | set_b)
        rows.append(
            {
                "paciente_minoritario_idx": i,
                f"vecinos_{scalers[0]}": set_a,
                f"vecinos_{scalers[1]}": set_b,
                "jaccard": jaccard,
            }
        )
    return pd.DataFrame(rows)


def smote_synthetic_samples_original_units(X_train, y_train, params: dict, scaler: str) -> pd.DataFrame:
    """Sintéticos de SMOTE bajo `scaler`, revertidos a las unidades originales de cada variable.

    Construye ``build_pipeline(scaler, True, <placeholder>, params, selector=None)``
    (sin selector, para conservar los nombres de columna de pandas) y corta el
    último paso (`pipe[:-1]`, deja ``["preprocess", "smote"]``) para llamar
    `.fit_resample(X_train, y_train)` sin ajustar ningún estimador real. Las
    filas sintéticas son las que aparecen después de las 456 reales
    (verificado empíricamente: `imbalanced-learn` conserva el orden de las
    filas reales y añade las sintéticas al final, con índice reiniciado). Se
    revierte **solo** el escalador (`named_transformers_["numeric"]
    .named_steps["scaler"].inverse_transform`), no el `ColumnTransformer`
    completo -- el imputador no tiene una inversa útil aquí y no hace falta:
    los nulos de `TB`/`DB` ya se imputaron por mediana antes del escalado, así
    que revertir el escalador basta para comparar "en las mismas unidades que
    el dataset original".

    Parameters
    ----------
    X_train, y_train : pandas.DataFrame / pandas.Series
        Datos de TRAIN.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    scaler : str
        Uno de ``params["preprocessing"]["scalers"]``.

    Returns
    -------
    pandas.DataFrame
        Filas sintéticas únicamente, columnas = `NUMERIC_COLS`, en unidades
        originales (post-imputación de mediana, pre-escalado).
    """
    pipe = build_pipeline(scaler, True, DummyClassifier(), params, selector=None)
    sub_pipe = pipe[:-1]  # ["preprocess", "smote"], sin estimador
    X_resampled, _y_resampled = sub_pipe.fit_resample(X_train, y_train)

    n_real = len(X_train)
    synthetic = X_resampled.iloc[n_real:]
    numeric_cols = [c for c in synthetic.columns if c.startswith("numeric__")]

    fitted_scaler = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
    original_units = fitted_scaler.inverse_transform(synthetic[numeric_cols])
    return pd.DataFrame(original_units, columns=[c.removeprefix("numeric__") for c in numeric_cols])


def compare_smote_synthetic_across_scalers(X_train, y_train, params: dict) -> dict:
    """Verificación empírica de Trampa 3: ¿genera SMOTE los mismos pacientes sintéticos bajo ambos escalados?

    Llama a :func:`smote_synthetic_samples_original_units` una vez por
    escalador y compara los pacientes sintéticos resultantes en unidades
    originales. Dos comparaciones, no una sola:

    1. **Fila i vs. fila i** (emparejamiento por orden de generación) -- válido
       porque `imbalanced-learn` genera los sintéticos en un orden determinista
       por fila de origen, verificado empíricamente contra la versión
       instalada en este proyecto (mismo número de sintéticos, mismo orden de
       aparición de las filas reales que los originan).
    2. **Vecino más cercano** (fallback, no depende de ningún supuesto sobre
       el orden interno de la librería) -- para cada sintético de un
       escalador, la distancia euclídea mínima contra todos los del otro.

    Si ambas coincidieran en concluir "son el mismo paciente" para todas las
    filas, el escalado no tendría ningún efecto sobre los sintéticos que
    genera SMOTE. Lo que se mide aquí es cuánto NO coinciden.

    Parameters
    ----------
    X_train, y_train : pandas.DataFrame / pandas.Series
        Datos de TRAIN.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    dict
        ``"n_sinteticos"``, ``"n_identicos_por_posicion"``,
        ``"n_distintos_por_posicion"``, ``"distancia_maxima_por_posicion"``,
        ``"distancia_media_vecino_mas_cercano"`` y
        ``"tabla_distancias"`` (`pandas.DataFrame`, una fila por sintético).
    """
    scalers = params["preprocessing"]["scalers"]
    synthetic_by_scaler = {
        scaler: smote_synthetic_samples_original_units(X_train, y_train, params, scaler) for scaler in scalers
    }
    a, b = synthetic_by_scaler[scalers[0]], synthetic_by_scaler[scalers[1]]

    distancias_por_posicion = np.linalg.norm(a.to_numpy() - b.to_numpy(), axis=1)
    tolerancia = 1e-6
    n_identicos = int((distancias_por_posicion < tolerancia).sum())

    nn = NearestNeighbors(n_neighbors=1).fit(b.to_numpy())
    distancias_vecino_mas_cercano, _ = nn.kneighbors(a.to_numpy())

    return {
        "n_sinteticos": len(a),
        "n_identicos_por_posicion": n_identicos,
        "n_distintos_por_posicion": len(a) - n_identicos,
        "distancia_maxima_por_posicion": float(distancias_por_posicion.max()),
        "distancia_media_vecino_mas_cercano": float(distancias_vecino_mas_cercano.mean()),
        "tabla_distancias": pd.DataFrame(
            {"distancia_por_posicion": distancias_por_posicion, "distancia_vecino_mas_cercano": distancias_vecino_mas_cercano.ravel()}
        ),
    }


# ---------------------------------------------------------------------------
# (f) F8-R5 -- análisis de sensibilidad, filas DB>TB
# ---------------------------------------------------------------------------


def db_gt_tb_row_indices(X: pd.DataFrame) -> pd.Index:
    """Índices de las filas con `TB`/`DB` nulos (las 3 marcadas por `DB > TB`, F8-R5).

    Deriva los índices de los propios datos -- nunca escritos literales en
    `src/` -- porque en `X_train` (pre-`Pipeline`) esas filas siguen siendo
    `NaN` en ambas columnas: la imputación por mediana vive dentro del
    `Pipeline` (Fase 5), no antes.

    Parameters
    ----------
    X : pandas.DataFrame
        Debe incluir las columnas ``TB`` y ``DB`` sin imputar.

    Returns
    -------
    pandas.Index
        Índices de las filas donde `TB` y `DB` son ambos `NaN`.
    """
    return X.index[X["TB"].isna() & X["DB"].isna()]


def sensitivity_without_db_gt_tb_rows(
    X_train, y_train, params: dict, best_params: dict, *, selector=None
) -> pd.DataFrame:
    """Repite la CV del mejor modelo (Fase 7: `logistic_regression`) con y sin las 3 filas `DB>TB` (F8-R5).

    Contrapartida prometida en §5.2 del PRD al decidir imputar (en vez de
    eliminar) las 3 filas con `DB > TB`: demuestra que las conclusiones no
    cambian si esas filas se hubieran excluido. Usa
    ``params["tuning_baseline"]`` (Z-Score + SMOTE) -- la configuración bajo
    la que se declaró ganador a `logistic_regression` en la Fase 6/7 -- para
    que la comparación sea contra la misma cifra ya reportada, no una celda
    arbitraria del factorial.

    Parameters
    ----------
    X_train, y_train : pandas.DataFrame / pandas.Series
        Datos de TRAIN completos (con las 3 filas `DB>TB`).
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    best_params : dict
        `best_params_` de `logistic_regression` (de
        :func:`collect_frozen_best_params`).
    selector : sklearn transformer o None, opcional
        Igual que en :func:`run_factorial`.

    Returns
    -------
    pandas.DataFrame
        2 filas (``variante`` = ``"con_DB_gt_TB"``/``"sin_DB_gt_TB"``), mismas
        columnas que :func:`evaluate_cell`.
    """
    baseline = params["tuning_baseline"]
    filas_a_excluir = db_gt_tb_row_indices(X_train)

    fila_completa = evaluate_cell(
        "logistic_regression", baseline["scaler"], baseline["balancing"], best_params, X_train, y_train, params, selector=selector
    )
    fila_completa["variante"] = "con_DB_gt_TB"

    X_sin = X_train.drop(index=filas_a_excluir)
    y_sin = y_train.drop(index=filas_a_excluir)
    fila_sin = evaluate_cell(
        "logistic_regression", baseline["scaler"], baseline["balancing"], best_params, X_sin, y_sin, params, selector=selector
    )
    fila_sin["variante"] = "sin_DB_gt_TB"

    return pd.DataFrame([fila_completa, fila_sin])


# ---------------------------------------------------------------------------
# (g) F8-R6 -- la predicción: MinMax aplasta la mediana de las variables asimétricas
# ---------------------------------------------------------------------------


def compare_scaled_medians(X_train, params: dict, *, cols: list[str] | None = None) -> pd.DataFrame:
    """Mediana post-escalado de las variables asimétricas, bajo MinMax y Z-Score (F8-R6).

    Ajusta el paso ``"preprocess"`` (sin selector, sin SMOTE, vía
    `build_pipeline`) bajo cada escalador y calcula la mediana escalada de
    cada columna de `cols`. Existe para medir, **después** de haberla escrito
    en markdown, la predicción de que MinMax comprime la mediana de las
    variables con asimetría fuerte hacia el extremo bajo de `[0, 1]` (a
    diferencia de Z-Score, donde la mediana de una variable asimétrica cae
    lejos de 0 pero no está acotada a un rango fijo) -- consecuencia de que
    `MinMaxScaler` fija sus dos extremos en los valores mínimo y máximo
    observados, así que una cola larga hacia la derecha estira el rango y
    comprime el grueso de las observaciones (incluida la mediana) hacia el 0.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Datos de TRAIN.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    cols : list of str o None, opcional
        Columnas a medir. Por defecto, :data:`src.config.SKEWED_COLS`.

    Returns
    -------
    pandas.DataFrame
        Columnas ``columna``, ``escalador``, ``mediana_escalada``.
    """
    if cols is None:
        cols = SKEWED_COLS

    rows = []
    for scaler in params["preprocessing"]["scalers"]:
        pipe = build_pipeline(scaler, False, DummyClassifier(), params, selector=None)
        transformed = pipe.named_steps["preprocess"].fit_transform(X_train)
        for col in cols:
            rows.append(
                {"columna": col, "escalador": scaler, "mediana_escalada": transformed[f"numeric__{col}"].median()}
            )
    return pd.DataFrame(rows)
