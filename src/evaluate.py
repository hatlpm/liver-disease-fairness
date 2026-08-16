"""Evaluación de los 5 modelos sobre el test congelado (Fase 7, F7-R1..R8).

El módulo se organiza en tres zonas que un revisor puede auditar por
separado, precisamente porque **el test se toca una sola vez** (Trampa 1 del
encargo):

- **Zona A — métricas y formato.** No dependen de ningún dato concreto:
  :func:`compute_metrics`, :func:`apply_threshold` y los `format_*` que arman
  las celdas de la tabla de T7.
- **Zona B — solo train/CV.** Ninguna de estas funciones recibe `X_test` ni
  `y_test`: :func:`fit_all_models` ajusta los 5 `GridSearchCV`,
  :func:`pick_winner_by_cv` elige el ganador por `balanced_accuracy` de CV
  (Trampa 2 -- nunca por el test), :func:`threshold_curve_cv` explora
  umbrales con `cross_val_predict` sobre train (Trampa 3).
- **Zona C — el único punto de entrada a test.**
  :func:`predict_all_on_test` es la única función de todo el proyecto que
  llama `.predict()` sobre `X_test`; todo lo demás (:func:`tabla_comparativa`,
  las matrices de confusión, el bootstrap) consume ese resultado ya
  calculado, sin volver a predecir.

Ninguna función de este módulo hardcodea `average`, `pos_label`, el umbral de
decisión o la semilla: todos se leen de `params` (F7-R2, H-R1).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_predict

from src.config import SKLEARN_ZERO_DIVISION
from src.models import MODEL_KEYS, build_cv, fit_grid_search, null_classifier_floor

NULL_ROW_LABEL = 'Clasificador nulo (predice siempre "enfermo")'

T7_REQUIRED_COLUMNS = [
    "Modelo",
    "Hiperparámetros y cuadrícula de búsqueda",
    "Hiperparámetros de mejor rendimiento",
    "Métricas de mejor rendimiento",
]

_BEST_PARAMS_STEP_PREFIXES = ("estimator__", "selector__")


# ---------------------------------------------------------------------------
# Zona A -- métricas y formato (sin dependencia de datos concretos)
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_pred, params: dict) -> dict:
    """Accuracy, precisión, recall, F1 y balanced accuracy como dict explícito.

    `average` y `pos_label` se leen de ``params["metrics"]`` (F7-R2) -- nunca
    literales. `accuracy_score` y `balanced_accuracy_score` no toman esos
    argumentos (no distinguen clase positiva), así que se calculan aparte.

    Parameters
    ----------
    y_true : array-like
        Etiquetas verdaderas, en {0, 1}.
    y_pred : array-like
        Etiquetas predichas, en {0, 1}.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    dict
        Con claves ``"accuracy"``, ``"precision"``, ``"recall"``, ``"f1"``
        y ``"balanced_accuracy"``.
    """
    average = params["metrics"]["average"]
    pos_label = params["metrics"]["pos_label"]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=SKLEARN_ZERO_DIVISION
        ),
        "recall": recall_score(
            y_true, y_pred, average=average, pos_label=pos_label, zero_division=SKLEARN_ZERO_DIVISION
        ),
        "f1": f1_score(y_true, y_pred, average=average, pos_label=pos_label, zero_division=SKLEARN_ZERO_DIVISION),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def apply_threshold(proba_positive, threshold: float) -> np.ndarray:
    """Convierte probabilidades de la clase positiva en etiquetas duras.

    Parameters
    ----------
    proba_positive : array-like
        `P(clase positiva)` por fila, típicamente `predict_proba(X)[:, 1]`.
    threshold : float
        Punto de corte: ``proba >= threshold`` se etiqueta como positivo.

    Returns
    -------
    numpy.ndarray
        Etiquetas en {0, 1}, mismo largo que `proba_positive`.
    """
    return (np.asarray(proba_positive) >= threshold).astype(int)


def format_grid_description(name: str, params: dict) -> str:
    """Rejilla de `name` desde `params.yaml`, formateada para la columna 2 de §3.2.

    Parameters
    ----------
    name : str
        Uno de :data:`src.models.MODEL_KEYS`.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    str
        P. ej. ``"C: [0.01, 0.1, 1, 10, 100] · solver: ['liblinear', 'saga']"``.
    """
    return " · ".join(f"{hp}: {values}" for hp, values in params["grids"][name].items())


def format_best_params(best_params: dict) -> str:
    """`best_params_` de `GridSearchCV` (con prefijos de paso) para la columna 3 de §3.2.

    Quita los prefijos ``"estimator__"``/``"selector__"`` que exige
    `GridSearchCV` cuando el estimador vive dentro de un `Pipeline`
    (:func:`src.models.build_search_grid`) -- la tabla de T7 no necesita
    mostrar la mecánica interna del pipeline, solo el valor de cada
    hiperparámetro.

    Parameters
    ----------
    best_params : dict
        `search.best_params_` de un `GridSearchCV` ya ajustado.

    Returns
    -------
    str
        P. ej. ``"C: 0.1 · solver: liblinear"``.
    """
    parts = []
    for key, value in best_params.items():
        short_key = key
        for prefix in _BEST_PARAMS_STEP_PREFIXES:
            if short_key.startswith(prefix):
                short_key = short_key[len(prefix) :]
                break
        parts.append(f"{short_key}: {value}")
    return " · ".join(parts)


def format_metrics_cell(metrics: dict) -> str:
    """Empaqueta accuracy/precisión/recall/F1 en el texto de la columna 4 de §3.2.

    `balanced_accuracy` no entra aquí a propósito: la tabla de T7 la reporta
    en una columna aparte (Trampa 4 del encargo) para no romper el formato
    de 4 columnas exigido por §3.2.

    Parameters
    ----------
    metrics : dict
        Salida de :func:`compute_metrics` (o de
        :func:`src.models.null_classifier_floor`).

    Returns
    -------
    str
    """
    return (
        f"Accuracy: {metrics['accuracy']:.4f} · "
        f"Precisión: {metrics['precision']:.4f} · "
        f"Recall: {metrics['recall']:.4f} · "
        f"F1: {metrics['f1']:.4f}"
    )


# ---------------------------------------------------------------------------
# Zona B -- solo train/CV (ninguna función de aquí abajo recibe X_test/y_test)
# ---------------------------------------------------------------------------


def fit_all_models(X_train, y_train, params: dict, selector=None) -> dict:
    """Ajusta los 5 `GridSearchCV` exigidos, uno por algoritmo (train/CV, nunca test).

    Reutiliza :func:`src.models.fit_grid_search`, que ya ajusta bajo
    ``params["tuning_baseline"]`` (Z-Score + SMOTE, fijado en la Fase 6) y
    refita el mejor estimador sobre el `X_train`/`y_train` completo que se le
    pase (`GridSearchCV(refit=True)`).

    Parameters
    ----------
    X_train : pandas.DataFrame
        Variables de entrada de TRAIN (variante oficial: sin `Gender`).
    y_train : pandas.Series
        Variable objetivo de TRAIN.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    selector : sklearn transformer o None, opcional
        Selector de variables sin ajustar, ver :func:`src.pipelines.build_pipeline`.

    Returns
    -------
    dict
        ``{nombre_modelo: GridSearchCV ya ajustado}``, una entrada por cada
        uno de :data:`src.models.MODEL_KEYS`.
    """
    return {name: fit_grid_search(name, X_train, y_train, params, selector=selector) for name in MODEL_KEYS}


def pick_winner_by_cv(searches: dict) -> str:
    """Elige el modelo ganador por `balanced_accuracy` de CV (Trampa 2 -- nunca por el test).

    Firma deliberada: no recibe el conjunto de prueba, ni ninguna
    predicción sobre él -- estructuralmente no puede mirar el test
    (verificado en
    `tests/test_fase7_evaluacion.py::test_ganador_elegido_por_cv_no_por_test`).

    Parameters
    ----------
    searches : dict
        Salida de :func:`fit_all_models` (o un dict compatible con atributo
        ``best_score_`` por valor).

    Returns
    -------
    str
        Nombre del modelo con mayor ``best_score_``.
    """
    return max(searches, key=lambda name: searches[name].best_score_)


def threshold_curve_cv(pipeline, X_train, y_train, params: dict, thresholds=None) -> pd.DataFrame:
    """Barrido de umbrales sobre probabilidades de CV en TRAIN (Trampa 3, F7-R7/B6).

    Usa ``cross_val_predict(..., method="predict_proba")`` para obtener,
    para cada fila de `X_train`, la probabilidad que predijo el pliegue en
    que esa fila quedó de validación -- nunca un ajuste sobre el conjunto
    completo, y nunca el test. `pipeline` se clona internamente en cada
    pliegue (comportamiento estándar de `cross_val_predict`), así que puede
    pasarse ya ajustado (p. ej. ``search.best_estimator_``) sin reutilizar su
    estado. Requiere que `pipeline` implemente `predict_proba` -- `SVC` sin
    `probability=True` no lo tiene; en la práctica esta función solo se
    invoca sobre el modelo ganador (`logistic_regression`), que sí lo tiene.

    Parameters
    ----------
    pipeline : sklearn estimator
        Pipeline (ajustado o no) con `predict_proba`.
    X_train, y_train : pandas.DataFrame / pandas.Series
        Datos de TRAIN.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    thresholds : list of float o None, opcional
        Umbrales a evaluar. Por defecto, ``params["evaluation"]["threshold_grid"]``.

    Returns
    -------
    pandas.DataFrame
        Una fila por umbral, columnas ``threshold`` + las 5 de
        :func:`compute_metrics`.
    """
    if thresholds is None:
        thresholds = params["evaluation"]["threshold_grid"]
    proba_positive = cross_val_predict(pipeline, X_train, y_train, cv=build_cv(params), method="predict_proba")[:, 1]
    rows = []
    for threshold in thresholds:
        y_pred = apply_threshold(proba_positive, threshold)
        rows.append({"threshold": threshold, **compute_metrics(y_train, y_pred, params)})
    return pd.DataFrame(rows)


def recommend_threshold(curve: pd.DataFrame, min_recall: float) -> pd.Series:
    """Punto de cribado recomendado: el umbral más alto que cumple `min_recall`.

    Regla reproducible en vez de un punto elegido a ojo sobre el gráfico:
    entre los umbrales de `curve` con `recall >= min_recall`, se queda con
    el más alto -- el que sacrifica menos precisión sin dejar de cumplir el
    piso de recall exigido.

    Parameters
    ----------
    curve : pandas.DataFrame
        Salida de :func:`threshold_curve_cv`.
    min_recall : float
        Recall mínimo aceptable, en (0, 1) -- decisión de producto, no
        estadística (``params["evaluation"]["screening_min_recall"]``).

    Returns
    -------
    pandas.Series
        La fila de `curve` elegida.
    """
    eligible = curve[curve["recall"] >= min_recall]
    return eligible.sort_values("threshold", ascending=False).iloc[0]


# ---------------------------------------------------------------------------
# Zona C -- el único punto de entrada a test + todo lo que consume su salida
# ---------------------------------------------------------------------------


def predict_all_on_test(searches: dict, X_test) -> dict:
    """Predice los 5 modelos sobre el test -- el ÚNICO `.predict()` sobre test del proyecto (F7-R4).

    Todo lo demás en este módulo (:func:`tabla_comparativa`,
    :func:`confusion_matrix_array`, :func:`confusion_matrix_narrative`,
    :func:`bootstrap_metric_ci`) consume el resultado de esta función en vez
    de volver a llamar `.predict()`.

    Parameters
    ----------
    searches : dict
        Salida de :func:`fit_all_models`.
    X_test : pandas.DataFrame
        Variables de entrada del test congelado (variante oficial: sin `Gender`).

    Returns
    -------
    dict
        ``{nombre_modelo: numpy.ndarray de predicciones}``.
    """
    return {name: search.best_estimator_.predict(X_test) for name, search in searches.items()}


def evaluate_at_threshold(pipeline, X, y_true, params: dict, threshold: float | None = None) -> dict:
    """Métricas de `pipeline` sobre `X`/`y_true` a un umbral explícito de `predict_proba`.

    Uso general (no específico de test): el llamador decide si `X`/`y_true`
    son de train, de un pliegue de CV o de test. Con ``threshold=None``
    (por defecto) usa ``params["metrics"]["decision_threshold"]`` -- el
    mismo umbral oficial de la tabla de T7 -- en vez de un literal aparte.

    Parameters
    ----------
    pipeline : sklearn estimator
        Pipeline ajustado, con `predict_proba`.
    X : pandas.DataFrame
        Variables de entrada.
    y_true : pandas.Series
        Variable objetivo verdadera.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    threshold : float o None, opcional
        Umbral de decisión. Por defecto, ``params["metrics"]["decision_threshold"]``.

    Returns
    -------
    dict
        Salida de :func:`compute_metrics`.
    """
    if threshold is None:
        threshold = params["metrics"]["decision_threshold"]
    proba_positive = pipeline.predict_proba(X)[:, 1]
    y_pred = apply_threshold(proba_positive, threshold)
    return compute_metrics(y_true, y_pred, params)


def tabla_comparativa(searches: dict, y_pred_by_model: dict, y_test, params: dict) -> pd.DataFrame:
    """La tabla oficial de T7, en el formato exacto de §3.2 del PRD + fila del suelo nulo.

    Una fila por cada uno de :data:`src.models.MODEL_KEYS` (columnas 1-4 de
    §3.2, vía los `format_*` de la Zona A) más una fila final para el
    clasificador nulo (Trampa 4, vía :func:`src.models.null_classifier_floor`
    sobre el `y_test` real -- nunca las cifras citadas a mano). Añade una
    quinta columna ``"Balanced accuracy"`` sin romper las 4 exigidas.

    Recibe predicciones ya calculadas, no un pipeline ni `X_test`: mantiene
    la función unitariamente testeable con datos sintéticos, sin ajustar
    ningún `GridSearchCV` real ni tocar el test congelado.

    Parameters
    ----------
    searches : dict
        Salida de :func:`fit_all_models` (o un dict compatible con atributos
        ``best_params_``/``best_score_`` por valor).
    y_pred_by_model : dict
        Salida de :func:`predict_all_on_test` (o equivalente).
    y_test : array-like
        Etiquetas verdaderas del test.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    pandas.DataFrame
        Columnas: :data:`T7_REQUIRED_COLUMNS` + ``"Balanced accuracy"``.
        `len(MODEL_KEYS) + 1` filas.
    """
    rows = []
    for name in MODEL_KEYS:
        metrics = compute_metrics(y_test, y_pred_by_model[name], params)
        rows.append(
            {
                "Modelo": name,
                "Hiperparámetros y cuadrícula de búsqueda": format_grid_description(name, params),
                "Hiperparámetros de mejor rendimiento": format_best_params(searches[name].best_params_),
                "Métricas de mejor rendimiento": format_metrics_cell(metrics),
                "Balanced accuracy": round(metrics["balanced_accuracy"], 4),
            }
        )

    floor = null_classifier_floor(y_test, pos_label=params["metrics"]["pos_label"])
    rows.append(
        {
            "Modelo": NULL_ROW_LABEL,
            "Hiperparámetros y cuadrícula de búsqueda": "—",
            "Hiperparámetros de mejor rendimiento": "—",
            "Métricas de mejor rendimiento": format_metrics_cell(floor),
            "Balanced accuracy": round(floor["balanced_accuracy"], 4),
        }
    )
    return pd.DataFrame(rows, columns=[*T7_REQUIRED_COLUMNS, "Balanced accuracy"])


def confusion_matrix_array(y_true, y_pred, pos_label: int = 1) -> np.ndarray:
    """Matriz de confusión 2x2, con el orden de clases fijado explícitamente (F7-R6).

    Parameters
    ----------
    y_true, y_pred : array-like
        Etiquetas verdaderas y predichas, en {0, 1}.
    pos_label : int, opcional
        Clase positiva. Por defecto ``1`` (enfermo, F4-R5).

    Returns
    -------
    numpy.ndarray
        Matriz 2x2 ``[[TN, FP], [FN, TP]]`` -- fila 0 / columna 0 es la
        clase negativa, nunca dejado al orden por defecto de `sklearn`.
    """
    neg_label = 1 - pos_label
    return confusion_matrix(y_true, y_pred, labels=[neg_label, pos_label])


def confusion_matrix_narrative(y_true, y_pred, pos_label: int = 1) -> dict:
    """Matriz de confusión leída en pacientes reales, no en porcentajes (F7-R6).

    Parameters
    ----------
    y_true, y_pred : array-like
        Etiquetas verdaderas y predichas, en {0, 1}.
    pos_label : int, opcional
        Clase positiva. Por defecto ``1`` (enfermo).

    Returns
    -------
    dict
        ``"enfermos_detectados"`` (TP), ``"enfermos_no_detectados"`` (FN),
        ``"sanos_correctos"`` (TN), ``"sanos_falsos_positivos"`` (FP).
    """
    tn, fp, fn, tp = confusion_matrix_array(y_true, y_pred, pos_label=pos_label).ravel()
    return {
        "enfermos_detectados": int(tp),
        "enfermos_no_detectados": int(fn),
        "sanos_correctos": int(tn),
        "sanos_falsos_positivos": int(fp),
    }


def bootstrap_indices(n: int, n_iterations: int, seed: int) -> np.ndarray:
    """Índices de remuestreo bootstrap, compartibles entre modelos para comparaciones pareadas.

    Parameters
    ----------
    n : int
        Tamaño de la muestra original (p. ej. `len(y_test)`).
    n_iterations : int
        Número de remuestreos.
    seed : int
        Semilla del generador -- reproducibilidad (`params["seed"]`).

    Returns
    -------
    numpy.ndarray
        Forma ``(n_iterations, n)``, cada fila son `n` índices en `[0, n)`
        muestreados con reemplazo.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_iterations, n))


def bootstrap_metric_ci(
    y_true,
    y_pred,
    metric_name: str,
    params: dict,
    n_iterations: int | None = None,
    seed: int | None = None,
    indices: np.ndarray | None = None,
) -> dict:
    """Intervalo de confianza bootstrap (percentil) de una métrica sobre el test (F7-R8).

    Con ~114 filas de test, dos modelos pueden estar separados por menos de
    lo que el muestreo explica por azar -- este intervalo es lo que permite
    decir "indistinguibles" en vez de declarar un ganador que es ruido.

    Parameters
    ----------
    y_true, y_pred : array-like
        Etiquetas verdaderas y predichas (de :func:`predict_all_on_test`).
    metric_name : str
        Una de las claves de :func:`compute_metrics` (p. ej.
        ``"balanced_accuracy"``).
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    n_iterations : int o None, opcional
        Número de remuestreos. Por defecto, ``params["evaluation"]["bootstrap"]["n_iterations"]``.
    seed : int o None, opcional
        Semilla. Por defecto, ``params["seed"]``.
    indices : numpy.ndarray o None, opcional
        Índices de remuestreo ya generados (:func:`bootstrap_indices`) --
        para comparar dos modelos sobre exactamente los mismos remuestreos.
        Si se pasa, `n_iterations`/`seed` se ignoran.

    Returns
    -------
    dict
        ``"metric"``, ``"point_estimate"``, ``"ci_lower"``, ``"ci_upper"``,
        ``"n_iterations"``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if indices is None:
        if n_iterations is None:
            n_iterations = params["evaluation"]["bootstrap"]["n_iterations"]
        if seed is None:
            seed = params["seed"]
        indices = bootstrap_indices(len(y_true), n_iterations, seed)

    point_estimate = compute_metrics(y_true, y_pred, params)[metric_name]
    samples = np.array([compute_metrics(y_true[idx], y_pred[idx], params)[metric_name] for idx in indices])

    confidence_level = params["evaluation"]["bootstrap"]["confidence_level"]
    alpha = 1 - confidence_level
    ci_lower, ci_upper = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {
        "metric": metric_name,
        "point_estimate": point_estimate,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_iterations": len(indices),
    }
