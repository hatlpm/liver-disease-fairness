"""Los 5 algoritmos exigidos, sus rejillas y el ajuste vía Grid Search (Fase 6, F6-R1..R8).

Construye, a partir de `params.yaml`, los cinco estimadores del enunciado
(regresión logística, KNN, Naive Bayes, árbol de decisión, SVM) y traduce
sus rejillas de ``params["grids"]`` a los nombres de parámetro que espera
`GridSearchCV` cuando el estimador vive dentro de un `Pipeline` bajo el paso
``"estimator"`` (prefijo ``estimator__``). Ninguna rejilla vive en este
módulo ni en el notebook (F6-R3): si un valor no está en `params.yaml`, no
existe.

`fit_grid_search` combina esto con :func:`src.pipelines.build_pipeline` bajo
la configuración fija de ``params["tuning_baseline"]`` -- los hiperparámetros
se ajustan UNA vez, no en cada celda del experimento factorial de la Fase 8,
para que el efecto del preprocesamiento (T8) no se confunda con el efecto de
volver a ajustar. El `k` del selector de variables se optimiza dentro de la
misma validación cruzada (F6A-R2), nunca por fuera.

Advertencia sobre Naive Bayes (F6-R8)
--------------------------------------
`GaussianNB` asume que cada variable, condicionada a la clase, sigue una
distribución normal. Las 9 variables numéricas de este dataset tienen
asimetría de hasta 10.5 (`Sgot`, ver Fase 2c del EDA) -- el supuesto se
viola abiertamente. Esto se reporta como **limitación del modelo**, no como
un error a corregir: es uno de los cinco algoritmos que el enunciado exige
evaluar, y que su desempeño resulte peor es en sí mismo evidencia de cuánto
le cuesta el incumplimiento del supuesto -- no una razón para sustituirlo.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import LOGREG_MAX_ITER

MODEL_KEYS = ["logistic_regression", "knn", "gaussian_nb", "decision_tree", "svm"]

_ESTIMATOR_FACTORIES = {
    "logistic_regression": lambda seed: LogisticRegression(random_state=seed, max_iter=LOGREG_MAX_ITER),
    "knn": lambda _seed: KNeighborsClassifier(),
    "gaussian_nb": lambda _seed: GaussianNB(),
    "decision_tree": lambda seed: DecisionTreeClassifier(random_state=seed),
    "svm": lambda seed: SVC(random_state=seed),
}


def build_estimator(name: str, params: dict):
    """Construye una instancia sin ajustar del estimador `name`.

    KNN y `GaussianNB` no tienen componente aleatorio en `fit`, así que no
    reciben semilla (`sklearn` ni la acepta como parámetro). Los otros tres
    sí, y la toman de ``params["seed"]`` -- nunca literal (F6-R1, F6-R2).

    Parameters
    ----------
    name : str
        Uno de :data:`MODEL_KEYS`.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    sklearn estimator
        Instancia sin ajustar, lista para entrar como paso ``"estimator"``
        de :func:`src.pipelines.build_pipeline`.

    Raises
    ------
    ValueError
        Si `name` no es uno de los cinco algoritmos exigidos.
    """
    if name not in _ESTIMATOR_FACTORIES:
        raise ValueError(f"name={name!r} no es uno de los 5 algoritmos exigidos: {MODEL_KEYS}")
    return _ESTIMATOR_FACTORIES[name](params["seed"])


def build_search_grid(
    name: str, params: dict, *, include_selector: bool = True, step_name: str = "estimator"
) -> dict:
    """Traduce ``params["grids"][name]`` a claves de `GridSearchCV` con prefijo de paso.

    Ninguna rejilla se escribe literal aquí: se lee completa de `params`
    (F6-R3). Si ``params["feature_selection"]["enabled"]`` es verdadero, se
    añade ``"selector__k"`` con los valores de
    ``params["feature_selection"]["k_values"]`` -- el `k` del selector se
    busca en la MISMA rejilla que los hiperparámetros del modelo, dentro de
    la misma validación cruzada (F6A-R2).

    Parameters
    ----------
    name : str
        Uno de :data:`MODEL_KEYS`.
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    include_selector : bool, opcional
        Si ``False``, omite ``"selector__k"`` aunque la selección esté
        habilitada en `params` (para ajustar un pipeline sin paso
        `"selector"`). Por defecto ``True``.
    step_name : str, opcional
        Nombre del paso del `Pipeline` que contiene al estimador. Por
        defecto ``"estimator"`` (el que usa
        :func:`src.pipelines.build_pipeline`).

    Returns
    -------
    dict
        Rejilla lista para pasar como ``param_grid`` a `GridSearchCV`.
    """
    grid = {f"{step_name}__{hp}": values for hp, values in params["grids"][name].items()}
    if include_selector and params["feature_selection"]["enabled"]:
        grid["selector__k"] = params["feature_selection"]["k_values"]
    return grid


def build_cv(params: dict) -> StratifiedKFold:
    """Construye la validación cruzada estratificada de ajuste (F6-R4).

    Lee ``params["cv"]["tuning"]`` y ``params["seed"]`` -- nunca literal.
    `StratifiedKFold`, no `KFold`: preserva la proporción de clase
    (2.49:1) en cada pliegue, condición para que SMOTE tenga suficiente
    minoría dentro de cada pliegue de entrenamiento.

    Parameters
    ----------
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    sklearn.model_selection.StratifiedKFold
    """
    cv_cfg = params["cv"]["tuning"]
    return StratifiedKFold(n_splits=cv_cfg["n_splits"], shuffle=cv_cfg["shuffle"], random_state=params["seed"])


def fit_grid_search(name: str, X, y, params: dict, *, selector=None, n_jobs: int = -1) -> GridSearchCV:
    """Ajusta `GridSearchCV` para el algoritmo `name` bajo la configuración base de ajuste.

    Combina, en un único `Pipeline`, el preprocesamiento + (opcional)
    selector + (opcional) `SMOTE` + el estimador `name`, bajo la
    configuración fija de ``params["tuning_baseline"]``. La rejilla
    combina ``params["grids"][name]`` con ``params["feature_selection"]["k_values"]``
    cuando se pasa un `selector` (F6A-R2). La búsqueda optimiza
    ``params["metrics"]["optimize_for"]`` (F6-R6, ver ADR-0012) por CV
    estratificada sobre `X`/`y` -- nunca sobre el test (F6-R4).

    Parameters
    ----------
    name : str
        Uno de :data:`MODEL_KEYS`.
    X : pandas.DataFrame
        Variables de entrada de TRAIN. Para la variante "sin `Gender`" de
        F6A-R4, pasar ``X.drop(columns=["Gender"])``. Nunca el conjunto de
        prueba -- esta función no lo conoce ni lo recibe.
    y : pandas.Series
        Variable objetivo de TRAIN (`Selector`, 1 = enfermo).
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.
    selector : sklearn transformer o None, opcional
        Selector de variables sin ajustar (p. ej. ``SelectKBest(f_classif)``)
        a insertar dentro del `Pipeline`. Si se pasa, su rejilla ``k`` se
        añade automáticamente a la búsqueda. Por defecto ``None``.
    n_jobs : int, opcional
        Paralelismo de cómputo de `GridSearchCV`. Por defecto ``-1``.

    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Ya ajustado (``.fit(X, y)`` ejecutado). `best_estimator_`,
        `best_params_` y `best_score_` disponibles tal cual los deja
        `sklearn`.
    """
    from src.pipelines import build_pipeline

    baseline = params["tuning_baseline"]
    pipe = build_pipeline(
        scaler=baseline["scaler"],
        use_smote=(baseline["balancing"] == "smote"),
        estimator=build_estimator(name, params),
        params=params,
        selector=selector,
    )
    grid = build_search_grid(name, params, include_selector=selector is not None)
    search = GridSearchCV(
        pipe,
        grid,
        scoring=params["metrics"]["optimize_for"],
        cv=build_cv(params),
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X, y)
    return search


def null_classifier_floor(y, pos_label: int = 1) -> dict:
    """Métricas del clasificador nulo "predice siempre `pos_label`" (Trampa 1).

    Es el suelo de comparación obligatorio de la fase: con clases 2.49:1 y
    la mayoritaria marcada como positiva, este clasificador que no mira
    ningún dato de entrada saca accuracy y F1 altos sin haber aprendido
    nada. Ningún modelo entrenado tiene mérito si no lo supera -- y ninguna
    métrica de selección de hiperparámetros tiene sentido si el propio
    suelo la satura (F6-R6).

    Parameters
    ----------
    y : array-like
        Vector de etiquetas verdaderas, en {0, 1}.
    pos_label : int, opcional
        Clase que el clasificador nulo predice siempre. Por defecto ``1``
        (enfermo, F4-R5/A4).

    Returns
    -------
    dict
        Con claves ``"accuracy"``, ``"recall"``, ``"precision"``, ``"f1"``
        y ``"balanced_accuracy"``.
    """
    y = np.asarray(y)
    y_null = np.full_like(y, fill_value=pos_label)
    return {
        "accuracy": accuracy_score(y, y_null),
        "recall": recall_score(y, y_null, pos_label=pos_label),
        "precision": precision_score(y, y_null, pos_label=pos_label),
        "f1": f1_score(y, y_null, pos_label=pos_label),
        "balanced_accuracy": balanced_accuracy_score(y, y_null),
    }
