"""Tests de la Fase 8: experimento factorial 20 configuraciones (PRD §12.2, encargo de la fase).

`params`, `train_data` y las funciones sin fixture real corren en segundos.
`best_params_by_model`, `factorial_table` y `degenerate_table` son
`scope="module"` a propósito: cada uno ajusta o evalúa datos reales sobre las
20 celdas (~20-30 s en total para el módulo completo) y se comparte entre
todos los tests que lo necesiten, en vez de repetirlo por test.
"""

import inspect
import re

import pytest
from sklearn.feature_selection import SelectKBest, f_classif

from src.config import PROJECT_ROOT, TARGET_COL, load_params
from src.data import load_modeling_data
from src.factorial import (
    collect_frozen_best_params,
    db_gt_tb_row_indices,
    detect_degenerate_cells,
    evaluate_cell,
    run_factorial,
    sensitivity_without_db_gt_tb_rows,
)
from src.models import MODEL_KEYS
from src.splitting import load_split_indices

FORBIDDEN_TEST_REFERENCES = re.compile(r"\b(test_index|test_indices|X_test|y_test)\b")

_INVARIANT_STRICT_CELLS = [("gaussian_nb", "none"), ("decision_tree", "none"), ("gaussian_nb", "smote")]
_STRICT_TOLERANCE = 1e-6
_NOT_INVARIANT_CELLS = [("decision_tree", "smote")]
_NOT_INVARIANT_MIN_DIFF = 1e-3


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def train_data(params):
    df = load_modeling_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    split = load_split_indices(PROJECT_ROOT / params["split"]["indices_path"])
    train_idx = split["train_index"]
    return X.loc[train_idx].drop(columns=["Gender"]), y.loc[train_idx]


@pytest.fixture(scope="module")
def best_params_by_model(params, train_data):
    """Único punto costoso del módulo: 5 GridSearchCV bajo tuning_baseline, una sola vez."""
    X_train, y_train = train_data
    return collect_frozen_best_params(X_train, y_train, params, selector=SelectKBest(f_classif))


@pytest.fixture(scope="module")
def factorial_table(params, train_data, best_params_by_model):
    X_train, y_train = train_data
    return run_factorial(X_train, y_train, params, best_params_by_model, selector=SelectKBest(f_classif))


@pytest.fixture(scope="module")
def degenerate_table(params, train_data, best_params_by_model):
    X_train, y_train = train_data
    return detect_degenerate_cells(X_train, y_train, params, best_params_by_model, selector=SelectKBest(f_classif))


def test_veinte_configuraciones(params, factorial_table):
    """F8-R1: {escaladores} x {balanceo} x 5 modelos -- el número de celdas sale de params.yaml, nunca literal."""
    n_esperado = len(params["preprocessing"]["scalers"]) * len(params["balancing"]["methods"]) * len(MODEL_KEYS)
    assert len(factorial_table) == n_esperado

    combinaciones = list(zip(factorial_table["modelo"], factorial_table["escalador"], factorial_table["balanceo"], strict=True))
    assert len(set(combinaciones)) == n_esperado, "hay celdas repetidas o faltantes"

    assert set(factorial_table["escalador"]) == set(params["preprocessing"]["scalers"])
    assert set(factorial_table["balanceo"]) == set(params["balancing"]["methods"])
    assert set(factorial_table["modelo"]) == set(MODEL_KEYS)

    assert factorial_table["balanced_accuracy_cv"].notna().all()
    assert factorial_table["balanced_accuracy_cv"].between(0, 1).all()


def test_invarianza_condicionada_al_balanceo(factorial_table):
    """Trampa 3: NB y árbol son invariantes al escalado SOLO sin SMOTE (gaussian_nb también con SMOTE).

    La afirmación ingenua del PRD/F6-R7 ("NB y árboles son insensibles al
    escalado, sin condición") es solo parcialmente cierta: SMOTE elige
    vecinos por distancia euclídea, que SÍ depende de la escala relativa
    entre variables -- bajo cada escalador genera pacientes sintéticos
    distintos (verificado en `08_factorial.ipynb` con
    `compare_smote_synthetic_across_scalers`), así que un estimador
    invariante al escalado deja de dar el mismo resultado si se entrena
    sobre datos distintos. `decision_tree` dejó de ser invariante con SMOTE;
    `gaussian_nb` (que agrega media/varianza por variable, no posiciones
    individuales) sigue siéndolo incluso con SMOTE. `svm` no se incluye en
    ninguna de las dos listas: su rejilla usa `gamma="scale"`, que depende de
    la varianza post-escalado, así que no tiene la garantía estructural de
    NB/árbol ni siquiera sin SMOTE -- se reporta en el notebook, no se
    exige aquí.
    """
    pivot = factorial_table.pivot_table(index=["modelo", "balanceo"], columns="escalador", values="balanced_accuracy_cv")
    escaladores = sorted(pivot.columns)
    diff = (pivot[escaladores[0]] - pivot[escaladores[1]]).abs()

    for modelo, balanceo in _INVARIANT_STRICT_CELLS:
        assert diff.loc[(modelo, balanceo)] < _STRICT_TOLERANCE, (
            f"({modelo}, {balanceo}) debería ser invariante al escalado (diff={diff.loc[(modelo, balanceo)]:.6f})"
        )

    for modelo, balanceo in _NOT_INVARIANT_CELLS:
        assert diff.loc[(modelo, balanceo)] > _NOT_INVARIANT_MIN_DIFF, (
            f"({modelo}, {balanceo}) se esperaba NO invariante (SMOTE depende de escala) "
            f"pero diff={diff.loc[(modelo, balanceo)]:.6f} es demasiado pequeña -- revisar si el mecanismo cambió"
        )


def test_factorial_evaluado_en_cv_no_en_test():
    """F8-R2/Trampa 1: ninguna función de src/factorial.py referencia el conjunto de prueba."""
    src_text = (PROJECT_ROOT / "src" / "factorial.py").read_text(encoding="utf-8")
    match = FORBIDDEN_TEST_REFERENCES.search(src_text)
    assert match is None, f"src/factorial.py referencia el conjunto de prueba: {match.group(0)!r}"


def test_hiperparametros_no_reajustados():
    """Trampa 2/ADR-0015: ninguna celda del factorial reajusta hiperparámetros -- los recibe, no los calcula.

    `GridSearchCV` sí se MENCIONA en las docstrings de `src/factorial.py`
    (para explicar de dónde vienen los `best_params_` congelados), pero nunca
    se IMPORTA ni se INSTANCIA -- eso es lo que este test verifica.
    """
    src_text = (PROJECT_ROOT / "src" / "factorial.py").read_text(encoding="utf-8")
    assert "import GridSearchCV" not in src_text and "GridSearchCV(" not in src_text, (
        "src/factorial.py no debería importar ni instanciar GridSearchCV -- los hiperparámetros vienen "
        "congelados de la Fase 6, vía collect_frozen_best_params, no se reajustan por celda"
    )

    for func in (evaluate_cell, run_factorial):
        params_declarados = inspect.signature(func).parameters
        assert "best_params" in params_declarados or "best_params_by_model" in params_declarados, (
            f"{func.__name__} debería recibir best_params como argumento, no recalcularlo"
        )


def test_analisis_de_sensibilidad_existe(params, train_data, best_params_by_model):
    """F8-R5: repetir la CV del ganador (logistic_regression) sin las 3 filas DB>TB no cambia la conclusión."""
    X_train, y_train = train_data
    indices = db_gt_tb_row_indices(X_train)
    assert list(indices) == [246, 261, 279]

    tabla = sensitivity_without_db_gt_tb_rows(
        X_train, y_train, params, best_params_by_model["logistic_regression"], selector=SelectKBest(f_classif)
    )
    assert len(tabla) == 2
    assert set(tabla["variante"]) == {"con_DB_gt_TB", "sin_DB_gt_TB"}

    con = tabla.loc[tabla["variante"] == "con_DB_gt_TB", "balanced_accuracy_cv"].iloc[0]
    sin = tabla.loc[tabla["variante"] == "sin_DB_gt_TB", "balanced_accuracy_cv"].iloc[0]
    assert abs(con - sin) < 0.03, (
        f"balanced_accuracy con/sin las filas DB>TB difiere más de lo esperado (con={con:.4f}, sin={sin:.4f}) "
        "-- las conclusiones deberían sostenerse sin esas 3 filas"
    )


def test_celdas_degeneradas_documentadas(degenerate_table, factorial_table):
    """Trampa 4: al menos una celda predice una sola clase -- verificado con predicciones reales, no inferido del 0.5."""
    degeneradas = degenerate_table[degenerate_table["n_clases_predichas"] == 1]
    assert len(degeneradas) > 0, "se esperaba al menos una celda degenerada (dominio: clase 2.48:1 sin balancear)"

    combinada = degeneradas.merge(factorial_table, on=["modelo", "escalador", "balanceo"], suffixes=("", "_tabla"))
    assert combinada["balanced_accuracy_cv"].apply(lambda v: abs(v - 0.5) < _STRICT_TOLERANCE).all(), (
        "una celda con predicción de una sola clase debe tener balanced_accuracy == 0.5 exacto"
    )

    total_sanos_train = 131
    predice_siempre_enfermo = degeneradas[degeneradas["clase_unica_predicha"] == 1]
    assert (predice_siempre_enfermo["sanos_falsos_positivos"] == total_sanos_train).all(), (
        "si el modelo predice siempre 'enfermo', todos los sanos de train deberían quedar como falsos positivos"
    )
    assert (predice_siempre_enfermo["enfermos_no_detectados"] == 0).all()
