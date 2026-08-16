"""Tests de la Fase 7: evaluación sobre el test congelado (PRD §12.2, encargo de la fase).

6 de los 7 tests exigidos usan datos SINTÉTICOS a propósito: `src/evaluate.py`
está diseñado para que la tabla de T7, el suelo nulo y el bootstrap sean
funciones puras sobre predicciones ya calculadas, así que se pueden probar
sin ajustar ningún `GridSearchCV` real (serían ~1-2 min, igual que en la Fase
6) -- la suite corre en segundos. La única excepción es
`test_suelo_nulo_en_test_es_el_esperado`, que necesita el `y_test` real
congelado -- pero solo LEE sus etiquetas (ningún modelo se ajusta ni se
evalúa), la misma justificación que ya usó `notebooks/06_modelos.ipynb` para
citar esa misma cifra sin recalcularla.
"""

import copy
import inspect
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

from src.config import PROJECT_ROOT, TARGET_COL, load_params
from src.data import load_modeling_data
from src.evaluate import (
    NULL_ROW_LABEL,
    T7_REQUIRED_COLUMNS,
    bootstrap_metric_ci,
    compute_metrics,
    evaluate_at_threshold,
    pick_winner_by_cv,
    tabla_comparativa,
)
from src.models import MODEL_KEYS, null_classifier_floor
from src.splitting import load_split_indices

FORBIDDEN_TEST_REFERENCES = re.compile(r"\b(test_index|test_indices|X_test|y_test)\b")


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def real_y_test(params):
    """Etiquetas verdaderas del test congelado -- solo lectura, ningún modelo ajustado."""
    df = load_modeling_data()
    split = load_split_indices(PROJECT_ROOT / params["split"]["indices_path"])
    return df[TARGET_COL].loc[split["test_index"]]


@pytest.fixture
def synthetic_table_inputs(params):
    """searches/y_pred_by_model/y_test fabricados -- ejercitan tabla_comparativa sin ajustar nada."""
    y_test = pd.Series([1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    y_pred_by_model = {name: np.array([1, 1, 1, 0, 1, 1, 0, 0, 1, 0]) for name in MODEL_KEYS}

    class _FakeSearch:
        def __init__(self, best_params, best_score):
            self.best_params_ = best_params
            self.best_score_ = best_score

    fake_searches = {
        name: _FakeSearch(
            best_params={f"estimator__{hp}": values[0] for hp, values in params["grids"][name].items()},
            best_score=0.7,
        )
        for name in MODEL_KEYS
    }
    return fake_searches, y_pred_by_model, y_test


def test_tabla_tiene_columnas_exigidas(params, synthetic_table_inputs):
    """F7-R5: la tabla incluye las 4 columnas exactas de §3.2 (puede tener más, nunca menos)."""
    fake_searches, y_pred_by_model, y_test = synthetic_table_inputs
    tabla = tabla_comparativa(fake_searches, y_pred_by_model, y_test, params)
    assert set(T7_REQUIRED_COLUMNS) <= set(tabla.columns)
    assert list(tabla.columns[:4]) == T7_REQUIRED_COLUMNS


def test_tabla_incluye_suelo_nulo(params, synthetic_table_inputs):
    """Trampa 4: la tabla siempre lleva la fila del clasificador nulo, calculada, no citada a mano."""
    fake_searches, y_pred_by_model, y_test = synthetic_table_inputs
    tabla = tabla_comparativa(fake_searches, y_pred_by_model, y_test, params)

    assert len(tabla) == len(MODEL_KEYS) + 1
    filas_nulas = tabla[tabla["Modelo"] == NULL_ROW_LABEL]
    assert len(filas_nulas) == 1

    floor = null_classifier_floor(y_test, pos_label=params["metrics"]["pos_label"])
    celda = filas_nulas.iloc[0]["Métricas de mejor rendimiento"]
    assert f"{floor['accuracy']:.4f}" in celda
    assert f"{floor['recall']:.4f}" in celda
    assert f"{floor['precision']:.4f}" in celda
    assert f"{floor['f1']:.4f}" in celda
    assert filas_nulas.iloc[0]["Balanced accuracy"] == round(floor["balanced_accuracy"], 4)


def test_metricas_con_pos_label_1(params):
    """F7-R2: average='binary'/pos_label=1 se LEEN de params, no están hardcodeados."""
    y_true = [1, 1, 1, 0, 0]
    y_pred = [1, 0, 0, 0, 1]

    assert params["metrics"]["average"] == "binary"
    assert params["metrics"]["pos_label"] == 1

    metrics = compute_metrics(y_true, y_pred, params)
    assert metrics["recall"] == recall_score(y_true, y_pred, average="binary", pos_label=1)

    mutado = copy.deepcopy(params)
    mutado["metrics"]["pos_label"] = 0
    metrics_mutado = compute_metrics(y_true, y_pred, mutado)
    assert metrics_mutado["recall"] != metrics["recall"], (
        "compute_metrics no reaccionó a un pos_label distinto en params -- probablemente hardcodeado"
    )


def test_suelo_nulo_en_test_es_el_esperado(real_y_test, params):
    """F7-R3/Trampa 4: el suelo nulo EN TEST reproduce exactamente las cifras del encargo."""
    assert len(real_y_test) == 114
    assert real_y_test.value_counts().to_dict() == {1: 81, 0: 33}

    floor = null_classifier_floor(real_y_test, pos_label=params["metrics"]["pos_label"])
    assert round(floor["accuracy"], 4) == 0.7105
    assert floor["recall"] == 1.0
    assert round(floor["precision"], 4) == 0.7105
    assert round(floor["f1"], 4) == 0.8308
    assert floor["balanced_accuracy"] == 0.5


def test_umbral_declarado_en_params(params):
    """F7-R7/B6: el umbral oficial (0.5) vive en params.yaml, no como literal en src/evaluate.py."""
    assert params["metrics"]["decision_threshold"] == 0.5

    rng = np.random.default_rng(params["seed"])
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    modelo = LogisticRegression(random_state=params["seed"]).fit(X, y)

    metrics_default = evaluate_at_threshold(modelo, X, y, params, threshold=None)
    metrics_explicito = evaluate_at_threshold(modelo, X, y, params, threshold=0.5)
    assert metrics_default == metrics_explicito, (
        "threshold=None debería resolver al mismo resultado que threshold=0.5 explícito -- "
        "prueba que el default se lee de params, no que coincide con 0.5 por casualidad"
    )

    src_text = (PROJECT_ROOT / "src" / "evaluate.py").read_text(encoding="utf-8")
    assert "0.5" not in src_text, "src/evaluate.py tiene un umbral literal en vez de leerlo de params"


def test_intervalos_reportados(params):
    """F7-R8: bootstrap_metric_ci produce un intervalo reproducible alrededor del punto estimado."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=50)
    y_pred = rng.integers(0, 2, size=50)

    ci = bootstrap_metric_ci(y_true, y_pred, "accuracy", params)
    assert ci["ci_lower"] <= ci["point_estimate"] <= ci["ci_upper"]
    assert ci["n_iterations"] == params["evaluation"]["bootstrap"]["n_iterations"]

    ci_repetido = bootstrap_metric_ci(y_true, y_pred, "accuracy", params)
    assert ci_repetido == ci, "misma semilla (params['seed']) debería dar el mismo intervalo"

    ci_rapido = bootstrap_metric_ci(y_true, y_pred, "accuracy", params, n_iterations=100)
    assert ci_rapido["n_iterations"] == 100


def test_ganador_elegido_por_cv_no_por_test():
    """Trampa 2: pick_winner_by_cv elige por balanced_accuracy de CV -- estructuralmente no ve el test."""

    class _FakeSearch:
        def __init__(self, best_score):
            self.best_score_ = best_score

    fake_searches = {"model_a": _FakeSearch(0.90), "model_b": _FakeSearch(0.60)}
    assert pick_winner_by_cv(fake_searches) == "model_a"

    firma = inspect.signature(pick_winner_by_cv)
    assert not FORBIDDEN_TEST_REFERENCES.search(" ".join(firma.parameters)), (
        "pick_winner_by_cv no debería declarar ningún parámetro relacionado con el test"
    )
    fuente = inspect.getsource(pick_winner_by_cv)
    match = FORBIDDEN_TEST_REFERENCES.search(fuente)
    assert match is None, f"pick_winner_by_cv referencia el conjunto de prueba: {match.group(0)!r}"
