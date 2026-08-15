"""Tests de la Fase 6: algoritmos, rejillas y ajuste por Grid Search (PRD §12.2).

Cubren: que los 5 algoritmos exigidos están presentes y correctamente
tipados, que sus rejillas se leen íntegras de `params.yaml` (ninguna
literal en `src/models.py`), que la validación cruzada de ajuste es
`StratifiedKFold` y no `KFold`, que la métrica de optimización no es
`accuracy` (F6-R6, ver ADR-0012), que los 5 modelos ajustados superan el
suelo nulo en la métrica que de verdad importa aquí -- `balanced_accuracy`
-- y el hallazgo explícito de que, precisamente por eso, su F1 queda por
DEBAJO del suelo nulo de F1 (0.83): es el costo esperado de dejar de premiar
al clasificador que ignora a los pacientes sanos. Y que ninguna función de
esta fase referencia el conjunto de prueba.
"""

import copy
import re

import pytest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import PROJECT_ROOT, TARGET_COL, load_params
from src.data import load_modeling_data
from src.models import (
    MODEL_KEYS,
    build_cv,
    build_estimator,
    build_search_grid,
    fit_grid_search,
    null_classifier_floor,
)
from src.splitting import load_split_indices

_ESTIMATOR_TYPES = {
    "logistic_regression": LogisticRegression,
    "knn": KNeighborsClassifier,
    "gaussian_nb": GaussianNB,
    "decision_tree": DecisionTreeClassifier,
    "svm": SVC,
}

FORBIDDEN_TEST_REFERENCES = re.compile(r"\b(test_index|test_indices|X_test|y_test)\b")


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def train_test(params):
    df = load_modeling_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    split = load_split_indices(PROJECT_ROOT / params["split"]["indices_path"])
    train_idx = split["train_index"]
    return X.loc[train_idx], y.loc[train_idx]


@pytest.fixture(scope="module")
def searches(params, train_test):
    """Ajusta los 5 GridSearchCV una sola vez (reutilizado por varios tests)."""
    X_train, y_train = train_test
    return {
        name: fit_grid_search(name, X_train, y_train, params, selector=SelectKBest(f_classif))
        for name in MODEL_KEYS
    }


def test_cinco_algoritmos(params):
    assert set(MODEL_KEYS) == set(_ESTIMATOR_TYPES.keys())
    for name, expected_type in _ESTIMATOR_TYPES.items():
        assert isinstance(build_estimator(name, params), expected_type)


def test_rejillas_desde_params(params):
    for name in MODEL_KEYS:
        grid = build_search_grid(name, params, include_selector=False)
        esperado = {f"estimator__{hp}": vals for hp, vals in params["grids"][name].items()}
        assert grid == esperado

    # Prueba de que build_search_grid() LEE params.yaml y no repite una
    # rejilla propia: mutar una copia de params cambia el resultado.
    mutado = copy.deepcopy(params)
    mutado["grids"]["knn"]["n_neighbors"] = [99]
    assert build_search_grid("knn", mutado, include_selector=False)["estimator__n_neighbors"] == [99]

    # Ninguna de las rejillas de params.yaml está copiada literal en models.py.
    src_text = (PROJECT_ROOT / "src" / "models.py").read_text(encoding="utf-8")
    for name in MODEL_KEYS:
        for values in params["grids"][name].values():
            literal = str(values)
            assert literal not in src_text, f"Rejilla de {name} hardcodeada en models.py: {literal}"


def test_cv_estratificada(params):
    cv = build_cv(params)
    assert isinstance(cv, StratifiedKFold)
    assert not isinstance(cv, KFold)
    assert cv.n_splits == params["cv"]["tuning"]["n_splits"]
    assert cv.random_state == params["seed"]


def test_scoring_no_es_accuracy(params):
    assert params["metrics"]["optimize_for"] != "accuracy"
    assert params["metrics"]["optimize_for"] == "balanced_accuracy"


def test_scoring_supera_el_suelo_nulo(params, train_test, searches):
    """El modelo ajustado bate el suelo nulo en su métrica de selección -- y por qué NO en F1.

    `balanced_accuracy` es la métrica de selección (ADR-0012) precisamente
    porque, a diferencia de F1 con la clase positiva mayoritaria, su suelo
    nulo (0.50) no lo satura un clasificador que no aprendió nada. Los 5
    modelos lo superan.

    F1 se sigue reportando (Trampa 1c), pero NO se exige que supere su
    propio suelo nulo (0.83): ese suelo lo alcanza el clasificador que
    predice "todos enfermos" y tiene recall perfecto sobre la clase
    mayoritaria a costa de no detectar jamás a un paciente sano. Optimizar
    por `balanced_accuracy` empuja al modelo a también acertar en la clase
    sana, lo que baja el recall de la clase enferma y con él el F1 -- es el
    costo esperado y deseado de dejar de premiar al modelo degenerado, no
    una regresión. Se deja fijado aquí para que no se lea como un error en
    una sesión futura.
    """
    _X_train, y_train = train_test
    floor = null_classifier_floor(y_train)
    cv = build_cv(params)

    for name, search in searches.items():
        assert search.best_score_ > floor["balanced_accuracy"], (
            f"{name}: balanced_accuracy de CV ({search.best_score_:.4f}) no supera "
            f"el suelo nulo ({floor['balanced_accuracy']:.4f})"
        )

        scores = cross_validate(
            search.best_estimator_,
            _X_train,
            y_train,
            cv=cv,
            scoring=["f1", "balanced_accuracy"],
        )
        f1_cv = scores["test_f1"].mean()
        assert f1_cv < floor["f1"], (
            f"{name}: F1 de CV ({f1_cv:.4f}) superó el suelo nulo ({floor['f1']:.4f}) -- "
            "revisar si sigue siendo cierto que balanced_accuracy penaliza F1 aquí; "
            "si el hallazgo cambió, actualizar este test y el notebook."
        )


def test_test_no_tocado():
    """Ninguna función de la Fase 6/6A referencia el conjunto de prueba en su código."""
    for filename in ["models.py", "pipelines.py"]:
        src_text = (PROJECT_ROOT / "src" / filename).read_text(encoding="utf-8")
        match = FORBIDDEN_TEST_REFERENCES.search(src_text)
        assert match is None, f"{filename} referencia el conjunto de prueba: {match.group(0)!r}"
