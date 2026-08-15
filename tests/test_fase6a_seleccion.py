"""Tests de la Fase 6A: selección de variables dentro del `Pipeline` (PRD §12.2).

Cubren, con datos reales del split congelado en la Fase 4: que el selector
de variables es un paso del `Pipeline` (nunca aplicado a `X` completo antes
de la validación cruzada -- Ambroise & McLachlan 2002, citados en §20 del
PRD), que `selector=None` deja el `Pipeline` idéntico al de la Fase 5, y que
ambas variantes de `Gender` (F6A-R4) son construibles y se diferencian
exactamente en esa columna.
"""

import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from src.config import PROJECT_ROOT, TARGET_COL, load_params
from src.data import load_modeling_data
from src.pipelines import build_pipeline
from src.splitting import load_split_indices


def _placeholder_estimator():
    return DummyClassifier(strategy="most_frequent")


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def df_modeling():
    return load_modeling_data()


@pytest.fixture(scope="module")
def split(params):
    return load_split_indices(PROJECT_ROOT / params["split"]["indices_path"])


@pytest.fixture(scope="module")
def train_test(df_modeling, split):
    X = df_modeling.drop(columns=[TARGET_COL])
    y = df_modeling[TARGET_COL]
    train_idx, test_idx = split["train_index"], split["test_index"]
    return X.loc[train_idx], y.loc[train_idx], X.loc[test_idx], y.loc[test_idx]


def test_seleccion_dentro_del_pipeline(params, train_test):
    """F6A-R2: el selector vive DENTRO del Pipeline, no se aplica antes a X completo."""
    X_train, y_train, _, _ = train_test
    selector = SelectKBest(f_classif, k=5)
    pipe = build_pipeline("zscore", True, _placeholder_estimator(), params, selector=selector)

    assert [name for name, _ in pipe.steps] == ["preprocess", "selector", "smote", "estimator"]

    # Construir el Pipeline no ajusta nada: el selector todavía no vio
    # ninguna etiqueta. Si esto fallara, significaría que build_pipeline()
    # está ajustando el selector por su cuenta en vez de dejarlo para
    # GridSearchCV/cross_val_score, que es exactamente la fuga que F6A-R2
    # prohíbe.
    assert not hasattr(pipe.named_steps["selector"], "scores_")

    pipe.fit(X_train, y_train)
    assert hasattr(pipe.named_steps["selector"], "scores_")

    Xt = pipe.named_steps["preprocess"].transform(X_train)
    Xsel = pipe.named_steps["selector"].transform(Xt)
    assert Xsel.shape[1] == 5

    # El selector se reajusta en cada pliegue (no es un objeto compartido ya
    # ajustado una vez sobre todo train): clonar el Pipeline y ajustarlo
    # sobre dos mitades distintas de train produce selectores con estado
    # propio, no el mismo objeto reutilizado.
    mitad = len(X_train) // 2
    pipe_a = clone(pipe)
    pipe_b = clone(pipe)
    pipe_a.fit(X_train.iloc[:mitad], y_train.iloc[:mitad])
    pipe_b.fit(X_train.iloc[mitad:], y_train.iloc[mitad:])
    assert pipe_a.named_steps["selector"] is not pipe_b.named_steps["selector"]
    assert pipe_a.named_steps["selector"].scores_.tolist() != pipe_b.named_steps["selector"].scores_.tolist()


def test_selector_none_equivale_a_fase5(params):
    """Sin selector, el Pipeline debe ser idéntico al de la Fase 5 (mismos pasos)."""
    for scaler in params["preprocessing"]["scalers"]:
        for use_smote in [True, False]:
            con_default = build_pipeline(scaler, use_smote, _placeholder_estimator(), params)
            con_none_explicito = build_pipeline(
                scaler, use_smote, _placeholder_estimator(), params, selector=None
            )
            pasos_esperados = ["preprocess", "smote", "estimator"] if use_smote else ["preprocess", "estimator"]
            assert [n for n, _ in con_default.steps] == pasos_esperados
            assert [n for n, _ in con_none_explicito.steps] == pasos_esperados
            assert "selector" not in con_default.named_steps
            assert "selector" not in con_none_explicito.named_steps


def test_ambas_variantes_de_gender(params, train_test):
    """F6A-R4: ambas variantes (con/sin Gender) son construibles y difieren solo en esa columna."""
    assert params["feature_selection"]["include_gender"] == [True, False]

    X_train, _y_train, _, _ = train_test
    pre_con = build_pipeline("zscore", False, _placeholder_estimator(), params).named_steps["preprocess"]
    pre_sin = clone(pre_con)

    Xt_con = pre_con.fit_transform(X_train)
    Xt_sin = pre_sin.fit_transform(X_train.drop(columns=["Gender"]))

    assert any(c.startswith("remainder__Gender") for c in Xt_con.columns)
    assert not any(c.startswith("remainder__Gender") for c in Xt_sin.columns)
    assert Xt_con.shape[1] == Xt_sin.shape[1] + 1
    assert Xt_con.shape[0] == Xt_sin.shape[0] == len(X_train)
