"""Tests de la Fase 5: pipeline de preprocesamiento y balanceo (PRD §12.2).

Cubren, con datos reales del split congelado en la Fase 4 (nunca
recalculado aquí): que el `Pipeline` es de `imblearn` y no de `sklearn`, el
orden de sus pasos, que `SMOTE` nunca toca el test, que el imputador usa
mediana con un indicador de ancho fijo sin escalar, que el escalador se
ajusta solo con train, que ese ancho fijo se sostiene entre los 5 pliegues
de la validación cruzada, y (F5-R8) que los pacientes sintéticos de SMOTE
subrepresentan a las mujeres frente a la clase minoritaria que replican.
"""

import imblearn.pipeline
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import MissingIndicator
from sklearn.model_selection import StratifiedKFold

from src.config import NUMERIC_COLS, PROJECT_ROOT, TARGET_COL, load_params
from src.data import load_modeling_data
from src.pipelines import build_pipeline
from src.splitting import load_split_indices

NULL_ROW_INDICES = [246, 261, 279]


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


@pytest.mark.parametrize("scaler", ["minmax", "zscore"])
@pytest.mark.parametrize("use_smote", [True, False])
def test_pipeline_es_imblearn(scaler, use_smote, params):
    pipe = build_pipeline(scaler, use_smote, _placeholder_estimator(), params)
    assert isinstance(pipe, imblearn.pipeline.Pipeline)


def test_orden_de_pasos(params):
    con_smote = build_pipeline("minmax", True, _placeholder_estimator(), params)
    assert [name for name, _ in con_smote.steps] == ["preprocess", "smote", "estimator"]

    sin_smote = build_pipeline("minmax", False, _placeholder_estimator(), params)
    assert [name for name, _ in sin_smote.steps] == ["preprocess", "estimator"]


def test_smote_no_toca_el_test(params, train_test):
    X_train, y_train, X_test, _y_test = train_test
    pipe = build_pipeline("zscore", True, _placeholder_estimator(), params)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    assert len(preds) == len(X_test) == 114

    # El paso "preprocess" ya quedó ajustado dentro de pipe.fit(); acceder a
    # él por nombre no depende de que el slicing del Pipeline esté soportado
    # (SMOTE no implementa .transform(), así que recortar con pipe[:-1]
    # dejaría a SMOTE como último paso y fallaría).
    X_test_pre = pipe.named_steps["preprocess"].transform(X_test)
    assert X_test_pre.shape[0] == len(X_test) == 114


def test_imputador_mediana_con_indicador(params, train_test):
    X_train, y_train, _, _ = train_test
    pipe = build_pipeline("zscore", False, _placeholder_estimator(), params)
    pipe.fit(X_train, y_train)
    pre = pipe.named_steps["preprocess"]

    imputer = pre.named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.strategy == "median"

    indicator = pre.named_transformers_["tb_db_indicator"]
    assert isinstance(indicator, MissingIndicator)
    assert indicator.features == "all"
    assert indicator.transform(X_train[["TB", "DB"]]).shape[1] == 2


def test_indicador_no_escalado(params, train_test):
    X_train, _y_train, _, _ = train_test
    pre = build_pipeline("zscore", False, _placeholder_estimator(), params).named_steps["preprocess"]
    X_t = pre.fit_transform(X_train)

    indicator_cols = [c for c in X_t.columns if c.startswith("tb_db_indicator__")]
    assert len(indicator_cols) == 2
    valores = np.unique(X_t[indicator_cols].to_numpy())
    assert set(valores).issubset({0, 1})


@pytest.mark.parametrize("scaler_name", ["minmax", "zscore"])
def test_sin_fuga_de_escalado(scaler_name, params, train_test):
    X_train, y_train, _, _ = train_test
    pipe = build_pipeline(scaler_name, False, _placeholder_estimator(), params)
    pipe.fit(X_train, y_train)
    fitted_scaler = pipe.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]

    # Mediana SOLO de train, igual que el imputador dentro del pipeline.
    ref = X_train[NUMERIC_COLS].fillna(X_train[NUMERIC_COLS].median())

    if scaler_name == "zscore":
        assert np.allclose(fitted_scaler.mean_, ref.mean().to_numpy())
        assert np.allclose(fitted_scaler.scale_, ref.std(ddof=0).to_numpy())
    else:
        assert np.allclose(fitted_scaler.data_min_, ref.min().to_numpy())
        assert np.allclose(fitted_scaler.data_max_, ref.max().to_numpy())


def test_ancho_estable_entre_pliegues(params, train_test):
    X_train, y_train, _, _ = train_test
    template = build_pipeline("zscore", False, _placeholder_estimator(), params).named_steps["preprocess"]

    skf = StratifiedKFold(
        n_splits=params["cv"]["tuning"]["n_splits"],
        shuffle=params["cv"]["tuning"]["shuffle"],
        random_state=params["seed"],
    )
    anchos = []
    for train_fold_idx, _ in skf.split(X_train, y_train):
        pre = clone(template)
        X_fold = X_train.iloc[train_fold_idx]
        anchos.append(pre.fit_transform(X_fold).shape[1])
    assert len(set(anchos)) == 1

    # Chequeo dirigido: el ancho no cambia ni siquiera si el pliegue de
    # entrenamiento no contiene NINGUNA de las 3 filas con TB/DB nulos --
    # el escenario exacto que MissingIndicator(features="all") existe para
    # blindar, y que RepeatedStratifiedKFold (Fase 9, 50 ajustes) puede
    # producir aunque hoy, con 5 pliegues, no ocurra.
    presentes = [i for i in NULL_ROW_INDICES if i in X_train.index]
    assert len(presentes) == 3  # las 3 filas nulas deben estar en train (Fase 4)
    ancho_sin_nulos = clone(template).fit_transform(X_train.drop(index=presentes)).shape[1]
    assert ancho_sin_nulos == anchos[0]


def test_smote_subrepresenta_mujeres_en_sinteticos(params, train_test):
    """F5-R8: los sintéticos de SMOTE tienen menos mujeres que la minoría que replican.

    Con `sampling_strategy='auto'`, SMOTE solo sintetiza la clase
    minoritaria (`Selector=0`, sanos). La comparación correcta no es train
    completo antes/después (se diluye entre las 456 filas reales que no
    cambian) sino minoría original vs. filas sintéticas que la replican.
    Medido con random_state=42: 29.77% de mujeres en la minoría de train
    frente a 23.71% entre los 194 sintéticos -- una brecha de ~6 pp que no
    depende de ninguna decisión explícita del pipeline, solo de la mecánica
    de SMOTE. Deja fijado el hallazgo para que no se pierda en un refactor
    futuro de `build_pipeline()`.
    """
    X_train, y_train, _, _ = train_test
    pipe = build_pipeline("zscore", True, _placeholder_estimator(), params)
    pipe.fit(X_train, y_train)

    Xt_train = pipe.named_steps["preprocess"].transform(X_train)
    X_res, _y_res = pipe.named_steps["smote"].fit_resample(Xt_train, y_train)
    gender_col = next(c for c in X_res.columns if c.startswith("remainder__Gender"))
    sinteticas = X_res.iloc[len(X_train) :]

    minoria_train = X_train.loc[y_train == 0]
    tasa_mujeres_minoria = (minoria_train["Gender"] == 1).mean()
    tasa_mujeres_sinteticas = (sinteticas[gender_col] == 1).mean()

    assert tasa_mujeres_sinteticas < tasa_mujeres_minoria
    assert round(tasa_mujeres_minoria, 4) == 0.2977
    assert round(tasa_mujeres_sinteticas, 4) == 0.2371
