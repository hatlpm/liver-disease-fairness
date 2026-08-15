"""Tests de la Fase 4: contrato de datos y split (F4-R1..R9, PRD §12.2).

Cubren, con datos reales de ``data/raw/``: deduplicación (570 filas),
reconstrucción determinista de ``A/G Ratio``, marcado de ``TB``/``DB``
bioquímicamente imposibles, ausencia de fuga entre train y test (índices y
filas idénticas), estratificación por clase y por sexo, reproducibilidad del
split, y la convención de clase positiva.
"""

import pandas as pd
import pytest

from src.config import RANDOM_STATE, load_params
from src.data import encode_selector, load_modeling_data, row_signature
from src.splitting import (
    load_split_indices,
    save_split_indices,
    split_train_test,
)

AG_RATIO_RECONSTRUCTED = {209: 1.44, 241: 0.91, 253: 1.08, 312: 1.30}


@pytest.fixture(scope="module")
def df_modeling() -> pd.DataFrame:
    return load_modeling_data()


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def split_result(df_modeling, params) -> tuple[pd.Index, pd.Index]:
    return split_train_test(df_modeling, params)


def test_dataset_deduplicado(df_modeling):
    assert len(df_modeling) == 570


def test_reconstruccion_ag_ratio(df_modeling):
    for idx, expected in AG_RATIO_RECONSTRUCTED.items():
        assert df_modeling.loc[idx, "A/G Ratio"] == pytest.approx(expected, abs=0.005)


def test_tb_db_marcados_como_nulos(df_modeling):
    assert df_modeling[["TB", "DB"]].isna().sum().sum() == 6
    assert df_modeling.drop(columns=["TB", "DB"]).isna().sum().sum() == 0


def test_indices_disjuntos(split_result):
    train_idx, test_idx = split_result
    assert set(train_idx).isdisjoint(set(test_idx))


def test_ninguna_fila_identica_cruza_el_split(df_modeling, split_result):
    train_idx, test_idx = split_result
    signatures = row_signature(df_modeling)
    train_signatures = set(signatures.loc[train_idx])
    test_signatures = set(signatures.loc[test_idx])
    assert train_signatures.isdisjoint(test_signatures)


def test_estratificacion_clase(df_modeling, split_result):
    train_idx, test_idx = split_result
    full_rate = df_modeling["Selector"].mean()
    train_rate = df_modeling.loc[train_idx, "Selector"].mean()
    test_rate = df_modeling.loc[test_idx, "Selector"].mean()
    assert abs(train_rate - full_rate) <= 0.02
    assert abs(test_rate - full_rate) <= 0.02


def test_estratificacion_sexo(df_modeling, split_result):
    train_idx, test_idx = split_result
    full_rate = df_modeling["Gender"].mean()
    train_rate = df_modeling.loc[train_idx, "Gender"].mean()
    test_rate = df_modeling.loc[test_idx, "Gender"].mean()
    assert abs(train_rate - full_rate) <= 0.02
    assert abs(test_rate - full_rate) <= 0.02


def test_split_reproducible(df_modeling, params, split_result, tmp_path):
    train_idx, test_idx = split_result
    train_idx_2, test_idx_2 = split_train_test(df_modeling, params)
    assert list(train_idx) == list(train_idx_2)
    assert list(test_idx) == list(test_idx_2)

    out_path = tmp_path / "split_indices.json"
    save_split_indices(
        train_idx,
        test_idx,
        out_path,
        seed=params["seed"],
        test_size=params["split"]["test_size"],
        stratify_by=params["split"]["stratify_by"],
    )
    reloaded = load_split_indices(out_path)
    assert list(reloaded["train_index"]) == list(train_idx)
    assert list(reloaded["test_index"]) == list(test_idx)
    assert reloaded["seed"] == RANDOM_STATE == params["seed"]


def test_clase_positiva_es_enfermo():
    df = pd.DataFrame({"Selector": [1, 2]})
    result = encode_selector(df)["Selector"].tolist()
    assert result == [1, 0]
