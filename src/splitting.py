"""Split train/test estratificado y persistencia de sus índices.

El split se calcula una sola vez sobre el dataset de modelado de
:func:`src.data.load_modeling_data` y se congela: toda fase posterior debe
cargar los mismos índices en vez de recalcular el split (F4-R9), para que
train/test nunca cambien entre sesiones o entre agentes.

La variante oficial (F4-R8) estratifica por la clave compuesta
``Selector x Gender``, no solo por ``Selector``: reparte mejor el subgrupo
crítico del proyecto (mujeres sanas) entre train y test. La variante simple
(solo ``Selector``) se mantiene aquí únicamente como término de comparación
narrado en ``notebooks/04_split.ipynb`` -- no es una configuración
intercambiable del experimento y por eso no vive en ``params.yaml``.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

_COMPARISON_ONLY_STRATIFY_COLUMNS = ["Selector"]


def build_stratify_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Concatena una o más columnas en una única clave de estratificación.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame de origen.
    columns : list of str
        Columnas a combinar, en el orden en que se concatenan.

    Returns
    -------
    pandas.Series
        Clave de texto, una por fila, con el mismo índice que ``df``.
    """
    return df[columns].astype(str).agg("_".join, axis=1)


def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratify_columns: list[str],
) -> tuple[pd.Index, pd.Index]:
    """Divide ``df`` en train/test, estratificado por ``stratify_columns``.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataset de modelado ya deduplicado (una fila = un paciente).
    test_size : float
        Proporción del conjunto de test, en (0, 1).
    random_state : int
        Semilla de ``train_test_split``.
    stratify_columns : list of str
        Columnas de ``df`` a combinar para la estratificación. Con una sola
        columna, equivale a estratificar por esa columna directamente.

    Returns
    -------
    tuple of pandas.Index
        ``(train_index, test_index)``, ordenados, sobre el índice original
        de ``df`` (no se resetea).
    """
    key = build_stratify_key(df, stratify_columns)
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=random_state, stratify=key
    )
    return pd.Index(sorted(train_idx)), pd.Index(sorted(test_idx))


def split_train_test(df: pd.DataFrame, params: dict) -> tuple[pd.Index, pd.Index]:
    """Split oficial de la Fase 4 (F4-R8): estratifica por ``Selector x Gender``.

    Lee ``seed``, ``split.test_size`` y ``split.stratify_by`` de
    ``params.yaml`` -- ningún número queda literal en este módulo.

    Parameters
    ----------
    df : pandas.DataFrame
        Salida de :func:`src.data.load_modeling_data`.
    params : dict
        Árbol de parámetros devuelto por :func:`src.config.load_params`.

    Returns
    -------
    tuple of pandas.Index
        ``(train_index, test_index)``.
    """
    return stratified_split(
        df,
        test_size=params["split"]["test_size"],
        random_state=params["seed"],
        stratify_columns=params["split"]["stratify_by"],
    )


def split_selector_only_for_comparison(
    df: pd.DataFrame, params: dict
) -> tuple[pd.Index, pd.Index]:
    """Split de comparación (F4-R6 tal cual): estratifica solo por ``Selector``.

    Existe únicamente para contrastar, en ``notebooks/04_split.ipynb``, el
    efecto de estratificar por la clave compuesta frente a estratificar solo
    por la clase. No es la variante oficial (esa es
    :func:`split_train_test`) y no se usa en ninguna fase posterior.

    Parameters
    ----------
    df : pandas.DataFrame
        Salida de :func:`src.data.load_modeling_data`.
    params : dict
        Árbol de parámetros devuelto por :func:`src.config.load_params`.

    Returns
    -------
    tuple of pandas.Index
        ``(train_index, test_index)``.
    """
    return stratified_split(
        df,
        test_size=params["split"]["test_size"],
        random_state=params["seed"],
        stratify_columns=_COMPARISON_ONLY_STRATIFY_COLUMNS,
    )


def save_split_indices(
    train_index: pd.Index,
    test_index: pd.Index,
    path: Path,
    *,
    seed: int,
    test_size: float,
    stratify_by: list[str],
) -> None:
    """Persiste los índices de train/test en JSON, con metadata de reproducibilidad.

    Parameters
    ----------
    train_index, test_index : pandas.Index
        Índices devueltos por :func:`split_train_test`.
    path : pathlib.Path
        Ruta de salida, típicamente ``data/processed/split_indices.json``.
    seed : int
        Semilla usada para el split.
    test_size : float
        Proporción de test usada.
    stratify_by : list of str
        Columnas usadas para la estratificación oficial.
    """
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "test_size": test_size,
        "stratify_by": stratify_by,
        "n_total": len(train_index) + len(test_index),
        "n_train": len(train_index),
        "n_test": len(test_index),
        "train_indices": [int(i) for i in train_index],
        "test_indices": [int(i) for i in test_index],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split_indices(path: Path) -> dict:
    """Carga el JSON de :func:`save_split_indices` y reconstruye los índices.

    Parameters
    ----------
    path : pathlib.Path
        Ruta al archivo generado por :func:`save_split_indices`.

    Returns
    -------
    dict
        El payload persistido, con ``train_indices``/``test_indices``
        adicionalmente disponibles como ``pandas.Index`` bajo las claves
        ``train_index``/``test_index``.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    payload["train_index"] = pd.Index(payload["train_indices"])
    payload["test_index"] = pd.Index(payload["test_indices"])
    return payload
