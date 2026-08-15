"""Contrato de datos de la Actividad 2: dataset de modelado desde el crudo.

Cadena de transformaciones deterministas que convierte el CSV crudo
(583 x 11) en el dataset que entra al split de la Fase 4 (570 x 11, con los
6 nulos de ``TB``/``DB`` que la Fase 5 imputará dentro del ``Pipeline``).
Cada paso es una función pura, testeable de forma aislada; ``load_modeling_data``
solo encadena los pasos en el orden exigido por el PRD (F4-R1..R5).

Orden crítico para la reproducibilidad bit a bit del split (Fase 4, Trampa 3):
deduplicar -> reconstruir ``A/G Ratio`` -> marcar ``TB``/``DB`` -> codificar
``Gender``/``Selector``. La codificación va ANTES del split porque
``train_test_split(stratify=...)`` itera según el orden de las clases únicas
del array de estratificación, y ese orden cambia con la codificación.
"""

import pandas as pd

from src.utils import flag_biochemical_violations, load_raw_data

GENDER_MAP = {"Male": 0, "Female": 1}
SELECTOR_MAP = {1: 1, 2: 0}  # F4-R5 / A4: Selector==1 (enfermo) es la clase positiva


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas exactamente duplicadas, conservando la primera aparición.

    Decisión F4-R2 / §5.3 del PRD: diverge deliberadamente de la Actividad 1
    (que conservaba los 13 pares porque no se entrenaba nada). Aquí un par
    idéntico repartido entre train y test es memorización garantizada, no
    generalización. Se conserva la etiqueta de índice original de ``df`` --
    no se resetea -- para que las filas sigan siendo identificables por su
    posición en ``data/raw/`` en cualquier fase posterior.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame crudo o parcialmente transformado.

    Returns
    -------
    pandas.DataFrame
        Mismas columnas, sin duplicados exactos, índice original conservado.
    """
    return df.drop_duplicates(keep="first")


def reconstruct_ag_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye los nulos de ``A/G Ratio`` con ``ALB / (TP - ALB)``.

    Operación fila a fila -- no usa ninguna estadística del conjunto -- por
    lo que no es fuga de datos y puede ir antes del split (F4-R3 / §5.1 del
    PRD). Es 4.9x más precisa que imputar por la media (error absoluto
    mediano 0.031 frente a 0.153, medido sobre las filas con valor
    presente). Solo toca las filas donde ``A/G Ratio`` es nulo.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con las columnas ``TP``, ``ALB`` y ``A/G Ratio``.

    Returns
    -------
    pandas.DataFrame
        Copia de ``df`` con ``A/G Ratio`` reconstruido donde era nulo.
    """
    df = df.copy()
    missing = df["A/G Ratio"].isna()
    df.loc[missing, "A/G Ratio"] = df.loc[missing, "ALB"] / (
        df.loc[missing, "TP"] - df.loc[missing, "ALB"]
    )
    return df


def nullify_biochemical_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Marca ``TB`` y ``DB`` como ``NaN`` en las filas donde ``DB > TB``.

    ``DB`` (bilirrubina directa) es por definición una fracción de ``TB``
    (bilirrubina total); ``DB > TB`` es bioquímicamente imposible y se trata
    como error de medición/captura, no como dato válido (F4-R4, decisión de
    la Fase 3 de la Actividad 1, re-aplicada aquí porque el CSV crudo no
    trae estos valores marcados). Reutiliza
    :func:`src.utils.flag_biochemical_violations` en vez de reimplementar la
    regla. Los ``NaN`` resultantes se dejan sin imputar: la imputación por
    mediana con indicador va dentro del ``Pipeline`` de la Fase 5, no aquí,
    porque usa una estadística de grupo (fuga si se calculara sobre el
    dataset completo antes del split).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con las columnas ``TB`` y ``DB``.

    Returns
    -------
    pandas.DataFrame
        Copia de ``df`` con ``TB``/``DB`` en ``NaN`` en las filas violatorias.
    """
    df = df.copy()
    violation = flag_biochemical_violations(df)["db_gt_tb"]
    df.loc[violation, ["TB", "DB"]] = None
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica ``Gender`` de texto a entero según :data:`GENDER_MAP`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con la columna ``Gender`` en {"Male", "Female"}.

    Returns
    -------
    pandas.DataFrame
        Copia de ``df`` con ``Gender`` en {0, 1}.
    """
    df = df.copy()
    df["Gender"] = df["Gender"].map(GENDER_MAP)
    return df


def encode_selector(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica ``Selector`` de {1, 2} a {1, 0} según :data:`SELECTOR_MAP`.

    ``Selector == 1`` (enfermo) queda mapeado a la clase positiva ``1``;
    ``Selector == 2`` (sano) queda mapeado a ``0`` (F4-R5, ambigüedad A4 del
    PRD resuelta explícitamente).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con la columna ``Selector`` en {1, 2}.

    Returns
    -------
    pandas.DataFrame
        Copia de ``df`` con ``Selector`` en {0, 1}, 1 = enfermo.
    """
    df = df.copy()
    df["Selector"] = df["Selector"].map(SELECTOR_MAP)
    return df


def row_signature(df: pd.DataFrame) -> pd.Series:
    """Construye una clave de fila completa, tolerante a ``NaN``.

    Se usa para detectar si alguna fila idéntica termina repartida entre
    train y test (el test de fuga de §5.3 del PRD). Dos filas con el mismo
    contenido en todas las columnas producen la misma firma,
    independientemente de su posición en el índice.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame cuyas filas se quieren identificar por contenido.

    Returns
    -------
    pandas.Series
        Una cadena de texto por fila, mismo índice que ``df``.
    """
    # No se usa astype(str) + join directo: en pandas 3.x un NaN sobrevive
    # como float dentro de una columna ya convertida a str, y "|".join
    # revienta con TypeError al toparse con él. map() normaliza cada NaN a
    # un token de texto explícito antes de unir.
    stringified = df.map(lambda v: "NaN" if pd.isna(v) else str(v))
    return stringified.agg("|".join, axis=1)


def load_modeling_data() -> pd.DataFrame:
    """Carga el dataset de modelado de la Fase 4, listo para el split.

    Orquesta, en este orden exacto, F4-R1 a F4-R5: carga el CSV crudo ->
    deduplica -> reconstruye ``A/G Ratio`` -> marca ``TB``/``DB`` violatorios
    como nulos -> codifica ``Gender`` y ``Selector``. El orden es el
    verificado empíricamente contra el split de la Fase 4: codificar antes
    del split es lo que reproduce las proporciones exactas documentadas en
    ``notebooks/04_split.ipynb``.

    Returns
    -------
    pandas.DataFrame
        570 filas x 11 columnas. Índice heredado de ``data/raw/`` (no
        contiguo: tiene huecos donde se quitaron duplicados). ``Gender`` y
        ``Selector`` numéricos. 6 valores ``NaN`` (``TB``/``DB`` de 3 filas),
        dejados sin imputar a propósito para la Fase 5.
    """
    df = load_raw_data()
    df = deduplicate(df)
    df = reconstruct_ag_ratio(df)
    df = nullify_biochemical_violations(df)
    df = encode_gender(df)
    df = encode_selector(df)
    return df
