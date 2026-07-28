"""Funciones auxiliares para carga de datos, gráficos y chequeos clínicos."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from src.config import (
    AGE_BANDS,
    ALT_ULN_BY_SEX,
    ALT_ULN_UNISEX,
    DPI,
    FIGURES_DIR,
    MIN_STRATUM_SIZE_FOR_SEX_SPLIT,
    RAW_DATA_PATH,
    WHIPPLE_AGE_RANGE,
    WHIPPLE_DIGIT_STEP,
)


def load_raw_data() -> pd.DataFrame:
    """Carga el dataset crudo de enfermedad hepática (ILPD).

    Returns
    -------
    pandas.DataFrame
        DataFrame con 583 filas y 11 columnas, tal como viene el archivo
        fuente en ``data/raw/act_liver_disease.csv``, sin ninguna
        transformación (ni imputación, ni escalado, ni eliminación de
        duplicados).
    """
    return pd.read_csv(RAW_DATA_PATH)


def save_figure(fig, filename: str) -> None:
    """Exporta una figura de matplotlib a ``reports/figures/`` en PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura a exportar.
    filename : str
        Nombre de archivo, sin ruta (por ejemplo ``"t4_hist_bilirubin.png"``).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / filename, dpi=DPI, bbox_inches="tight")


def whipple_index(ages: pd.Series) -> float:
    """Calcula el índice de Whipple para cuantificar el *age heaping*.

    El *age heaping* es la acumulación artificial de edades terminadas en 0 o
    5, señal de que la edad se estimó o se reportó de memoria en vez de
    leerse de un documento. El índice compara cuánta gente cae realmente en
    las edades terminadas en 0/5 dentro del rango 23–62 contra cuánta debería
    caer si no hubiera redondeo (una de cada cinco).

    Escala de Naciones Unidas: <105 muy exacta · 105–110 bastante exacta ·
    110–125 aproximada · 125–175 tosca · >175 muy tosca. El valor 100 indica
    ausencia total de preferencia de dígito; 500 es el máximo teórico.

    Parameters
    ----------
    ages : pandas.Series
        Serie de edades enteras.

    Returns
    -------
    float
        Índice de Whipple. 100 = sin *heaping*.

    References
    ----------
    United Nations, Demographic Yearbook — notas metodológicas.
    Ver ``docs/fuentes/Consulta_4.md``.
    """
    low, high = WHIPPLE_AGE_RANGE
    in_range = ages[(ages >= low) & (ages <= high)]
    on_round = in_range[in_range % WHIPPLE_DIGIT_STEP == 0]
    expected = len(in_range) / WHIPPLE_DIGIT_STEP
    return len(on_round) / expected * 100


def whipple_label(index: float) -> str:
    """Traduce un índice de Whipple a su categoría de calidad de la ONU.

    Parameters
    ----------
    index : float
        Valor devuelto por :func:`whipple_index`.

    Returns
    -------
    str
        Categoría cualitativa ("Muy exacta", "Tosca", …).
    """
    for limit, label in (
        (105, "Muy exacta"),
        (110, "Bastante exacta"),
        (125, "Aproximada"),
        (175, "Tosca"),
    ):
        if index < limit:
            return label
    return "Muy tosca"


def de_ritis_ratio(df: pd.DataFrame) -> pd.Series:
    """Calcula el cociente De Ritis (AST/ALT) fila a fila.

    El cociente contrasta dos enzimas que se derraman a la sangre cuando el
    hepatocito se rompe. Se usa como indicio **sugerente** del patrón de daño
    — nunca como prueba etiológica: el ratio depende también del estadio de la
    enfermedad y del método analítico del laboratorio.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con las columnas ``Sgot`` (AST) y ``Sgpt`` (ALT).

    Returns
    -------
    pandas.Series
        Cociente AST/ALT. ``NaN`` donde ``Sgpt`` es cero (división indefinida).

    References
    ----------
    Botros & Sikaris (2013), *The De Ritis Ratio: The Test of Time*.
    Ver ``docs/fuentes/Consulta_3.md``.
    """
    return df["Sgot"] / df["Sgpt"].where(df["Sgpt"] > 0)


def flag_biochemical_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Marca filas que violan restricciones bioquímicas duras.

    Son restricciones de contención, no rangos de referencia: se cumplen por
    definición y solo pueden romperse por error de medición o de captura.

    - ``DB <= TB``: la bilirrubina directa es una fracción de la total.
    - ``ALB <= TP``: la albúmina está contenida en la proteína total.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame crudo del ILPD.

    Returns
    -------
    pandas.DataFrame
        Mismo índice que ``df``, con una columna booleana por regla y la
        columna ``any_violation``.

    References
    ----------
    Kahn et al. (2016), marco de calidad de datos adoptado por OHDSI:
    violación de *plausibilidad atemporal*. Ver ``docs/fuentes/Consulta_4.md``.
    """
    flags = pd.DataFrame(index=df.index)
    flags["db_gt_tb"] = df["DB"] > df["TB"]
    flags["alb_gt_tp"] = df["ALB"] > df["TP"]
    flags["any_violation"] = flags.any(axis=1)
    return flags


def alt_threshold_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compara la clasificación de ALT bajo umbral unisex vs sexo-específico.

    Responde a la pregunta: ¿cuántas personas cambian de "normal" a "elevado"
    al pasar del punto de corte único (que usaban los laboratorios de la época)
    a los puntos de corte diferenciados por sexo de Prati/ACG/AASLD?

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con las columnas ``Gender`` y ``Sgpt``.

    Returns
    -------
    pandas.DataFrame
        Una fila por sexo con los conteos y porcentajes bajo ambos criterios y
        la diferencia en puntos porcentuales.

    References
    ----------
    Prati et al. (2002), Ann Intern Med; ACG 2017; AASLD 2023.
    Ver ``docs/fuentes/Consulta_1.md``.
    """
    rows = []
    for sex, group in df.groupby("Gender"):
        uln_sex = ALT_ULN_BY_SEX[sex]
        n = len(group)
        n_unisex = int((group["Sgpt"] > ALT_ULN_UNISEX).sum())
        n_by_sex = int((group["Sgpt"] > uln_sex).sum())
        rows.append(
            {
                "Gender": sex,
                "n": n,
                "ULN_sexo": uln_sex,
                "anormales_unisex": n_unisex,
                "pct_unisex": round(100 * n_unisex / n, 1),
                "anormales_por_sexo": n_by_sex,
                "pct_por_sexo": round(100 * n_by_sex / n, 1),
                "reclasificados": n_by_sex - n_unisex,
                "delta_pp": round(100 * (n_by_sex - n_unisex) / n, 1),
            }
        )
    return pd.DataFrame(rows).set_index("Gender")


def age_band_label(age: int) -> str:
    """Traduce una edad a su etiqueta de banda según ``AGE_BANDS``.

    Parameters
    ----------
    age : int
        Edad en años.

    Returns
    -------
    str
        Etiqueta ``"lo-hi"`` de la banda que contiene ``age``.
    """
    for lo, hi in AGE_BANDS:
        if lo <= age <= hi:
            return f"{lo}-{hi}"
    raise ValueError(f"Edad {age} fuera de todas las bandas de AGE_BANDS")


def assign_age_sex_stratum(df: pd.DataFrame) -> pd.Series:
    """Asigna cada fila a un estrato edad × sexo para el clustering de Fase 2c.

    Reutiliza las bandas de ``AGE_BANDS``. Una banda se divide por sexo si
    **al menos una** de las dos submuestras alcanza
    ``MIN_STRATUM_SIZE_FOR_SEX_SPLIT`` (el sexo grande se analiza aparte, el
    chico simplemente hereda una muestra menor y se reporta con ese
    caveat). Solo si **ambas** submuestras quedan por debajo del umbral —
    caso de la banda pediátrica, 0-17, n=25: 7 mujeres/18 hombres — se
    colapsa la banda entera en un solo grupo mixto, porque ahí ni siquiera
    el sexo mayoritario alcanza una muestra mínima para un dendrograma
    estable.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas ``Age`` y ``Gender``.

    Returns
    -------
    pandas.Series
        Etiqueta de estrato por fila, p. ej. ``"18-39 · Female"`` o
        ``"0-17 · ambos sexos"``.
    """
    band = df["Age"].apply(age_band_label)
    counts = pd.crosstab(band, df["Gender"])
    splittable_bands = counts[counts.max(axis=1) >= MIN_STRATUM_SIZE_FOR_SEX_SPLIT].index

    def label(row_band: str, row_gender: str) -> str:
        if row_band in splittable_bands:
            return f"{row_band} · {row_gender}"
        return f"{row_band} · ambos sexos"

    return pd.Series(
        [label(b, g) for b, g in zip(band, df["Gender"])],
        index=df.index,
        name="Stratum",
    )


def hierarchical_cluster_cut(
    X: np.ndarray, method: str = "ward"
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Corta un dendrograma jerárquico en el mayor salto de distancia de fusión.

    Heurística reproducible para elegir el número de clusters sin fijarlo a
    mano: ordena las distancias de fusión del *linkage* y corta justo antes
    del salto más grande entre fusiones consecutivas — el equivalente del
    método del "codo" aplicado al propio dendrograma, en vez de a una curva
    de inercia.

    Parameters
    ----------
    X : numpy.ndarray
        Variables estandarizadas (media 0, varianza 1); el corte por
        distancia solo tiene sentido si todas las variables están en la
        misma escala.
    method : str
        Método de enlace de ``scipy.cluster.hierarchy.linkage``.

    Returns
    -------
    tuple
        ``(Z, labels, cut_distance, k)`` — la matriz de *linkage*, las
        etiquetas de cluster por fila (1-indexadas), la distancia de corte
        elegida y el número de clusters resultante.
    """
    Z = linkage(X, method=method)
    merge_distances = Z[:, 2]
    gaps = np.diff(merge_distances)
    cut_idx = int(np.argmax(gaps))
    cut_distance = merge_distances[cut_idx] + gaps[cut_idx] / 2
    labels = fcluster(Z, t=cut_distance, criterion="distance")
    k = len(np.unique(labels))
    return Z, labels, cut_distance, k
