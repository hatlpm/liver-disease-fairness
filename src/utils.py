"""Funciones auxiliares para carga de datos y generación de gráficos."""

import pandas as pd

from src.config import DPI, FIGURES_DIR, RAW_DATA_PATH


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
