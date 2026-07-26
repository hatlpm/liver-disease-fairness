"""Constantes globales del proyecto: rutas, semilla y umbrales nombrados."""

from pathlib import Path

# --- Rutas (relativas a la raíz del repo) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RAW_DATA_PATH = DATA_RAW_DIR / "act_liver_disease.csv"

# --- Reproducibilidad ---
RANDOM_STATE = 42

# --- Umbrales nombrados (evitar magic numbers) ---
IQR_MULTIPLIER = 1.5
CORR_THRESHOLD = 0.7
DPI = 300

# --- Columnas ---
NUMERIC_COLS = [
    "Age",
    "TB",
    "DB",
    "Alkphos",
    "Sgpt",
    "Sgot",
    "TP",
    "ALB",
    "A/G Ratio",
]
CATEGORICAL_COLS = ["Gender"]
TARGET_COL = "Selector"
