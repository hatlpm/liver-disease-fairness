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

# --- Ejes funcionales hepáticos (agrupación clínica de las variables) ---
# Ver docs/data_dictionary.md para el significado de cada eje.
LIVER_AXES = {
    "dano_celular": ["Sgpt", "Sgot"],
    "excrecion_biliar": ["TB", "DB", "Alkphos"],
    "funcion_sintetica": ["TP", "ALB", "A/G Ratio"],
}

# --- Umbrales de referencia ALT (Sgpt) ---
# Unisex: el que usaban los laboratorios en la época de recolección del ILPD
# (~2007-2012). Sexo-específico: Prati et al. 2002 (Ann Intern Med), adoptado
# por ACG 2017 y AASLD 2023. Ver docs/fuentes/Consulta_1.md.
ALT_ULN_UNISEX = 40
ALT_ULN_BY_SEX = {"Male": 30, "Female": 19}

# --- Rangos de referencia adulto (para chequeos de coherencia clínica) ---
# Usados solo para clasificar "perfil normal / alterado", no para diagnosticar.
ADULT_REFERENCE_RANGES = {
    "TB": (0.3, 1.2),
    "DB": (0.0, 0.3),
    "Alkphos": (40, 129),
    "Sgpt": (0, 40),
    "Sgot": (0, 40),
    "TP": (6.0, 8.3),
    "ALB": (3.5, 5.5),
}

# --- Cociente De Ritis (AST/ALT) ---
# Puntos de corte clásicos. Interpretación SUGERENTE, no diagnóstica:
# el ratio refleja también el estadio de la enfermedad, no solo la etiología.
# Ver docs/fuentes/Consulta_3.md.
DE_RITIS_ALCOHOLIC = 2.0
DE_RITIS_VIRAL = 1.0

# --- Estratificación por edad ---
# CLSI C28-A3 / CALIPER exigen particionar por edad ANTES de detectar outliers.
# Ver docs/fuentes/Consulta_2.md.
AGE_ADULT_MIN = 18
AGE_BANDS = [(0, 17), (18, 39), (40, 59), (60, 120)]

# --- Índice de Whipple (age heaping) ---
# Rango canónico 23-62 y dígitos terminales 0/5. Escala ONU en utils.
WHIPPLE_AGE_RANGE = (23, 62)
WHIPPLE_DIGIT_STEP = 5

# --- Top-coding documentado por UCI ---
# "Any patient whose age exceeded 89 is listed as being of age '90'."
AGE_TOP_CODE = 90

# --- Paleta de figuras ---
# Slots categóricos 1 y 2 de una paleta validada para daltonismo:
# separación CVD ΔE 24.7 (protan) y visión normal ΔE 33.6, ambos muy por
# encima de los umbrales (8 y 15). El color nunca es el único canal:
# toda figura lleva leyenda y etiquetas directas.
SEX_COLORS = {"Female": "#2a78d6", "Male": "#eb6834"}

# Rampa ORDINAL (un solo tono, claro→oscuro) para las franjas de edad, que
# tienen orden natural. Ningún paso queda por debajo de 2:1 de contraste
# contra la superficie clara.
ORDINAL_RAMP_4 = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
AGE_BAND_COLORS = ORDINAL_RAMP_4

# Tinta de la figura (ejes, rejilla, anotaciones). Nunca color de serie.
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"
