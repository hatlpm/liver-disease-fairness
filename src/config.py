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

# --- Fase 2c: clustering jerárquico exploratorio (edad × sexo) ---
# Método de enlace estándar para clustering jerárquico con distancia euclídea
# sobre variables estandarizadas (minimiza varianza intra-cluster).
CLUSTER_LINKAGE_METHOD = "ward"

# Una banda de edad se divide por sexo si al menos uno de los dos grupos
# alcanza este tamaño (regla práctica, no un valor validado formalmente);
# si ambos quedan por debajo, la banda completa se trata como un solo grupo
# mixto en vez de partirla en submuestras demasiado chicas para que un
# dendrograma refleje señal real en vez de ruido. Solo la banda pediátrica
# (0-17, n=25: 7 mujeres/18 hombres) cae en ese caso.
MIN_STRATUM_SIZE_FOR_SEX_SPLIT = 30

# Tamaño mínimo de un estrato para que se le corra clustering. Por debajo de
# esto el dendrograma aísla individuos sueltos en vez de encontrar grupos, y
# cualquier "mediana de cluster" resultante se calcula sobre 1-2 personas.
# Se usa el mismo valor que el umbral de partición por sexo por coherencia.
# Excluye: "0-17 · ambos sexos" (n=25) y "60-120 · Female" (n=20).
MIN_STRATUM_SIZE_FOR_CLUSTERING = 30

# Tamaño mínimo de cada cluster resultante. Sin esta restricción, el corte por
# mayor salto de distancia aísla al paciente más extremo como "cluster" de 1.
# Con 5, cualquier mediana de cluster se calcula sobre suficientes personas
# para no ser el valor de un solo individuo.
MIN_CLUSTER_SIZE = 5

# Variables con asimetría fuerte (skew > 3, documentado en T4 de 02_eda.ipynb).
# El clustering euclídeo sobre estas variables SIN transformar está dominado
# por los valores extremos: `Sgot` tiene skew 10.5 y su paciente más alto
# queda a ~17 desviaciones estándar. `log1p` comprime la cola sin borrar el
# orden. Ver el análisis de sensibilidad en 02c_eda_clustering.ipynb.
SKEWED_COLS = ["TB", "DB", "Alkphos", "Sgpt", "Sgot"]

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
