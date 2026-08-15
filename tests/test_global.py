"""Tests globales de higiene del repo (PRD §12.2, no atados a una fase).

Alcance deliberadamente acotado a **código versionado real**: ``src/``,
``tests/``, ``notebooks/*.py`` (el emparejado jupytext), ``params.yaml`` y
``pyproject.toml``. Se excluye a propósito la documentación en Markdown
(``AGENTS.md``, ``docs/CHANGELOG_iteraciones.md``), que ya narra en prosa
histórica tanto una ruta absoluta rota de otra máquina como el incidente de
``matplotlib.use("Agg")`` del Loop E -- escanearla produciría falsos
positivos sobre texto que documenta el problema, no que lo reintroduce.
"""

import json
import re

from src.config import PROJECT_ROOT, RANDOM_STATE, load_params

ABS_PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
AGG_BACKEND_PATTERN = re.compile(r"""matplotlib\.use\(\s*["']Agg["']\s*\)""")
RANDOM_STATE_LITERAL_PATTERN = re.compile(r"random_state\s*=\s*\d+")

CODE_GLOBS = ["src/**/*.py", "tests/**/*.py", "notebooks/*.py"]
CONFIG_FILES = ["params.yaml", "pyproject.toml"]


def _versioned_code_files():
    for pattern in CODE_GLOBS:
        yield from PROJECT_ROOT.glob(pattern)
    for name in CONFIG_FILES:
        path = PROJECT_ROOT / name
        if path.exists():
            yield path


def _notebook_code_sources(ipynb_path):
    notebook = json.loads(ipynb_path.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            yield "".join(cell.get("source", []))


def test_sin_rutas_absolutas():
    """Ningún archivo de código/config versionado contiene una ruta absoluta de Windows."""
    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in _versioned_code_files()
        if ABS_PATH_PATTERN.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"Rutas absolutas encontradas en: {offenders}"


def test_sin_backend_agg():
    """Ningún notebook (ni su .py emparejado) reintroduce matplotlib.use('Agg')."""
    offenders = []
    for ipynb_path in (PROJECT_ROOT / "notebooks").glob("*.ipynb"):
        if any(AGG_BACKEND_PATTERN.search(src) for src in _notebook_code_sources(ipynb_path)):
            offenders.append(str(ipynb_path.relative_to(PROJECT_ROOT)))
    for py_path in (PROJECT_ROOT / "notebooks").glob("*.py"):
        if AGG_BACKEND_PATTERN.search(py_path.read_text(encoding="utf-8")):
            offenders.append(str(py_path.relative_to(PROJECT_ROOT)))
    assert not offenders, f"'matplotlib.use(\"Agg\")' encontrado en: {offenders}"


def test_semilla_fijada():
    """params.yaml y src/config.py declaran la misma semilla, y src/ nunca la hardcodea suelta."""
    params = load_params()
    assert params["seed"] == RANDOM_STATE == 42

    offenders = [
        str(path.relative_to(PROJECT_ROOT))
        for path in (PROJECT_ROOT / "src").glob("**/*.py")
        if RANDOM_STATE_LITERAL_PATTERN.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"random_state literal (no desde config/params) en: {offenders}"
