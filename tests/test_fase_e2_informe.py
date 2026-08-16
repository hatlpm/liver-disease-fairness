"""Fase E2: las cifras publicadas coinciden con las que produce ``src/``.

Este archivo existe por una razón concreta. El riesgo de la fase de
entregables **no es calcular mal, es transcribir mal**: la Actividad 1
sufrió exactamente esa clase de error dos veces (Loop D y Loop E), con
cifras que estaban bien en el notebook y mal en el informe. Aquí se
reconstruyen las cantidades clave desde los módulos de ``src/`` y se
verifica que aparecen, con esos mismos dígitos, en los tres artefactos que
se entregan:

- ``reports/informe_act2.md``
- ``reports/informe_act2.tex`` (una segunda copia mantenida a mano --
  precisamente el modo de fallo del Loop E)
- ``notebooks/act2_anexo.ipynb`` (sus salidas **guardadas**, no
  re-ejecutadas: se verifica lo que el evaluador va a leer)

Coste: los *fixtures* de ámbito módulo ajustan los 5 `GridSearchCV` una vez
(~25 s) y corren la auditoría de equidad una vez (~15 s). Es deliberado:
verificar contra `src/` de verdad, no contra constantes copiadas aquí, que
tendrían el mismo problema de transcripción que se quiere evitar.
"""

import json
import re
import warnings

import pytest
from sklearn.feature_selection import SelectKBest, f_classif

from src.config import PROJECT_ROOT, load_params
from src.data import load_modeling_data
from src.evaluate import (
    NULL_ROW_LABEL,
    T7_REQUIRED_COLUMNS,
    confusion_matrix_narrative,
    fit_all_models,
    format_metrics_cell,
    predict_all_on_test,
    tabla_comparativa,
)
from src.factorial import evaluate_cell
from src.fairness import run_fairness_variant
from src.models import MODEL_KEYS, null_classifier_floor
from src.splitting import split_train_test

INFORME_MD = PROJECT_ROOT / "reports" / "informe_act2.md"
INFORME_TEX = PROJECT_ROOT / "reports" / "informe_act2.tex"
ANEXO = PROJECT_ROOT / "notebooks" / "act2_anexo.ipynb"

# Las 3 celdas que la Fase 8 encontró degeneradas (Trampa 4). Se declaran
# como el ESCENARIO a verificar, no como el resultado: el test comprueba
# contra `evaluate_cell` que efectivamente predicen una sola clase.
CELDAS_DEGENERADAS = [
    ("logistic_regression", "minmax", "none"),
    ("svm", "minmax", "none"),
    ("svm", "zscore", "none"),
]

_NUMERO = re.compile(r"\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Utilidades de lectura
# ---------------------------------------------------------------------------


def _numeros(texto: str) -> set[str]:
    """Conjunto de literales numéricos de un texto.

    Comparar conjuntos de literales, y no el texto formateado, hace el test
    inmune al marcado: ``0.7234`` es el mismo token en Markdown que dentro
    de un ``\\textbf{}`` de LaTeX. Un dígito mal transcrito, en cambio, deja
    de pertenecer al conjunto y el test falla.
    """
    return set(_NUMERO.findall(texto))


def _contiene(texto: str, *valores: float, decimales: int = 4) -> list[str]:
    """Devuelve los valores que NO aparecen en `texto`, ya formateados."""
    presentes = _numeros(texto)
    return [f"{v:.{decimales}f}" for v in valores if f"{v:.{decimales}f}" not in presentes]


def _salidas_del_anexo() -> str:
    """Todo el texto de las salidas GUARDADAS del anexo, concatenado."""
    notebook = json.loads(ANEXO.read_text(encoding="utf-8"))
    partes = []
    for celda in notebook.get("cells", []):
        for salida in celda.get("outputs", []):
            if salida.get("output_type") == "stream":
                partes.append("".join(salida.get("text", [])))
            elif "data" in salida:
                partes.append("".join(salida["data"].get("text/plain", [])))
    return "\n".join(partes)


def _fuentes_del_anexo() -> list[str]:
    notebook = json.loads(ANEXO.read_text(encoding="utf-8"))
    return ["".join(c.get("source", [])) for c in notebook.get("cells", [])]


@pytest.fixture(scope="module")
def md() -> str:
    return INFORME_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tex() -> str:
    return INFORME_TEX.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def anexo() -> str:
    return _salidas_del_anexo()


# ---------------------------------------------------------------------------
# Fixtures caros: se recalcula desde src/ una sola vez por módulo
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def datos(params):
    df = load_modeling_data()
    idx_train, idx_test = split_train_test(df, params)
    X = df.drop(columns=["Selector"])
    y = df["Selector"]
    return {
        "df": df,
        "X_train": X.loc[idx_train].drop(columns=["Gender"]),
        "y_train": y.loc[idx_train],
        "X_test": X.loc[idx_test].drop(columns=["Gender"]),
        "y_test": y.loc[idx_test],
        "gender_train": X.loc[idx_train, "Gender"],
    }


@pytest.fixture(scope="module")
def t7(datos, params):
    """Los 5 GridSearchCV + la tabla oficial de T7, desde `src/` (~25 s)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        searches = fit_all_models(datos["X_train"], datos["y_train"], params, selector=SelectKBest(f_classif))
        y_pred = predict_all_on_test(searches, datos["X_test"])
        tabla = tabla_comparativa(searches, y_pred, datos["y_test"], params)
    return {"searches": searches, "y_pred": y_pred, "tabla": tabla}


@pytest.fixture(scope="module")
def equidad(datos, params, t7):
    """La variante oficial de la auditoría de equidad, desde `src/` (~15 s)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run_fairness_variant(
            datos["X_train"],
            datos["y_train"],
            datos["gender_train"],
            params,
            use_smote=True,
            include_gender=False,
            best_params=t7["searches"]["logistic_regression"].best_params_,
        )


# ---------------------------------------------------------------------------
# Informe <-> src/
# ---------------------------------------------------------------------------


def test_tamanos_del_dataset(md, tex, datos):
    """570 / 456 / 114 y las composiciones de clase salen de `src/`, no de memoria."""
    esperados = [
        len(datos["df"]),
        len(datos["X_train"]),
        len(datos["X_test"]),
        int((datos["y_train"] == 1).sum()),
        int((datos["y_train"] == 0).sum()),
        int((datos["y_test"] == 1).sum()),
        int((datos["y_test"] == 0).sum()),
    ]
    assert esperados == [570, 456, 114, 325, 131, 81, 33], "cambió el contrato de datos de la Fase 4"

    for nombre, texto in (("informe_act2.md", md), ("informe_act2.tex", tex)):
        faltan = [str(v) for v in esperados if str(v) not in _numeros(texto)]
        assert not faltan, f"{nombre} no cita los tamaños {faltan}"


def test_tabla_t7_completa(md, tex, t7):
    """Las 4 columnas de §3.2 y, para los 5 modelos, sus métricas exactas."""
    tabla = t7["tabla"]
    for columna in T7_REQUIRED_COLUMNS:
        assert columna in tabla.columns

    numeros_tex = _numeros(tex)
    for _, fila in tabla.iterrows():
        if fila["Modelo"] == NULL_ROW_LABEL:
            continue
        celda = fila["Métricas de mejor rendimiento"]
        # En Markdown la celda se transcribe literal: se exige idéntica.
        assert celda in md, f"informe_act2.md: métricas de {fila['Modelo']} no coinciden con src/ -> {celda!r}"
        # En LaTeX el separador cambia; se exigen los 4 valores numéricos.
        faltan = [v for v in _NUMERO.findall(celda) if v not in numeros_tex]
        assert not faltan, f"informe_act2.tex: a {fila['Modelo']} le faltan {faltan}"

        bal = f"{fila['Balanced accuracy']:.4f}"
        assert bal in _numeros(md) and bal in numeros_tex, f"balanced accuracy de {fila['Modelo']} ({bal})"


def test_fila_del_clasificador_nulo(md, tex, datos, params, t7):
    """El suelo nulo está en la tabla y sus 5 cifras salen de `null_classifier_floor`."""
    tabla = t7["tabla"]
    assert (tabla["Modelo"] == NULL_ROW_LABEL).any(), "la tabla de T7 debe llevar la fila del clasificador nulo"

    suelo = null_classifier_floor(datos["y_test"], pos_label=params["metrics"]["pos_label"])
    assert format_metrics_cell(suelo) in md, "informe_act2.md: la fila del nulo no coincide con src/"

    for nombre, texto in (("md", md), ("tex", tex)):
        faltan = _contiene(texto, suelo["accuracy"], suelo["f1"], suelo["balanced_accuracy"])
        assert not faltan, f"informe_act2.{nombre} no cita el suelo nulo {faltan}"


def test_matriz_de_confusion_del_ganador(md, tex, datos, params, t7):
    """Los 43 enfermos sin detectar (y los otros 3 conteos) salen de `src/`."""
    ganador = max(t7["searches"], key=lambda n: t7["searches"][n].best_score_)
    assert ganador == "logistic_regression", "cambió el ganador declarado por CV"

    narr = confusion_matrix_narrative(
        datos["y_test"], t7["y_pred"][ganador], pos_label=params["metrics"]["pos_label"]
    )
    assert narr == {
        "enfermos_detectados": 38,
        "enfermos_no_detectados": 43,
        "sanos_correctos": 28,
        "sanos_falsos_positivos": 5,
    }

    for nombre, texto in (("md", md), ("tex", tex)):
        faltan = [str(v) for v in narr.values() if str(v) not in _numeros(texto)]
        assert not faltan, f"informe_act2.{nombre} no cita los conteos {faltan}"


def test_celdas_degeneradas(md, tex, datos, params, t7):
    """Las 3 celdas de T8 predicen UNA sola clase para todo train (Trampa 4)."""
    for modelo, escalador, balanceo in CELDAS_DEGENERADAS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fila = evaluate_cell(
                modelo,
                escalador,
                balanceo,
                t7["searches"][modelo].best_params_,
                datos["X_train"],
                datos["y_train"],
                params,
                selector=SelectKBest(f_classif),
            )
        assert fila["balanced_accuracy_cv"] == pytest.approx(0.5, abs=1e-9), (
            f"{modelo}/{escalador}/{balanceo} ya no es degenerada: el informe la reporta como tal"
        )

    n_sanos = int((datos["y_train"] == 0).sum())
    for nombre, texto in (("md", md), ("tex", tex)):
        assert str(n_sanos) in _numeros(texto), f"informe_act2.{nombre} no cita los {n_sanos} sanos de train"


def test_brecha_fnr_y_p_valor(md, tex, equidad):
    """+19.08 pp, las dos FNR, el IC95 y p=0.0043, contra `src.fairness`."""
    gap = equidad["gap"]
    significancia = equidad["significance"]
    bootstrap = equidad["bootstrap"]

    assert gap["gap_pp"] == pytest.approx(19.08, abs=0.01)
    assert significancia["p_value"] == pytest.approx(0.0043, abs=0.0001)

    # Comprobación literal sobre el Markdown, además de la de conjunto: si la
    # misma cifra aparece en varios sitios (prosa + tabla), buscarla en el
    # conjunto de números NO detecta que UNA de esas apariciones se haya
    # tecleado mal. Estas dos son las cifras titulares de la sección.
    literales = [f"+{gap['gap_pp']:.2f} pp", f"p = {significancia['p_value']:.4f}"]
    faltan_literales = [v for v in literales if v not in md]
    assert not faltan_literales, (
        f"informe_act2.md no contiene, en su forma canónica, {faltan_literales} -- "
        "si se reformuló el texto, actualizar aquí y en el .tex a la vez"
    )

    for nombre, texto in (("md", md), ("tex", tex)):
        numeros = _numeros(texto)
        faltan = [
            f"{gap['gap_pp']:.2f}",
            f"{gap['fnr_group_a']:.4f}",
            f"{gap['fnr_group_b']:.4f}",
            f"{significancia['p_value']:.4f}",
        ]
        faltan = [v for v in faltan if v not in numeros]
        assert not faltan, f"informe_act2.{nombre} no cita la brecha/p-valor {faltan}"

        # `bootstrap["gap"]` viene en fracción; el informe lo publica en pp
        # con 1 decimal ([+5.6, +31.8]).
        ic = [
            f"{100 * bootstrap['gap']['ci_lower']:.1f}",
            f"{100 * bootstrap['gap']['ci_upper']:.1f}",
        ]
        faltan_ic = [v for v in ic if v not in numeros]
        assert not faltan_ic, f"informe_act2.{nombre} no cita el IC95 {faltan_ic}"


def test_informe_sin_bloques_de_codigo(md):
    """El enunciado exige que el código vaya al anexo, no al informe."""
    assert "```" not in md, "informe_act2.md contiene un bloque de código; deben ir en act2_anexo.ipynb"


def test_md_y_tex_coinciden(md, tex, datos, params, t7, equidad):
    """Las dos copias del informe se mantienen a mano: modo de fallo del Loop E."""
    ganador = max(t7["searches"], key=lambda n: t7["searches"][n].best_score_)
    claves = [
        f"{t7['searches'][ganador].best_score_:.4f}",  # 0.7234
        f"{equidad['gap']['gap_pp']:.2f}",  # 19.08
        f"{equidad['significance']['p_value']:.4f}",  # 0.0043
        f"{null_classifier_floor(datos['y_test'], pos_label=params['metrics']['pos_label'])['f1']:.4f}",
    ]
    numeros_md, numeros_tex = _numeros(md), _numeros(tex)
    divergentes = [v for v in claves if (v in numeros_md) != (v in numeros_tex)]
    assert not divergentes, f".md y .tex divergen en {divergentes}"


# ---------------------------------------------------------------------------
# Anexo <-> src/
# ---------------------------------------------------------------------------


def test_anexo_sin_errores():
    """El anexo entregado pasó *restart & run all*: ninguna salida es un error."""
    notebook = json.loads(ANEXO.read_text(encoding="utf-8"))
    errores = [
        (i, salida.get("ename"))
        for i, celda in enumerate(notebook.get("cells", []))
        for salida in celda.get("outputs", [])
        if salida.get("output_type") == "error"
    ]
    assert not errores, f"act2_anexo.ipynb tiene celdas con error: {errores}"


def test_anexo_orden_literal_t1_t8():
    """Las 8 tareas, en el orden del enunciado (FE2-R1)."""
    fuentes = "\n".join(_fuentes_del_anexo())
    posiciones = []
    for n in range(1, 9):
        encabezado = f"## Tarea {n} "
        assert encabezado in fuentes, f"falta el encabezado de la Tarea {n}"
        posiciones.append(fuentes.index(encabezado))
    assert posiciones == sorted(posiciones), "las 8 tareas no están en orden literal T1->T8"


def test_anexo_declarado_como_copia_de_presentacion():
    """La excepción de ADR-0009 se declara en el propio anexo, no solo en el ADR."""
    primera = _fuentes_del_anexo()[0]
    assert "copia de presentación" in primera
    assert "ADR-0009" in primera or "0009-src-fuente-de-verdad" in primera


def test_anexo_reproduce_las_cifras_de_src(anexo, datos, params, t7, equidad):
    """Las salidas guardadas del anexo llevan las mismas cifras que `src/`."""
    numeros = _numeros(anexo)
    suelo = null_classifier_floor(datos["y_test"], pos_label=params["metrics"]["pos_label"])
    ganador = max(t7["searches"], key=lambda n: t7["searches"][n].best_score_)
    narr = confusion_matrix_narrative(
        datos["y_test"], t7["y_pred"][ganador], pos_label=params["metrics"]["pos_label"]
    )

    esperados = [
        str(len(datos["df"])),
        str(len(datos["X_train"])),
        str(len(datos["X_test"])),
        str(narr["enfermos_no_detectados"]),
        f"{t7['searches'][ganador].best_score_:.4f}",
        f"{suelo['f1']:.4f}",
        f"{equidad['gap']['gap_pp']:.2f}",
        f"{equidad['significance']['p_value']:.4f}",
    ]
    faltan = [v for v in esperados if v not in numeros]
    assert not faltan, f"act2_anexo.ipynb no reproduce {faltan}"

    # La tabla de T7 del anexo, celda a celda.
    for _, fila in t7["tabla"].iterrows():
        celda = fila["Métricas de mejor rendimiento"]
        faltan_celda = [v for v in _NUMERO.findall(celda) if v not in numeros]
        assert not faltan_celda, f"act2_anexo.ipynb: métricas de {fila['Modelo']} -> faltan {faltan_celda}"


def test_anexo_verifica_su_equivalencia_con_src(anexo):
    """El propio anexo corre el contraste contra `src/` y lo deja registrado."""
    assert "ANEXO B COMPLETO" in anexo, (
        "el Anexo B de verificación no llegó a ejecutarse: el anexo entregado no demuestra "
        "que coincide con src/"
    )
    assert all(m in anexo for m in MODEL_KEYS), "el anexo no reporta los 5 algoritmos exigidos"
