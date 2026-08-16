"""Tests de la Fase 9: auditoría de equidad por sexo (PRD §12.2, encargo de la fase).

`params`/`train_data` corren en milisegundos. `oof_dummy` usa un
`DummyClassifier` en vez de `logistic_regression` para ejercitar la
maquinaria real de :func:`src.fairness.oof_predictions_repeated_cv` (50
ajustes reales) sin pagar el costo de un estimador real -- suficiente para
verificar formas y conteos, no para verificar calidad predictiva.
`compare_table` es el único fixture caro del módulo (2 `GridSearchCV` +
4 variantes x 50 ajustes, ~60-70 s) y se comparte entre los tests que lo
necesitan, igual que `test_fase8_factorial.py`.
"""

import re
import warnings

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier

import src.fairness as fr
from src.config import PROJECT_ROOT, TARGET_COL, load_params
from src.data import load_modeling_data
from src.pipelines import build_pipeline
from src.splitting import load_split_indices

FORBIDDEN_TEST_REFERENCES = re.compile(r"\b(test_index|test_indices|X_test|y_test)\b")
_MISSING_TB_DB_INDICES = [246, 261, 279]


@pytest.fixture(scope="module")
def params() -> dict:
    return load_params()


@pytest.fixture(scope="module")
def train_data(params):
    df = load_modeling_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    split = load_split_indices(PROJECT_ROOT / params["split"]["indices_path"])
    train_idx = split["train_index"]
    return X.loc[train_idx], y.loc[train_idx]


@pytest.fixture(scope="module")
def oof_dummy(params, train_data):
    """OOF real (50 ajustes) con `DummyClassifier` -- rápido, ejercita `oof_predictions_repeated_cv` de verdad."""
    X_train, y_train = train_data
    X_sin_gender = X_train.drop(columns=["Gender"])
    pipe = build_pipeline(
        params["tuning_baseline"]["scaler"],
        False,
        DummyClassifier(strategy="stratified", random_state=params["seed"]),
        params,
        selector=fr.build_shielded_selector(k=5),
    )
    oof_matrix = fr.oof_predictions_repeated_cv(pipe, X_sin_gender, y_train, params)
    aggregated = fr.aggregate_oof_by_majority_vote(oof_matrix)
    return oof_matrix, aggregated


@pytest.fixture(scope="module")
def compare_table(params, train_data):
    """Único fixture caro del módulo: las 4 variantes de F9-R4/F9-R5 completas."""
    X_train, y_train = train_data
    return fr.compare_variants(X_train, y_train, params)


def test_metricas_oof_no_split_unico(params, train_data, oof_dummy):
    """F9-R2: las métricas se calculan sobre las 73 mujeres enfermas de TRAIN, no sobre un split único (~20)."""
    X_train, y_train = train_data
    oof_matrix, aggregated = oof_dummy

    assert oof_matrix.shape == (params["cv"]["fairness"]["n_repeats"], len(X_train))
    assert len(aggregated) == len(X_train) == 456

    gender = X_train["Gender"].to_numpy()
    metrics_df = fr.metrics_by_group(y_train.to_numpy(), aggregated, gender, params)

    assert metrics_df.loc[fr.GROUP_FEMALE, "n_enfermos"] == 73
    assert metrics_df.loc[fr.GROUP_MALE, "n_enfermos"] == 252
    assert metrics_df["n"].sum() == 456
    # Ni de cerca el tamaño de un split único (~28 mujeres, ~10 enfermas -- F4 §12).
    assert metrics_df.loc[fr.GROUP_FEMALE, "n_enfermos"] > 60


def test_test_no_usado_en_fase9():
    """Trampa 1: ninguna función de src/fairness.py referencia el conjunto de prueba."""
    src_text = (PROJECT_ROOT / "src" / "fairness.py").read_text(encoding="utf-8")
    match = FORBIDDEN_TEST_REFERENCES.search(src_text)
    assert match is None, f"src/fairness.py referencia el conjunto de prueba: {match.group(0)!r}"


def test_intervalos_por_bootstrap_de_pacientes(params):
    """Trampa 2: el IC de bootstrap_fnr_gap_ci viene de remuestrear PACIENTES, no repeticiones.

    Construye un vector sintético con la composición real declarada en el
    encargo (73 mujeres enfermas, 252 hombres enfermos, FNR≈0.45) y compara
    el ancho del IC bootstrap contra el ancho analítico de
    :func:`declared_resolution` para el mismo escenario. Si el bootstrap
    remuestreara repeticiones en vez de pacientes (el error que produciría
    una brecha "significativa" falsa), el IC saldría mucho más angosto que
    el analítico -- aquí se exige que esté en el mismo orden de magnitud.
    """
    rng = np.random.default_rng(123)
    n_sick_women, n_sick_men = 73, 252
    n_healthy_women, n_healthy_men = 39, 92
    assumed_fnr = 0.45

    y_true = np.array([1] * n_sick_women + [0] * n_healthy_women + [1] * n_sick_men + [0] * n_healthy_men)
    group = np.array([1] * (n_sick_women + n_healthy_women) + [0] * (n_sick_men + n_healthy_men))
    y_pred = y_true.copy()
    sick_mask = y_true == 1
    flip = rng.random(sick_mask.sum()) < assumed_fnr
    y_pred_sick = y_pred[sick_mask]
    y_pred_sick[flip] = 0
    y_pred[sick_mask] = y_pred_sick

    result = fr.bootstrap_fnr_gap_ci(y_true, y_pred, group, params, n_iterations=1500, seed=params["seed"])
    bootstrap_half_width_pp = (result["gap"]["ci_upper"] - result["gap"]["point_estimate"]) * 100

    analytic = fr.declared_resolution(n_sick_women, n_sick_men, assumed_fnr, params["evaluation"]["bootstrap"]["confidence_level"])
    analytic_half_width_pp = analytic["ci_half_width_gap_pp"]

    # Mismo orden de magnitud (factor 2), nunca "sospechosamente angosto":
    # remuestrear las 10 repeticiones en vez de los pacientes encogería el
    # error estándar en ~sqrt(10) ~ 3.16x -- justo el error que esta prueba
    # existe para atrapar.
    assert bootstrap_half_width_pp > analytic_half_width_pp / 2, (
        f"IC bootstrap (±{bootstrap_half_width_pp:.1f} pp) mucho más angosto que el analítico "
        f"(±{analytic_half_width_pp:.1f} pp) -- ¿se remuestrearon repeticiones en vez de pacientes?"
    )
    assert bootstrap_half_width_pp < analytic_half_width_pp * 2


def test_ambas_variantes_de_gender(params, compare_table):
    """F9-R5: existen resultados con y sin `Gender`, para ambas variantes de SMOTE (F9-R4)."""
    assert set(compare_table["include_gender"]) == {True, False}
    assert set(compare_table["use_smote"]) == {True, False}
    assert len(compare_table) == 4
    assert compare_table["gap_pp"].notna().all()
    assert compare_table["p_value"].between(0, 1).all()


def test_k_congelado_se_acota_sin_warnings_redundantes(params, train_data):
    """Higiene: el selector oficial acota `k`, sin depender de que `sklearn` absorba un `k` excesivo con un aviso.

    Fabrica un `best_params` con un `k` deliberadamente mayor que las
    columnas no-indicadoras disponibles (999, muy por encima de las 9
    reales sin `Gender`) para forzar el caso que antes disparaba un
    `UserWarning` ("k=N is greater than n_features=M") en cada uno de los
    50 ajustes de la CV repetida -- 100 avisos idénticos por variante que
    podrían tapar uno real en una fase futura. `run_fairness_variant` debe
    acotar `k` internamente (`_default_shielded_selector`) y no emitir ese
    aviso ni una sola vez.
    """
    X_train, y_train = train_data
    X_sin_gender = X_train.drop(columns=["Gender"])
    gender_train = X_train["Gender"]
    best_params_exagerado = {"estimator__C": 1.0, "estimator__solver": "liblinear", "selector__k": 999}

    with warnings.catch_warnings(record=True) as capturados:
        warnings.simplefilter("always")
        resultado = fr.run_fairness_variant(
            X_sin_gender,
            y_train,
            gender_train,
            params,
            use_smote=False,
            include_gender=False,
            best_params=best_params_exagerado,
        )

    avisos_k = [w for w in capturados if "is greater than n_features" in str(w.message)]
    assert not avisos_k, f"el selector oficial no debería depender del aviso de sklearn: {[str(w.message) for w in avisos_k]}"
    assert resultado["oof_matrix"].shape == (params["cv"]["fairness"]["n_repeats"], len(X_sin_gender))


def test_indicador_constante_resuelto(params, train_data):
    """Trampa 4: el selector blindado conserva el indicador TB/DB incluso en el pliegue sin las 3 filas nulas.

    Reproduce el escenario exacto medido en la Fase 6 (traspaso a la
    Fase 9, `AGENTS.md`): un pliegue de entrenamiento sin ninguna de las 3
    filas con `TB`/`DB` nulos dentro de ese pliegue deja el indicador
    constante. `SelectKBest(f_classif)` plano lo descarta en silencio (score
    `NaN`); el selector blindado lo conserva siempre, porque nunca compite.
    """
    X_train, y_train = train_data
    X_no_missing = X_train.drop(columns=["Gender"]).drop(index=_MISSING_TB_DB_INDICES)
    y_no_missing = y_train.drop(index=_MISSING_TB_DB_INDICES)

    from sklearn.feature_selection import SelectKBest, f_classif

    pre = build_pipeline("zscore", False, DummyClassifier(), params, selector=None).named_steps["preprocess"]
    transformed = pre.fit_transform(X_no_missing)
    assert (transformed.filter(like="tb_db_indicator").nunique() == 1).all(), "el escenario no reproduce el fold constante"

    plain = SelectKBest(f_classif, k=7)
    plain.set_output(transform="pandas")
    plain_out = plain.fit_transform(transformed, y_no_missing)
    assert not fr.indicator_columns_present(plain_out), "el selector plano debería descartar el indicador en este fold"

    shielded = fr.build_shielded_selector(k=7)
    shielded_out = shielded.fit_transform(transformed, y_no_missing)
    assert fr.indicator_columns_present(shielded_out), "el selector blindado debe conservar el indicador siempre"


def test_resolucion_declarada():
    """Trampa 3: `declared_resolution(...)` aparece en el notebook ANTES de la brecha real observada.

    Declarar la resolución después de ver el resultado sería racionalizarlo
    (misma disciplina que F8-R6 exigió para la predicción de MinMax). Se
    verifica el orden textual en el `.py` emparejado con jupytext, igual
    que `test_fase8_factorial.py::test_factorial_evaluado_en_cv_no_en_test`
    lee `src/factorial.py` como texto.
    """
    notebook_path = PROJECT_ROOT / "notebooks" / "09_fairness.py"
    assert notebook_path.exists(), "notebooks/09_fairness.py (emparejado con jupytext) no existe todavía"
    src_text = notebook_path.read_text(encoding="utf-8")

    resolution_call = "declared_resolution("
    real_gap_call = "fr.fnr_gap("

    resolution_idx = src_text.find(resolution_call)
    real_gap_idx = src_text.find(real_gap_call)

    assert resolution_idx != -1, "el notebook debe llamar a declared_resolution(...)"
    assert real_gap_idx != -1, "el notebook debe calcular la brecha real observada con fr.fnr_gap(...)"
    assert resolution_idx < real_gap_idx, (
        "la resolución declarada debe aparecer ANTES de la celda que calcula la brecha real observada"
    )
