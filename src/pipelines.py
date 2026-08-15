"""Fábrica de `Pipeline` de preprocesamiento y balanceo (Fase 5, F5-R1..R3).

Construye, a partir de `params.yaml`, un `imblearn.pipeline.Pipeline` que
encadena imputación -> escalado -> (opcional) SMOTE -> estimador. Es una
fábrica, no un pipeline fijo: la Fase 8 la reutiliza sin cambios para
construir las 20 combinaciones {MinMax, Z-Score} x {con SMOTE, sin SMOTE} x
5 modelos.

Por qué `imblearn.pipeline.Pipeline` y no `sklearn.pipeline.Pipeline`
----------------------------------------------------------------------
El `Pipeline` de sklearn no sabe qué hacer con un paso de remuestreo: le
pediría a `SMOTE` que transforme también en `predict`/`transform`, aplicando
el sobremuestreo al conjunto de prueba. El de `imblearn` aplica los pasos de
remuestreo SOLO durante `fit` (vía `fit_resample`) y los salta en cualquier
otra llamada -- es la propiedad que garantiza F5-R5 (SMOTE nunca toca test
ni los pliegues de validación).

Por qué el indicador de nulos NO usa `SimpleImputer(add_indicator=True)`
--------------------------------------------------------------------------
Ver ADR-0011. En resumen: `add_indicator=True` crea el indicador con
`features="missing-only"`, que solo genera una columna para una variable si
ESE pliegue concreto tiene algún nulo en ella -- el número de columnas de
salida cambiaría entre pliegues de `RepeatedStratifiedKFold` (Fase 9, 50
ajustes). Aquí el indicador se construye en una rama independiente del
`ColumnTransformer` con `MissingIndicator(features="all")`, que siempre
genera 2 columnas (`TB`, `DB`), tenga o no nulos el pliegue.
"""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.config import NUMERIC_COLS

_SCALERS = {"minmax": MinMaxScaler, "zscore": StandardScaler}
_INDICATOR_SOURCE_COLS = ["TB", "DB"]


def build_pipeline(scaler: str, use_smote: bool, estimator, params: dict) -> ImbPipeline:
    """Construye el `Pipeline` de preprocesamiento (+ balanceo) + estimador.

    Encadena, en este orden: imputación por mediana y escalado de las
    variables numéricas (F5-R2), un indicador de ancho fijo para los nulos
    de `TB`/`DB` (F5-R3, sin escalar), opcionalmente `SMOTE` (F5-R4), y el
    estimador que se le pase. Todo parámetro numérico (semilla, `k_neighbors`
    de SMOTE, estrategia del imputador) sale de `params`, nunca literal.

    Parameters
    ----------
    scaler : str
        Nombre del escalador a usar. Debe estar en
        ``params["preprocessing"]["scalers"]`` (hoy: ``"minmax"`` o
        ``"zscore"``).
    use_smote : bool
        Si ``True``, añade un paso `SMOTE` entre el preprocesamiento y el
        estimador. Si ``False``, el paso se omite por completo (no se deja
        un `"passthrough"`: el `Pipeline` resultante tiene un paso menos).
    estimator : sklearn estimator
        Estimador final del pipeline. La Fase 5 no elige modelo (eso es la
        Fase 6): aquí se admite cualquier estimador compatible con la API de
        `sklearn` (`DummyClassifier` como placeholder de prueba, o
        cualquiera de los 5 algoritmos de fases posteriores).
    params : dict
        Árbol de parámetros de :func:`src.config.load_params`.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Pipeline con pasos ``"preprocess"``, opcionalmente ``"smote"``, y
        ``"estimator"``. Espera como entrada un DataFrame con exactamente
        las columnas de `NUMERIC_COLS` + `"Gender"` (es decir,
        ``load_modeling_data().drop(columns=["Selector"])``) -- nunca
        incluir la columna objetivo, o el `remainder="passthrough"` la
        dejaría entrar como si fuera una variable más.

    Raises
    ------
    ValueError
        Si `scaler` no está entre los valores declarados en
        ``params["preprocessing"]["scalers"]``, o si `use_smote=True` pero
        ``"smote"`` no está en ``params["balancing"]["methods"]``.
    """
    valid_scalers = params["preprocessing"]["scalers"]
    if scaler not in valid_scalers:
        raise ValueError(f"scaler={scaler!r} no está en params.preprocessing.scalers={valid_scalers}")
    if use_smote and "smote" not in params["balancing"]["methods"]:
        raise ValueError('use_smote=True pero "smote" no está en params.balancing.methods')

    numeric_branch = SkPipeline(
        [
            ("imputer", SimpleImputer(strategy=params["preprocessing"]["imputer"]["strategy"])),
            ("scaler", _SCALERS[scaler]()),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric_branch, NUMERIC_COLS),
            ("tb_db_indicator", MissingIndicator(features="all"), _INDICATOR_SOURCE_COLS),
        ],
        remainder="passthrough",
    )
    preprocess.set_output(transform="pandas")

    steps = [("preprocess", preprocess)]
    if use_smote:
        steps.append(
            (
                "smote",
                SMOTE(
                    random_state=params["seed"],
                    k_neighbors=params["balancing"]["smote"]["k_neighbors"],
                ),
            )
        )
    steps.append(("estimator", estimator))

    return ImbPipeline(steps)
