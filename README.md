# Liver Disease Fairness

Predicción de enfermedad hepática con auditoría de *fairness* por sexo, sobre el dataset ILPD (Indian Liver Patient Dataset). Proyecto desarrollado siguiendo CRISP-DM.

## Problema

Un grupo hospitalario quiere reducir la mortalidad por cirrosis mejorando el diagnóstico temprano. El hígado no duele: el daño puede ser extenso antes de que aparezcan síntomas. Un análisis de sangre barato (*Liver Function Test*) puede alimentar un modelo de cribado que priorice a quién derivar al especialista.

El proyecto no se limita a un EDA genérico: replica el marco de Straw & Wu (2022), que documentó sesgo de sexo en este mismo dataset (mayor tasa de falsos negativos en mujeres). Todo el análisis se hace también estratificado por `Gender`.

## Estado actual

Fases 0–3 (Setup, Problem Framing, EDA, Preprocessing) en desarrollo, según `PRD_liver_disease_fases_0-3.md`. Las Fases 4–7 (modelado, fairness audit, pipeline, dashboard) quedan para un PRD posterior.

## Estructura

```
liver-disease-fairness/
├── data/
│   ├── raw/            dataset original (inmutable)
│   └── processed/      dataset tratado (Fase 3)
├── notebooks/           01_problem_framing, 02_eda, 03_preprocessing, act1_anexo
├── src/                 config.py, utils.py
├── reports/             informe_act1.md + figures/
├── docs/                data_dictionary.md, CHANGELOG_iteraciones.md
```

## Cómo ejecutar

```bash
python -m venv venv
./venv/Scripts/activate       # Windows
pip install -r requirements.txt
jupyter notebook
```

Todos los notebooks son reproducibles: *restart & run all* debe correr sin errores, sin paths absolutos, con `RANDOM_STATE = 42` fijo en todo el código.

## Referencia

Straw, I. & Wu, H. (2022). *Investigating for bias in healthcare algorithms: a sex-stratified analysis of supervised machine learning models in liver disease prediction.* BMJ Health Care Inform, 29(1). https://doi.org/10.1136/bmjhci-2021-100457
