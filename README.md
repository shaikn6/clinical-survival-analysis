> **Private Repository** — Source code available on request for verified employers.
> Contact: shaikn6@udayton.edu

# clinical-survival-analysis

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e)
![Tests](https://img.shields.io/badge/Tests-passing-22c55e)
![Stack](https://img.shields.io/badge/Stack-lifelines%20·%20scikit--survival%20·%20XGBoost-6366f1)

## What's New in V2

- **DeepSurv** — Neural Cox Proportional Hazard model (PyTorch, Katzman et al. 2018): MLP backbone with BatchNorm and Dropout, trained on Breslow partial likelihood loss
- **DeepHit** — Discrete-time competing-risks model (PyTorch, Lee et al. 2018): shared trunk + cause-specific heads, combined log-likelihood + ranking loss
- **Competing Risks Analysis** — Cause-specific hazard functions via separate Cox PH models; Aalen-Johansen Cumulative Incidence Function estimator with visualisation
- **Extended Evaluation** — Unified 6-model comparison table (KM, Cox PH, RSF, XGBoost, DeepSurv, DeepHit) with C-index and Brier scores; Tab 6 in the Streamlit dashboard shows loss curves and patient-level survival from deep models

## About

Production-grade survival analysis pipeline comparing four statistical and machine-learning models across real public clinical datasets and a synthetic ICU cohort. Built for healthcare AI practitioners and engineered for reproducibility, interpretability, and deployment.

## Clinical Context

Survival analysis is the statistical backbone of clinical decision support in healthcare AI. Unlike binary classification, it answers the question: *when* will an event occur, given that it has not happened yet? This is critical for:

- **ICU mortality prediction** — stratifying patients by 30/90/180-day risk to prioritise interventions
- **Oncology** — estimating time-to-recurrence, informing adjuvant therapy decisions
- **Cardiology** — post-MI risk stratification and discharge planning

Correct handling of *right-censored* observations (patients who leave follow-up before the event) is what distinguishes survival models from naive classification. All four models in this project handle censoring appropriately.

## Datasets

| Dataset | Description | N | Outcome | Source |
|---------|-------------|---|---------|--------|
| **WHAS500** | Worcester Heart Attack Study | 500 | All-cause mortality post-MI | `lifelines.datasets.load_whas500()` |
| **GBSG2** | German Breast Cancer Study Group | 686 | Recurrence-free survival | `lifelines.datasets.load_gbsg2()` |
| **Synthetic ICU** | Calibrated to MIMIC-III stats (age 65±15, LOS 5±3d, ~12% 30d mortality) | 1 000 | 30-day mortality | Generated programmatically (seed-reproducible) |

All datasets are public domain or freely available — no IRB approval or data access agreements required.

## Models

| Model | Library | Handles Censoring | Interpretability |
|-------|---------|-------------------|-----------------|
| Kaplan-Meier | lifelines | Yes (non-parametric) | Survival curve + log-rank test |
| Cox Proportional Hazards | lifelines | Yes | Hazard ratios + Schoenfeld residuals |
| Random Survival Forest | scikit-survival | Yes | Permutation importance |
| Gradient Boosting (Cox loss) | scikit-survival | Yes | MDI + SHAP values |

## Model Comparison (Representative Results)

| Model | WHAS500 C-index | GBSG2 C-index | ICU C-index |
|-------|-----------------|---------------|-------------|
| Kaplan-Meier (baseline) | — | — | — |
| Cox PH | ~0.72 | ~0.68 | ~0.73 |
| Random Survival Forest | ~0.76 | ~0.71 | ~0.76 |
| Gradient Boosting | ~0.77 | ~0.70 | ~0.75 |

*Actual values generated at runtime and saved to `results/comparison.csv`.*

## Project Structure

```
clinical-survival-analysis/
├── src/
│   ├── data.py             — dataset loaders, ICU generator, preprocessing
│   ├── evaluate.py         — C-index, IBS, time-AUC, calibration
│   ├── visualize.py        — KM curves, forest plot, feature importance
│   ├── api.py              — FastAPI service (POST /predict, GET /models)
│   └── models/
│       ├── kaplan_meier.py         — KM + log-rank + Nelson-Aalen
│       ├── cox_ph.py               — Cox PH + PH assumption check
│       ├── random_survival_forest.py  — RSF + permutation importance
│       └── xgboost_survival.py     — GBM survival + SHAP values
├── tests/                  — 60+ pytest tests
├── app.py                  — Streamlit dashboard (5 tabs)
├── results/
│   ├── comparison.csv
│   └── figures/
└── requirements.txt
```

## Quick Start

```bash
# Clone and install
git clone https://github.com/shaikn6/clinical-survival-analysis
cd clinical-survival-analysis
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Launch Streamlit dashboard
streamlit run app.py

# Start FastAPI server
uvicorn src.api:app --reload --port 8000

# Predict from CLI
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 70, "sofa_score": 10, "cci": 5, "creatinine": 2.5,
       "lactate": 3.0, "map": 65, "spo2": 93, "los": 8,
       "diabetes": 1, "hypertension": 1, "heart_failure": 1,
       "ckd": 0, "copd": 0}'
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check |
| `GET` | `/models` | List models and C-index scores |
| `POST` | `/predict` | Patient features → survival at [30, 90, 180, 365] days |

## Author

Nagizaaz Shaik — MLOps Engineer · [github.com/shaikn6](https://github.com/shaikn6)
