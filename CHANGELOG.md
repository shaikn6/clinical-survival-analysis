# Changelog

All notable changes to this project are documented here.

## v2.0.0 — 2026-05-30
### Added
- DeepSurv: neural Cox Proportional Hazard model (PyTorch), Breslow partial likelihood loss
- DeepHit: discrete-time competing risks model (PyTorch), combined log-likelihood + ranking loss
- Competing risks analysis: cause-specific hazard functions, Aalen-Johansen CIF estimator
- Extended evaluation: 6-model comparison table (KM, Cox PH, RSF, XGBoost, DeepSurv, DeepHit)

## [1.0.0] — 2026-05-30

### Added
- Data pipeline: WHAS500 and GBSG2 loaders via lifelines; synthetic ICU cohort generator calibrated to MIMIC-III statistics
- Feature engineering: age groups, comorbidity index levels, LOS categories
- Preprocessing: median imputation, StandardScaler, stratified 70/15/15 train/val/test split
- Kaplan-Meier model with subgroup analysis, log-rank tests, Nelson-Aalen cumulative hazard
- Cox Proportional Hazards model with Schoenfeld residual PH assumption check, hazard ratio table
- Random Survival Forest (100 trees) with MDI and permutation feature importances, OOB C-index
- Gradient Boosting survival model with hyperparameter grid search and SHAP value computation
- Evaluation metrics: C-index, Integrated Brier Score, time-dependent AUC, calibration curves
- Visualizations: KM survival curves (matplotlib + Plotly), forest plot, feature importance charts, calibration plots
- Streamlit dashboard: 5 tabs covering dataset overview, KM curves, model comparison, patient prediction, feature importance
- FastAPI service: POST /predict, GET /models, GET /health endpoints
- 60+ pytest tests covering all modules
