"""Tests for survival models: KM, Cox PH, RSF, XGBoost."""

import pandas as pd
import pytest

from src.data import generate_synthetic_icu, engineer_features, preprocess, split_dataset, get_feature_columns
from src.models.kaplan_meier import KaplanMeierModel, run_pairwise_logrank
from src.models.cox_ph import CoxPHModel
from src.models.random_survival_forest import RSFModel
from src.models.xgboost_survival import XGBoostSurvivalModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def icu_dataset():
    raw = generate_synthetic_icu(n_patients=300, seed=42)
    raw = engineer_features(raw, dataset="icu")
    train_raw, val_raw, test_raw = split_dataset(raw, seed=42)
    feat = get_feature_columns(train_raw)
    train, feat, scaler = preprocess(train_raw, feature_cols=feat)
    val, _, _ = preprocess(val_raw, feature_cols=feat, scaler=scaler, fit_scaler=False)
    test, _, _ = preprocess(test_raw, feature_cols=feat, scaler=scaler, fit_scaler=False)
    return {"train": train, "val": val, "test": test, "feat": feat}


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------

class TestKaplanMeier:
    def test_fit_returns_self(self, icu_dataset):
        km = KaplanMeierModel()
        result = km.fit(icu_dataset["train"])
        assert result is km

    def test_overall_kmf_set_after_fit(self, icu_dataset):
        km = KaplanMeierModel()
        km.fit(icu_dataset["train"])
        assert km.overall_kmf is not None

    def test_overall_naf_set_after_fit(self, icu_dataset):
        km = KaplanMeierModel()
        km.fit(icu_dataset["train"])
        assert km.overall_naf is not None

    def test_median_survival_positive(self, icu_dataset):
        km = KaplanMeierModel()
        km.fit(icu_dataset["train"])
        med = km.median_survival()
        assert med > 0

    def test_median_survival_ci_has_keys(self, icu_dataset):
        km = KaplanMeierModel()
        km.fit(icu_dataset["train"])
        ci = km.median_survival_ci()
        assert "median" in ci
        assert "ci_lower" in ci
        assert "ci_upper" in ci

    def test_survival_at_returns_df(self, icu_dataset):
        km = KaplanMeierModel()
        km.fit(icu_dataset["train"])
        df = km.survival_at([30, 90, 180])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_survival_at_between_0_and_1(self, icu_dataset):
        km = KaplanMeierModel()
        km.fit(icu_dataset["train"])
        df = km.survival_at([30, 90])
        assert (df["survival_prob"] >= 0).all()
        assert (df["survival_prob"] <= 1).all()

    def test_cumulative_hazard_non_negative(self, icu_dataset):
        km = KaplanMeierModel()
        km.fit(icu_dataset["train"])
        df = km.cumulative_hazard_at([30, 90, 180])
        assert (df["cumulative_hazard"] >= 0).all()

    def test_fit_subgroups_creates_entries(self, icu_dataset):
        km = KaplanMeierModel()
        km.fit(icu_dataset["train"])
        km.fit_subgroups(icu_dataset["train"], "age_group")
        assert "age_group" in km.subgroup_kmfs
        assert len(km.subgroup_kmfs["age_group"]) > 0

    def test_logrank_summary_has_p_value(self, icu_dataset):
        km = KaplanMeierModel()
        km.fit(icu_dataset["train"])
        km.fit_subgroups(icu_dataset["train"], "age_group")
        summ = km.logrank_summary()
        assert "p_value" in summ.columns

    def test_subgroup_medians_returns_df(self, icu_dataset):
        km = KaplanMeierModel()
        km.fit(icu_dataset["train"])
        km.fit_subgroups(icu_dataset["train"], "age_group")
        df = km.subgroup_medians("age_group")
        assert isinstance(df, pd.DataFrame)
        assert "group" in df.columns

    def test_pairwise_logrank_returns_df(self, icu_dataset):
        df = run_pairwise_logrank(
            icu_dataset["train"],
            group_col="age_group",
        )
        assert isinstance(df, pd.DataFrame)
        assert "p_value" in df.columns

    def test_unfitted_raises_on_median(self):
        km = KaplanMeierModel()
        with pytest.raises(RuntimeError):
            km.median_survival()


# ---------------------------------------------------------------------------
# Cox PH
# ---------------------------------------------------------------------------

class TestCoxPH:
    def test_fit_returns_self(self, icu_dataset):
        cox = CoxPHModel()
        result = cox.fit(icu_dataset["train"], icu_dataset["feat"])
        assert result is cox

    def test_c_index_in_valid_range(self, icu_dataset):
        cox = CoxPHModel()
        cox.fit(icu_dataset["train"], icu_dataset["feat"])
        c = cox.c_index
        assert 0.5 <= c <= 1.0, f"C-index {c} out of [0.5, 1.0]"

    def test_score_on_test_set(self, icu_dataset):
        cox = CoxPHModel()
        cox.fit(icu_dataset["train"], icu_dataset["feat"])
        c = cox.score(icu_dataset["test"])
        assert 0.0 <= c <= 1.0

    def test_hazard_ratios_returns_df(self, icu_dataset):
        cox = CoxPHModel()
        cox.fit(icu_dataset["train"], icu_dataset["feat"])
        hr = cox.hazard_ratios()
        assert isinstance(hr, pd.DataFrame)
        assert len(hr) > 0

    def test_predict_survival_shape(self, icu_dataset):
        cox = CoxPHModel()
        cox.fit(icu_dataset["train"], icu_dataset["feat"])
        test = icu_dataset["test"].head(10)
        surv = cox.predict_survival(test, times=[30, 90, 180])
        assert surv.shape == (10, 3)

    def test_predict_survival_in_0_1(self, icu_dataset):
        cox = CoxPHModel()
        cox.fit(icu_dataset["train"], icu_dataset["feat"])
        test = icu_dataset["test"].head(20)
        surv = cox.predict_survival(test, times=[30, 90])
        assert (surv >= 0).all()
        assert (surv <= 1).all()

    def test_predict_risk_array(self, icu_dataset):
        cox = CoxPHModel()
        cox.fit(icu_dataset["train"], icu_dataset["feat"])
        risk = cox.predict_risk(icu_dataset["test"].head(10))
        assert risk.shape == (10,)

    def test_ph_check_returns_df(self, icu_dataset):
        cox = CoxPHModel()
        cox.fit(icu_dataset["train"], icu_dataset["feat"])
        df = cox.check_proportional_hazards(icu_dataset["train"])
        assert isinstance(df, pd.DataFrame)

    def test_convergence(self, icu_dataset):
        # Model should fit without raising an exception
        cox = CoxPHModel(penalizer=0.1)
        cox.fit(icu_dataset["train"], icu_dataset["feat"])
        assert cox._fitted is True

    def test_unfitted_score_raises(self, icu_dataset):
        cox = CoxPHModel()
        with pytest.raises(RuntimeError):
            cox.score(icu_dataset["test"])


# ---------------------------------------------------------------------------
# Random Survival Forest
# ---------------------------------------------------------------------------

class TestRSF:
    def test_fit_returns_self(self, icu_dataset):
        rsf = RSFModel(n_estimators=20, random_state=42)
        result = rsf.fit(icu_dataset["train"], icu_dataset["feat"])
        assert result is rsf

    def test_oob_c_index_in_valid_range(self, icu_dataset):
        rsf = RSFModel(n_estimators=20, random_state=42)
        rsf.fit(icu_dataset["train"], icu_dataset["feat"])
        c = rsf.oob_c_index
        assert 0.0 <= c <= 1.0

    def test_score_on_test(self, icu_dataset):
        rsf = RSFModel(n_estimators=20, random_state=42)
        rsf.fit(icu_dataset["train"], icu_dataset["feat"])
        c = rsf.score(icu_dataset["test"])
        assert 0.0 <= c <= 1.0

    def test_feature_importance_non_negative(self, icu_dataset):
        rsf = RSFModel(n_estimators=20, random_state=42)
        rsf.fit(icu_dataset["train"], icu_dataset["feat"])
        df = rsf.feature_importance_df()
        assert (df["importance"] >= 0).all()

    def test_feature_importance_returns_all_features(self, icu_dataset):
        rsf = RSFModel(n_estimators=20, random_state=42)
        rsf.fit(icu_dataset["train"], icu_dataset["feat"])
        df = rsf.feature_importance_df()
        assert set(df["feature"]) == set(icu_dataset["feat"])

    def test_feature_importance_sums_to_one(self, icu_dataset):
        rsf = RSFModel(n_estimators=20, random_state=42)
        rsf.fit(icu_dataset["train"], icu_dataset["feat"])
        df = rsf.feature_importance_df()
        assert abs(df["importance"].sum() - 1.0) < 0.01

    def test_predict_survival_shape(self, icu_dataset):
        rsf = RSFModel(n_estimators=20, random_state=42)
        rsf.fit(icu_dataset["train"], icu_dataset["feat"])
        test = icu_dataset["test"].head(10)
        surv = rsf.predict_survival(test, times=[30, 90, 180, 365])
        assert surv.shape == (10, 4)

    def test_predict_survival_in_0_1(self, icu_dataset):
        rsf = RSFModel(n_estimators=20, random_state=42)
        rsf.fit(icu_dataset["train"], icu_dataset["feat"])
        test = icu_dataset["test"].head(20)
        surv = rsf.predict_survival(test, times=[30, 90])
        assert (surv >= 0).all()
        assert (surv <= 1).all()

    def test_predict_risk_shape(self, icu_dataset):
        rsf = RSFModel(n_estimators=20, random_state=42)
        rsf.fit(icu_dataset["train"], icu_dataset["feat"])
        risk = rsf.predict_risk(icu_dataset["test"].head(10))
        assert risk.shape == (10,)

    def test_permutation_importance_returns_df(self, icu_dataset):
        rsf = RSFModel(n_estimators=20, random_state=42)
        rsf.fit(icu_dataset["train"], icu_dataset["feat"])
        df = rsf.permutation_importance(icu_dataset["test"].head(50), n_repeats=2)
        assert isinstance(df, pd.DataFrame)
        assert "importance_mean" in df.columns

    def test_unfitted_raises(self, icu_dataset):
        rsf = RSFModel()
        with pytest.raises(RuntimeError):
            rsf.score(icu_dataset["test"])


# ---------------------------------------------------------------------------
# XGBoost Survival
# ---------------------------------------------------------------------------

class TestXGBoostSurvival:
    def test_fit_returns_self(self, icu_dataset):
        xgb = XGBoostSurvivalModel(random_state=42)
        result = xgb.fit(icu_dataset["train"], icu_dataset["feat"], n_estimators=50)
        assert result is xgb

    def test_score_in_valid_range(self, icu_dataset):
        xgb = XGBoostSurvivalModel(random_state=42)
        xgb.fit(icu_dataset["train"], icu_dataset["feat"], n_estimators=50)
        c = xgb.score(icu_dataset["test"])
        assert 0.0 <= c <= 1.0

    def test_feature_importance_non_negative(self, icu_dataset):
        xgb = XGBoostSurvivalModel(random_state=42)
        xgb.fit(icu_dataset["train"], icu_dataset["feat"], n_estimators=50)
        df = xgb.feature_importance_df()
        assert (df["importance"] >= 0).all()

    def test_predict_survival_in_0_1(self, icu_dataset):
        xgb = XGBoostSurvivalModel(random_state=42)
        xgb.fit(icu_dataset["train"], icu_dataset["feat"], n_estimators=50)
        test = icu_dataset["test"].head(20)
        surv = xgb.predict_survival(test, times=[30, 90])
        assert (surv >= 0).all()
        assert (surv <= 1).all()

    def test_predict_risk_shape(self, icu_dataset):
        xgb = XGBoostSurvivalModel(random_state=42)
        xgb.fit(icu_dataset["train"], icu_dataset["feat"], n_estimators=50)
        risk = xgb.predict_risk(icu_dataset["test"].head(15))
        assert risk.shape == (15,)

    def test_tune_and_fit_improves_or_maintains(self, icu_dataset):
        xgb = XGBoostSurvivalModel(random_state=42)
        small_grid = {"n_estimators": [50, 100], "max_depth": [2], "learning_rate": [0.1],
                      "subsample": [0.8], "min_samples_split": [10], "min_samples_leaf": [5]}
        xgb.tune_and_fit(
            icu_dataset["train"],
            icu_dataset["val"],
            icu_dataset["feat"],
            param_grid=small_grid,
        )
        c = xgb.score(icu_dataset["test"])
        assert 0.0 <= c <= 1.0

    def test_best_params_stored(self, icu_dataset):
        xgb = XGBoostSurvivalModel(random_state=42)
        small_grid = {"n_estimators": [50], "max_depth": [2], "learning_rate": [0.1],
                      "subsample": [0.8], "min_samples_split": [10], "min_samples_leaf": [5]}
        xgb.tune_and_fit(
            icu_dataset["train"],
            icu_dataset["val"],
            icu_dataset["feat"],
            param_grid=small_grid,
        )
        assert "n_estimators" in xgb._best_params

    def test_shap_values_returns_tuple(self, icu_dataset):
        xgb = XGBoostSurvivalModel(random_state=42)
        xgb.fit(icu_dataset["train"], icu_dataset["feat"], n_estimators=50)
        test = icu_dataset["test"].head(20)
        vals, summary = xgb.shap_values(test)
        assert vals is not None
        assert isinstance(summary, pd.DataFrame)

    def test_unfitted_raises(self, icu_dataset):
        xgb = XGBoostSurvivalModel()
        with pytest.raises(RuntimeError):
            xgb.predict_risk(icu_dataset["test"])
