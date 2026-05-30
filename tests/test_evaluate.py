"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from src.data import generate_synthetic_icu, engineer_features, preprocess, split_dataset, get_feature_columns
from src.evaluate import (
    _to_surv_array,
    compute_brier_scores,
    compute_c_index,
    compute_calibration,
    compute_time_auc,
    evaluate_model,
    build_comparison_table,
)
from src.models.random_survival_forest import RSFModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_rsf():
    raw = generate_synthetic_icu(n_patients=400, seed=1)
    raw = engineer_features(raw, dataset="icu")
    train_raw, _, test_raw = split_dataset(raw, seed=1)
    feat = get_feature_columns(train_raw)
    train, feat, scaler = preprocess(train_raw, feature_cols=feat)
    test, _, _ = preprocess(test_raw, feature_cols=feat, scaler=scaler, fit_scaler=False)

    rsf = RSFModel(n_estimators=30, random_state=1)
    rsf.fit(train, feat)

    y_train = _to_surv_array(train, "event", "duration")
    y_test = _to_surv_array(test, "event", "duration")
    risk = rsf.predict_risk(test)
    surv = rsf.predict_survival(test, times=[30, 90, 180, 365])

    return {
        "rsf": rsf,
        "train": train,
        "test": test,
        "y_train": y_train,
        "y_test": y_test,
        "risk": risk,
        "surv": surv,
        "feat": feat,
    }


# ---------------------------------------------------------------------------
# C-index
# ---------------------------------------------------------------------------

class TestCIndex:
    def test_c_index_in_0_1(self, fitted_rsf):
        c = compute_c_index(fitted_rsf["y_test"], fitted_rsf["risk"])
        assert 0.0 <= c <= 1.0

    def test_c_index_float(self, fitted_rsf):
        c = compute_c_index(fitted_rsf["y_test"], fitted_rsf["risk"])
        assert isinstance(c, float)

    def test_perfect_risk_scores_c_index_above_0_5(self, fitted_rsf):
        # Perfect risk score = actual survival time (inverted)
        times = fitted_rsf["y_test"]["time"]
        perfect_risk = -times  # higher risk for shorter survival
        c = compute_c_index(fitted_rsf["y_test"], perfect_risk)
        assert c >= 0.5


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_returns_dataframe(self, fitted_rsf):
        df = compute_brier_scores(
            fitted_rsf["y_train"], fitted_rsf["y_test"], fitted_rsf["surv"]
        )
        assert isinstance(df, pd.DataFrame)

    def test_has_ibs_row(self, fitted_rsf):
        df = compute_brier_scores(
            fitted_rsf["y_train"], fitted_rsf["y_test"], fitted_rsf["surv"]
        )
        assert "IBS" in df["time"].values

    def test_brier_scores_in_0_1(self, fitted_rsf):
        df = compute_brier_scores(
            fitted_rsf["y_train"], fitted_rsf["y_test"], fitted_rsf["surv"]
        )
        numeric_rows = df[df["time"] != "IBS"]
        assert (numeric_rows["brier_score"] >= 0).all()
        assert (numeric_rows["brier_score"] <= 1).all()


# ---------------------------------------------------------------------------
# Time-dependent AUC
# ---------------------------------------------------------------------------

class TestTimeAUC:
    def test_returns_dataframe(self, fitted_rsf):
        df = compute_time_auc(fitted_rsf["y_train"], fitted_rsf["y_test"], fitted_rsf["risk"])
        assert isinstance(df, pd.DataFrame)

    def test_auc_values_in_0_1(self, fitted_rsf):
        df = compute_time_auc(fitted_rsf["y_train"], fitted_rsf["y_test"], fitted_rsf["risk"])
        if len(df) > 0:
            assert (df["auc"] >= 0).all()
            assert (df["auc"] <= 1).all()


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_returns_dataframe(self, fitted_rsf):
        df = compute_calibration(fitted_rsf["y_test"], fitted_rsf["surv"])
        assert isinstance(df, pd.DataFrame)

    def test_predicted_in_0_1(self, fitted_rsf):
        df = compute_calibration(fitted_rsf["y_test"], fitted_rsf["surv"])
        if len(df) > 0:
            assert (df["predicted"] >= 0).all()
            assert (df["predicted"] <= 1).all()

    def test_observed_in_0_1(self, fitted_rsf):
        df = compute_calibration(fitted_rsf["y_test"], fitted_rsf["surv"])
        if len(df) > 0:
            assert (df["observed"] >= 0).all()
            assert (df["observed"] <= 1).all()


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    def test_returns_dict(self, fitted_rsf):
        result = evaluate_model(
            "RSF",
            "icu",
            fitted_rsf["y_train"],
            fitted_rsf["y_test"],
            fitted_rsf["risk"],
            fitted_rsf["surv"],
        )
        assert isinstance(result, dict)

    def test_has_c_index(self, fitted_rsf):
        result = evaluate_model(
            "RSF",
            "icu",
            fitted_rsf["y_train"],
            fitted_rsf["y_test"],
            fitted_rsf["risk"],
            fitted_rsf["surv"],
        )
        assert "c_index" in result

    def test_c_index_valid(self, fitted_rsf):
        result = evaluate_model(
            "RSF",
            "icu",
            fitted_rsf["y_train"],
            fitted_rsf["y_test"],
            fitted_rsf["risk"],
            fitted_rsf["surv"],
        )
        assert 0.0 <= result["c_index"] <= 1.0

    def test_has_ibs(self, fitted_rsf):
        result = evaluate_model(
            "RSF",
            "icu",
            fitted_rsf["y_train"],
            fitted_rsf["y_test"],
            fitted_rsf["risk"],
            fitted_rsf["surv"],
        )
        assert "ibs" in result


# ---------------------------------------------------------------------------
# build_comparison_table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    def test_returns_dataframe(self, fitted_rsf):
        result = evaluate_model(
            "RSF", "icu",
            fitted_rsf["y_train"], fitted_rsf["y_test"],
            fitted_rsf["risk"], fitted_rsf["surv"],
        )
        df = build_comparison_table([result])
        assert isinstance(df, pd.DataFrame)

    def test_has_model_and_dataset_cols(self, fitted_rsf):
        result = evaluate_model(
            "RSF", "icu",
            fitted_rsf["y_train"], fitted_rsf["y_test"],
            fitted_rsf["risk"], fitted_rsf["surv"],
        )
        df = build_comparison_table([result])
        assert "model" in df.columns
        assert "dataset" in df.columns
