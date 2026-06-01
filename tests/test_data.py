"""Tests for data loading, generation, and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.data import (
    engineer_features,
    generate_synthetic_icu,
    load_gbsg2,
    load_whas500,
    preprocess,
    prepare_dataset,
    split_dataset,
)

# ---------------------------------------------------------------------------
# WHAS500 loading
# ---------------------------------------------------------------------------

class TestLoadWhas500:
    def test_returns_dataframe(self):
        df = load_whas500()
        assert isinstance(df, pd.DataFrame)

    def test_has_duration_column(self):
        df = load_whas500()
        assert "duration" in df.columns

    def test_has_event_column(self):
        df = load_whas500()
        assert "event" in df.columns

    def test_duration_positive(self):
        df = load_whas500()
        assert (df["duration"] > 0).all()

    def test_event_binary(self):
        df = load_whas500()
        assert set(df["event"].unique()).issubset({0, 1})

    def test_non_empty(self):
        df = load_whas500()
        assert len(df) > 0

    def test_no_all_nan_columns(self):
        df = load_whas500()
        for col in df.columns:
            assert not df[col].isna().all(), f"Column {col} is all NaN"


# ---------------------------------------------------------------------------
# GBSG2 loading
# ---------------------------------------------------------------------------

class TestLoadGbsg2:
    def test_returns_dataframe(self):
        df = load_gbsg2()
        assert isinstance(df, pd.DataFrame)

    def test_has_duration_column(self):
        df = load_gbsg2()
        assert "duration" in df.columns

    def test_has_event_column(self):
        df = load_gbsg2()
        assert "event" in df.columns

    def test_duration_positive(self):
        df = load_gbsg2()
        assert (df["duration"] > 0).all()

    def test_event_binary(self):
        df = load_gbsg2()
        assert set(df["event"].unique()).issubset({0, 1})

    def test_non_empty(self):
        df = load_gbsg2()
        assert len(df) > 0


# ---------------------------------------------------------------------------
# Synthetic ICU generator
# ---------------------------------------------------------------------------

class TestSyntheticICU:
    def test_returns_dataframe(self):
        df = generate_synthetic_icu(n_patients=100)
        assert isinstance(df, pd.DataFrame)

    def test_exact_row_count(self):
        df = generate_synthetic_icu(n_patients=200)
        assert len(df) == 200

    def test_reproducibility_same_seed(self):
        df1 = generate_synthetic_icu(n_patients=50, seed=7)
        df2 = generate_synthetic_icu(n_patients=50, seed=7)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_synthetic_icu(n_patients=50, seed=1)
        df2 = generate_synthetic_icu(n_patients=50, seed=2)
        assert not df1["duration"].equals(df2["duration"])

    def test_age_in_valid_range(self):
        df = generate_synthetic_icu(n_patients=500)
        assert (df["age"] >= 18).all()
        assert (df["age"] <= 99).all()

    def test_duration_positive(self):
        df = generate_synthetic_icu(n_patients=500)
        assert (df["duration"] > 0).all()

    def test_event_binary(self):
        df = generate_synthetic_icu(n_patients=500)
        assert set(df["event"].unique()).issubset({0, 1})

    def test_approximate_mortality_rate(self):
        # MIMIC-III calibration: ~12% 30-day mortality
        # With censoring at random up to 365d, observed event rate should be moderate
        df = generate_synthetic_icu(n_patients=2000, seed=42)
        rate = df["event"].mean()
        # Allow generous range: calibration is approximate
        assert 0.05 < rate < 0.95

    def test_required_columns_present(self):
        df = generate_synthetic_icu(n_patients=50)
        required = {"duration", "event", "age", "los", "sofa_score", "creatinine", "lactate"}
        assert required.issubset(set(df.columns))

    def test_cci_non_negative(self):
        df = generate_synthetic_icu(n_patients=200)
        assert (df["cci"] >= 0).all()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_adds_age_group(self):
        df = generate_synthetic_icu(n_patients=100)
        out = engineer_features(df, dataset="icu")
        assert "age_group" in out.columns

    def test_adds_cci_level(self):
        df = generate_synthetic_icu(n_patients=100)
        out = engineer_features(df, dataset="icu")
        assert "cci_level" in out.columns

    def test_adds_los_category(self):
        df = generate_synthetic_icu(n_patients=100)
        out = engineer_features(df, dataset="icu")
        assert "los_category" in out.columns

    def test_original_df_not_mutated(self):
        df = generate_synthetic_icu(n_patients=50)
        original_cols = set(df.columns)
        engineer_features(df, dataset="icu")
        assert set(df.columns) == original_cols


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_returns_tuple(self):
        df = generate_synthetic_icu(n_patients=100)
        result = preprocess(df)
        assert isinstance(result, tuple) and len(result) == 3

    def test_no_nans_after_imputation(self):
        df = generate_synthetic_icu(n_patients=100)
        # Introduce some NaN
        df.loc[0, "age"] = np.nan
        processed, feat_cols, _ = preprocess(df)
        assert processed[feat_cols].isna().sum().sum() == 0

    def test_scaler_fitted(self):
        from sklearn.preprocessing import StandardScaler
        df = generate_synthetic_icu(n_patients=100)
        _, _, scaler = preprocess(df)
        assert isinstance(scaler, StandardScaler)

    def test_val_test_use_train_scaler(self):
        df = generate_synthetic_icu(n_patients=300)
        train, val, test = split_dataset(df)
        _, feat_cols, scaler = preprocess(train)
        val_proc, _, _ = preprocess(val, feature_cols=feat_cols, scaler=scaler, fit_scaler=False)
        assert not val_proc[feat_cols].isna().any().any()


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

class TestSplitDataset:
    def test_sizes_approx(self):
        df = generate_synthetic_icu(n_patients=1000)
        train, val, test = split_dataset(df)
        total = len(train) + len(val) + len(test)
        assert total == len(df)
        assert 0.65 < len(train) / total < 0.75

    def test_no_overlap_train_test(self):
        df = generate_synthetic_icu(n_patients=500)
        df = df.reset_index()  # ensure index is unique
        train, val, test = split_dataset(df)
        # Reset index added 'index' column; check no shared original indices
        train_idx = set(train["index"])
        test_idx = set(test["index"])
        assert len(train_idx & test_idx) == 0


# ---------------------------------------------------------------------------
# prepare_dataset high-level
# ---------------------------------------------------------------------------

class TestPrepareDataset:
    def test_icu(self):
        result = prepare_dataset("icu", n_icu=200)
        assert "train" in result
        assert "test" in result
        assert "feature_cols" in result

    def test_whas500(self):
        result = prepare_dataset("whas500")
        assert len(result["train"]) > 0

    def test_gbsg2(self):
        result = prepare_dataset("gbsg2")
        assert len(result["train"]) > 0

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            prepare_dataset("nonexistent")
