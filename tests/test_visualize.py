"""Tests for visualization — verify output files are created."""

import pytest
import pandas as pd
import numpy as np

from src.data import generate_synthetic_icu, engineer_features, preprocess, split_dataset, get_feature_columns
from src.models.kaplan_meier import KaplanMeierModel
from src.models.random_survival_forest import RSFModel
from src.visualize import (
    plot_calibration,
    plot_feature_importance,
    plot_forest,
    plot_km_curves,
    plot_km_interactive,
    plot_model_comparison,
)


@pytest.fixture(scope="module")
def vis_fixtures():
    raw = generate_synthetic_icu(n_patients=200, seed=99)
    raw = engineer_features(raw, dataset="icu")
    train_raw, _, test_raw = split_dataset(raw, seed=99)
    feat = get_feature_columns(train_raw)
    train, feat, scaler = preprocess(train_raw, feature_cols=feat)
    test, _, _ = preprocess(test_raw, feature_cols=feat, scaler=scaler, fit_scaler=False)

    km = KaplanMeierModel()
    km.fit(train)
    km.fit_subgroups(train, "age_group")

    rsf = RSFModel(n_estimators=20, random_state=99)
    rsf.fit(train, feat)

    return {
        "km": km,
        "rsf": rsf,
        "train": train,
        "test": test,
        "feat": feat,
    }


class TestPlotKMCurves:
    def test_creates_file(self, vis_fixtures, tmp_path):
        km = vis_fixtures["km"]
        kmfs = {"overall": km.overall_kmf}
        out = plot_km_curves(kmfs, title="Test KM", save_path=tmp_path / "km.png")
        assert out.exists()

    def test_file_has_content(self, vis_fixtures, tmp_path):
        km = vis_fixtures["km"]
        kmfs = {"overall": km.overall_kmf}
        out = plot_km_curves(kmfs, save_path=tmp_path / "km2.png")
        assert out.stat().st_size > 1000


class TestPlotKMInteractive:
    def test_creates_html(self, vis_fixtures, tmp_path):
        km = vis_fixtures["km"]
        kmfs = {"overall": km.overall_kmf}
        out = plot_km_interactive(kmfs, save_path=tmp_path / "km.html")
        assert out.exists()
        assert out.suffix == ".html"

    def test_html_not_empty(self, vis_fixtures, tmp_path):
        km = vis_fixtures["km"]
        kmfs = {"overall": km.overall_kmf}
        out = plot_km_interactive(kmfs, save_path=tmp_path / "km2.html")
        assert out.stat().st_size > 500


class TestPlotForest:
    def test_creates_file(self, tmp_path):
        hr_df = pd.DataFrame({
            "feature": ["age", "sofa_score", "cci"],
            "hr": [1.4, 1.8, 1.2],
            "lower_95": [1.1, 1.3, 0.9],
            "upper_95": [1.8, 2.5, 1.6],
            "p_value": [0.01, 0.001, 0.1],
        })
        out = plot_forest(hr_df, save_path=tmp_path / "forest.png")
        assert out.exists()


class TestPlotFeatureImportance:
    def test_creates_file_rsf(self, vis_fixtures, tmp_path):
        imp = vis_fixtures["rsf"].feature_importance_df()
        out = plot_feature_importance(imp, model_name="RSF", save_path=tmp_path / "imp.png")
        assert out.exists()


class TestPlotCalibration:
    def test_creates_file(self, vis_fixtures, tmp_path):
        from src.evaluate import compute_calibration, _to_surv_array
        rsf = vis_fixtures["rsf"]
        test = vis_fixtures["test"]
        y_test = _to_surv_array(test, "event", "duration")
        surv = rsf.predict_survival(test, times=[30, 90, 180, 365])
        calib = compute_calibration(y_test, surv)
        if len(calib) == 0:
            pytest.skip("Not enough data for calibration bins.")
        out = plot_calibration(calib, model_name="RSF", save_path=tmp_path / "calib.png")
        assert out.exists()


class TestPlotModelComparison:
    def test_creates_file(self, tmp_path):
        df = pd.DataFrame({
            "model": ["KM", "Cox PH", "RSF", "XGB"] * 3,
            "dataset": ["whas500"] * 4 + ["gbsg2"] * 4 + ["icu"] * 4,
            "c_index": np.random.uniform(0.55, 0.80, 12),
        })
        out = plot_model_comparison(df, save_path=tmp_path / "cmp.png")
        assert out.exists()
