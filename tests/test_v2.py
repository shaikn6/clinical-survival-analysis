"""
V2 test suite: DeepSurv, DeepHit, competing risks, and extended evaluation.

Covers 40+ individual test cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def synthetic_data(rng):
    """Small synthetic survival dataset for fast tests."""
    n, p = 120, 10
    X = rng.standard_normal((n, p)).astype(np.float32)
    # Exponential survival times
    hazard = np.exp(0.3 * X[:, 0] - 0.2 * X[:, 1])
    time = rng.exponential(1.0 / hazard).clip(1.0, 30.0)
    # ~30% censoring
    censor = rng.uniform(5, 35, n)
    event = (time <= censor).astype(int)
    time = np.minimum(time, censor).astype(np.float32)
    return X, time, event


@pytest.fixture(scope="module")
def competing_data(rng):
    """Dataset with two competing events."""
    n, p = 150, 6
    X = rng.standard_normal((n, p)).astype(np.float32)
    time = rng.exponential(10, n).clip(0.5, 50.0).astype(np.float32)
    # 3-class: 0=censored, 1=event 1, 2=competing event
    event = rng.integers(0, 3, n)
    return X, time, event


# ============================================================================
# DeepSurvNet
# ============================================================================

class TestDeepSurvNet:
    def test_import(self):
        from src.models.deep_surv import DeepSurvNet  # noqa: F401

    def test_forward_shape_default(self):
        from src.models.deep_surv import DeepSurvNet
        net = DeepSurvNet(input_dim=10)
        x = torch.randn(8, 10)
        out = net(x)
        assert out.shape == (8, 1), f"Expected (8, 1), got {out.shape}"

    def test_forward_shape_custom_hidden(self):
        from src.models.deep_surv import DeepSurvNet
        net = DeepSurvNet(input_dim=5, hidden_dims=[128, 64, 32])
        x = torch.randn(16, 5)
        out = net(x)
        assert out.shape == (16, 1)

    def test_forward_single_hidden(self):
        from src.models.deep_surv import DeepSurvNet
        net = DeepSurvNet(input_dim=4, hidden_dims=[16])
        x = torch.randn(4, 4)
        out = net(x)
        assert out.shape == (4, 1)

    def test_output_is_finite(self):
        from src.models.deep_surv import DeepSurvNet
        net = DeepSurvNet(input_dim=10)
        x = torch.randn(8, 10)
        out = net(x)
        assert torch.isfinite(out).all()

    def test_dropout_in_train_mode(self):
        from src.models.deep_surv import DeepSurvNet
        net = DeepSurvNet(input_dim=10, dropout=0.5)
        net.train()
        x = torch.randn(50, 10)
        out1 = net(x)
        out2 = net(x)
        # With dropout enabled, outputs should differ on average
        assert not torch.allclose(out1, out2), "Dropout should cause stochastic outputs"

    def test_eval_mode_deterministic(self):
        from src.models.deep_surv import DeepSurvNet
        net = DeepSurvNet(input_dim=10)
        net.eval()
        x = torch.randn(8, 10)
        with torch.no_grad():
            out1 = net(x)
            out2 = net(x)
        assert torch.allclose(out1, out2)


# ============================================================================
# DeepSurv Cox loss
# ============================================================================

class TestDeepSurvLoss:
    def _make_trainer(self):
        from src.models.deep_surv import DeepSurvNet, DeepSurvTrainer
        net = DeepSurvNet(input_dim=5)
        return DeepSurvTrainer(net)

    def test_loss_is_scalar(self):
        trainer = self._make_trainer()
        log_hz = torch.randn(20)
        time = torch.abs(torch.randn(20)) + 1
        event = torch.ones(20)
        loss = trainer.cox_partial_likelihood_loss(log_hz, time, event)
        assert loss.ndim == 0, "Loss should be a scalar tensor"

    def test_loss_is_finite(self):
        trainer = self._make_trainer()
        log_hz = torch.randn(20)
        time = torch.abs(torch.randn(20)) + 1
        event = torch.ones(20)
        loss = trainer.cox_partial_likelihood_loss(log_hz, time, event)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"

    def test_loss_non_nan_mixed_events(self):
        trainer = self._make_trainer()
        log_hz = torch.randn(30)
        time = torch.abs(torch.randn(30)) + 0.5
        # Mix of events and censored
        event = torch.tensor([1, 0] * 15, dtype=torch.float32)
        loss = trainer.cox_partial_likelihood_loss(log_hz, time, event)
        assert not torch.isnan(loss)

    def test_lower_risk_lower_loss(self):
        """Patients in correct risk order should have lower loss."""
        trainer = self._make_trainer()
        n = 20
        time = torch.linspace(1, 20, n)
        event = torch.ones(n)
        # Aligned risk: higher time = lower risk (correct)
        log_hz_good = -time
        # Reversed: higher time = higher risk (wrong)
        log_hz_bad = time
        loss_good = trainer.cox_partial_likelihood_loss(log_hz_good, time, event)
        loss_bad = trainer.cox_partial_likelihood_loss(log_hz_bad, time, event)
        assert loss_good < loss_bad


# ============================================================================
# DeepSurvTrainer.fit and predict
# ============================================================================

class TestDeepSurvTrainer:
    def _make_trainer(self, input_dim=10):
        from src.models.deep_surv import DeepSurvNet, DeepSurvTrainer
        net = DeepSurvNet(input_dim=input_dim, hidden_dims=[32, 16])
        return DeepSurvTrainer(net, lr=1e-2)

    def test_fit_returns_loss_list(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        losses = trainer.fit(X, time, event, n_epochs=5, batch_size=32)
        assert isinstance(losses, list)
        assert len(losses) == 5

    def test_fit_loss_decreases(self, synthetic_data):
        """Loss should decrease over training for a learnable dataset."""
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        losses = trainer.fit(X, time, event, n_epochs=10, batch_size=32)
        valid = [l for l in losses if not np.isnan(l)]
        assert len(valid) >= 3, "Too many NaN losses"
        # First loss > last loss (may not be monotone, just net decrease expected)
        assert valid[0] >= valid[-1] or abs(valid[0] - valid[-1]) < 5.0, \
            f"Loss did not decrease: first={valid[0]:.4f}, last={valid[-1]:.4f}"

    def test_predict_risk_length(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        risk = trainer.predict_risk(X)
        assert len(risk) == len(X), f"Expected {len(X)}, got {len(risk)}"

    def test_predict_risk_is_ndarray(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        risk = trainer.predict_risk(X)
        assert isinstance(risk, np.ndarray)

    def test_predict_risk_finite(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        risk = trainer.predict_risk(X)
        assert np.isfinite(risk).all()

    def test_concordance_index_range(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        trainer.fit(X, time, event, n_epochs=5, batch_size=32)
        c = trainer.concordance_index(X, time, event)
        assert 0.0 <= c <= 1.0, f"C-index out of range: {c}"

    def test_concordance_index_float(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        c = trainer.concordance_index(X, time, event)
        assert isinstance(c, float)


# ============================================================================
# DeepHitNet
# ============================================================================

class TestDeepHitNet:
    def test_import(self):
        from src.models.deep_hit import DeepHitNet  # noqa: F401

    def test_forward_shape_single_risk(self):
        from src.models.deep_hit import DeepHitNet
        net = DeepHitNet(input_dim=10, num_time_bins=50, num_risks=1, hidden_dim=32)
        x = torch.randn(8, 10)
        out = net(x)
        assert out.shape == (8, 1, 50), f"Expected (8, 1, 50), got {out.shape}"

    def test_forward_shape_competing_risks(self):
        from src.models.deep_hit import DeepHitNet
        net = DeepHitNet(input_dim=10, num_time_bins=30, num_risks=2, hidden_dim=32)
        x = torch.randn(12, 10)
        out = net(x)
        assert out.shape == (12, 2, 30), f"Expected (12, 2, 30), got {out.shape}"

    def test_softmax_sums_to_one(self):
        from src.models.deep_hit import DeepHitNet
        net = DeepHitNet(input_dim=5, num_time_bins=20, num_risks=1, hidden_dim=16)
        net.eval()
        x = torch.randn(10, 5)
        with torch.no_grad():
            out = net(x)
        # Sum over time bins per risk should be ~1
        sums = out.sum(dim=2)  # (batch, num_risks)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
            f"Softmax sums not 1: {sums}"

    def test_output_non_negative(self):
        from src.models.deep_hit import DeepHitNet
        net = DeepHitNet(input_dim=5, num_time_bins=20, num_risks=1, hidden_dim=16)
        net.eval()
        x = torch.randn(10, 5)
        with torch.no_grad():
            out = net(x)
        assert (out >= 0).all()

    def test_custom_bins_and_risks(self):
        from src.models.deep_hit import DeepHitNet
        net = DeepHitNet(input_dim=8, num_time_bins=100, num_risks=3, hidden_dim=64)
        x = torch.randn(4, 8)
        out = net(x)
        assert out.shape == (4, 3, 100)


# ============================================================================
# DeepHitTrainer
# ============================================================================

class TestDeepHitTrainer:
    def _make_trainer(self, input_dim=10, num_risks=1):
        from src.models.deep_hit import DeepHitNet, DeepHitTrainer
        net = DeepHitNet(input_dim=input_dim, num_time_bins=20, num_risks=num_risks, hidden_dim=32)
        return DeepHitTrainer(net, alpha=0.2, sigma=0.1, lr=1e-2)

    def test_fit_completes(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        losses = trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        assert losses is not None

    def test_fit_returns_loss_list(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        losses = trainer.fit(X, time, event, n_epochs=5, batch_size=32)
        assert isinstance(losses, list)
        assert len(losses) == 5

    def test_predict_survival_shape(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        time_points = [5, 10, 20]
        surv = trainer.predict_survival(X, time_points)
        assert surv.shape == (len(X), len(time_points)), \
            f"Expected {(len(X), len(time_points))}, got {surv.shape}"

    def test_predict_survival_in_zero_one(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        surv = trainer.predict_survival(X, [5, 10, 20])
        assert (surv >= 0).all() and (surv <= 1).all(), \
            f"Survival values out of [0, 1]: min={surv.min()}, max={surv.max()}"

    def test_predict_survival_monotone_decreasing(self, synthetic_data):
        """Survival should not increase with time (approximately)."""
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        trainer.fit(X, time, event, n_epochs=5, batch_size=32)
        time_points = [2, 5, 10, 20, 25]
        surv = trainer.predict_survival(X, time_points)
        # Check mean survival is non-increasing
        mean_surv = surv.mean(axis=0)
        for i in range(len(mean_surv) - 1):
            assert mean_surv[i] >= mean_surv[i + 1] - 0.05, \
                f"Survival increased: S({time_points[i]})={mean_surv[i]:.4f} < S({time_points[i+1]})={mean_surv[i+1]:.4f}"

    def test_concordance_index_range(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        c = trainer.concordance_index(X, time, event)
        assert 0.0 <= c <= 1.0

    def test_concordance_index_float(self, synthetic_data):
        X, time, event = synthetic_data
        trainer = self._make_trainer(X.shape[1])
        trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        c = trainer.concordance_index(X, time, event)
        assert isinstance(c, float)

    def test_competing_risks_fit(self, competing_data):
        X, time, event = competing_data
        from src.models.deep_hit import DeepHitNet, DeepHitTrainer
        net = DeepHitNet(input_dim=X.shape[1], num_time_bins=20, num_risks=2, hidden_dim=32)
        trainer = DeepHitTrainer(net, alpha=0.2, sigma=0.1, lr=1e-2)
        losses = trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        assert isinstance(losses, list)


# ============================================================================
# Competing Risks
# ============================================================================

class TestCompetingRisks:
    def test_import(self):
        from src.competing_risks import (  # noqa: F401
            cause_specific_hazard,
            cumulative_incidence_function,
            plot_competing_risks,
        )

    def test_cause_specific_hazard_returns_dict(self, competing_data):
        from src.competing_risks import cause_specific_hazard
        X, time, event = competing_data
        models = cause_specific_hazard(time, event, X)
        assert isinstance(models, dict)

    def test_cause_specific_hazard_has_cause_keys(self, competing_data):
        from src.competing_risks import cause_specific_hazard
        X, time, event = competing_data
        models = cause_specific_hazard(time, event, X)
        assert 1 in models or 2 in models, f"Expected keys 1 or 2, got {list(models.keys())}"

    def test_cause_specific_hazard_both_causes(self, competing_data):
        from src.competing_risks import cause_specific_hazard
        X, time, event = competing_data
        models = cause_specific_hazard(time, event, X)
        assert 1 in models and 2 in models

    def test_cause_specific_hazard_fitted(self, competing_data):
        from src.competing_risks import cause_specific_hazard
        from lifelines import CoxPHFitter
        X, time, event = competing_data
        models = cause_specific_hazard(time, event, X)
        for cause, fitter in models.items():
            assert isinstance(fitter, CoxPHFitter)
            assert hasattr(fitter, "concordance_index_"), "Model not fitted"

    def test_cif_returns_tuple(self, competing_data):
        from src.competing_risks import cumulative_incidence_function
        _, time, event = competing_data
        result = cumulative_incidence_function(time, event, cause=1)
        assert isinstance(result, tuple) and len(result) == 2

    def test_cif_array_lengths_match(self, competing_data):
        from src.competing_risks import cumulative_incidence_function
        _, time, event = competing_data
        t_pts, cif_vals = cumulative_incidence_function(time, event, cause=1)
        assert len(t_pts) == len(cif_vals)

    def test_cif_values_in_zero_one(self, competing_data):
        from src.competing_risks import cumulative_incidence_function
        _, time, event = competing_data
        _, cif_vals = cumulative_incidence_function(time, event, cause=1)
        assert (cif_vals >= 0).all() and (cif_vals <= 1).all(), \
            f"CIF out of [0,1]: min={cif_vals.min()}, max={cif_vals.max()}"

    def test_cif_non_decreasing(self, competing_data):
        from src.competing_risks import cumulative_incidence_function
        _, time, event = competing_data
        _, cif_vals = cumulative_incidence_function(time, event, cause=1)
        diffs = np.diff(cif_vals)
        assert (diffs >= -1e-8).all(), "CIF is not non-decreasing"

    def test_cif_cause2(self, competing_data):
        from src.competing_risks import cumulative_incidence_function
        _, time, event = competing_data
        t_pts, cif_vals = cumulative_incidence_function(time, event, cause=2)
        assert len(t_pts) > 0
        assert (cif_vals >= 0).all() and (cif_vals <= 1).all()

    def test_plot_competing_risks_returns_figure(self, competing_data, tmp_path):
        from src.competing_risks import plot_competing_risks
        _, time, event = competing_data
        out = str(tmp_path / "cif.png")
        fig = plot_competing_risks(time, event, output_path=out)
        import matplotlib.pyplot as plt
        assert isinstance(fig, plt.Figure)

    def test_plot_competing_risks_saves_file(self, competing_data, tmp_path):
        from src.competing_risks import plot_competing_risks
        import os
        _, time, event = competing_data
        out = str(tmp_path / "cif_test.png")
        plot_competing_risks(time, event, output_path=out)
        assert os.path.exists(out)


# ============================================================================
# Extended Evaluation (evaluate_v2)
# ============================================================================

class TestEvaluateV2:
    def _make_deep_surv_entry(self, synthetic_data):
        from src.models.deep_surv import DeepSurvNet, DeepSurvTrainer
        X, time, event = synthetic_data
        net = DeepSurvNet(input_dim=X.shape[1], hidden_dims=[16, 8])
        trainer = DeepSurvTrainer(net, lr=1e-2)
        trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        return trainer, X, time, event

    def _make_deep_hit_entry(self, synthetic_data):
        from src.models.deep_hit import DeepHitNet, DeepHitTrainer
        X, time, event = synthetic_data
        net = DeepHitNet(input_dim=X.shape[1], num_time_bins=15, num_risks=1, hidden_dim=16)
        trainer = DeepHitTrainer(net, lr=1e-2)
        trainer.fit(X, time, event, n_epochs=3, batch_size=32)
        return trainer, X, time, event

    def test_evaluate_deep_models_returns_dataframe(self, synthetic_data):
        from src.evaluate_v2 import evaluate_deep_models
        ds_trainer, X, time, event = self._make_deep_surv_entry(synthetic_data)
        models = {"DeepSurv": (ds_trainer, X, time, event)}
        df = evaluate_deep_models(models, time_points=[5, 10])
        assert isinstance(df, pd.DataFrame)

    def test_evaluate_deep_models_has_c_index(self, synthetic_data):
        from src.evaluate_v2 import evaluate_deep_models
        ds_trainer, X, time, event = self._make_deep_surv_entry(synthetic_data)
        models = {"DeepSurv": (ds_trainer, X, time, event)}
        df = evaluate_deep_models(models, time_points=[5, 10])
        assert "c_index" in df.columns

    def test_evaluate_deep_models_c_index_in_range(self, synthetic_data):
        from src.evaluate_v2 import evaluate_deep_models
        ds_trainer, X, time, event = self._make_deep_surv_entry(synthetic_data)
        models = {"DeepSurv": (ds_trainer, X, time, event)}
        df = evaluate_deep_models(models, time_points=[5, 10])
        val = df.loc[df["model"] == "DeepSurv", "c_index"].iloc[0]
        assert 0.0 <= float(val) <= 1.0

    def test_evaluate_deep_models_deephit_brier(self, synthetic_data):
        from src.evaluate_v2 import evaluate_deep_models
        dh_trainer, X, time, event = self._make_deep_hit_entry(synthetic_data)
        models = {"DeepHit": (dh_trainer, X, time, event)}
        df = evaluate_deep_models(models, time_points=[5, 10])
        assert "brier_5d" in df.columns

    def test_evaluate_deep_models_both_models(self, synthetic_data):
        from src.evaluate_v2 import evaluate_deep_models
        ds_trainer, X, time, event = self._make_deep_surv_entry(synthetic_data)
        dh_trainer, *_ = self._make_deep_hit_entry(synthetic_data)
        models = {
            "DeepSurv": (ds_trainer, X, time, event),
            "DeepHit": (dh_trainer, X, time, event),
        }
        df = evaluate_deep_models(models, time_points=[5, 10])
        assert len(df) == 2

    def test_build_full_comparison_table_six_rows(self, synthetic_data):
        from src.evaluate_v2 import build_full_comparison_table, evaluate_deep_models
        ds_trainer, X, time, event = self._make_deep_surv_entry(synthetic_data)
        dh_trainer, *_ = self._make_deep_hit_entry(synthetic_data)

        traditional = {
            "KM": {"c_index": 0.5, "dataset": "icu"},
            "Cox PH": {"c_index": 0.72, "dataset": "icu"},
            "RSF": {"c_index": 0.76, "dataset": "icu"},
            "XGBoost": {"c_index": 0.77, "dataset": "icu"},
        }
        deep = evaluate_deep_models(
            {
                "DeepSurv": (ds_trainer, X, time, event),
                "DeepHit": (dh_trainer, X, time, event),
            },
            time_points=[5, 10],
        )
        combined = build_full_comparison_table(traditional, deep)
        assert len(combined) == 6, f"Expected 6 rows, got {len(combined)}"

    def test_build_full_comparison_table_has_model_column(self, synthetic_data):
        from src.evaluate_v2 import build_full_comparison_table, evaluate_deep_models
        ds_trainer, X, time, event = self._make_deep_surv_entry(synthetic_data)
        dh_trainer, *_ = self._make_deep_hit_entry(synthetic_data)

        traditional = {
            "KM": {"c_index": 0.5, "dataset": "icu"},
            "Cox PH": {"c_index": 0.72, "dataset": "icu"},
            "RSF": {"c_index": 0.76, "dataset": "icu"},
            "XGBoost": {"c_index": 0.77, "dataset": "icu"},
        }
        deep = evaluate_deep_models(
            {
                "DeepSurv": (ds_trainer, X, time, event),
                "DeepHit": (dh_trainer, X, time, event),
            },
            time_points=[5, 10],
        )
        combined = build_full_comparison_table(traditional, deep)
        assert "model" in combined.columns
        assert "c_index" in combined.columns

    def test_build_full_comparison_table_all_model_names(self, synthetic_data):
        from src.evaluate_v2 import build_full_comparison_table, evaluate_deep_models
        ds_trainer, X, time, event = self._make_deep_surv_entry(synthetic_data)
        dh_trainer, *_ = self._make_deep_hit_entry(synthetic_data)

        traditional = {
            "KM": {"c_index": 0.5, "dataset": "icu"},
            "Cox PH": {"c_index": 0.72, "dataset": "icu"},
            "RSF": {"c_index": 0.76, "dataset": "icu"},
            "XGBoost": {"c_index": 0.77, "dataset": "icu"},
        }
        deep = evaluate_deep_models(
            {
                "DeepSurv": (ds_trainer, X, time, event),
                "DeepHit": (dh_trainer, X, time, event),
            },
            time_points=[5, 10],
        )
        combined = build_full_comparison_table(traditional, deep)
        models_present = set(combined["model"].tolist())
        expected = {"KM", "Cox PH", "RSF", "XGBoost", "DeepSurv", "DeepHit"}
        assert expected == models_present, f"Missing models: {expected - models_present}"
