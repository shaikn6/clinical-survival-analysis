"""
Cox Proportional Hazards model using lifelines.

Implements:
- CoxPHFitter with regularisation
- Proportional hazards assumption check (Schoenfeld residuals)
- Hazard ratios table with confidence intervals and p-values
- C-index (concordance index)
- Survival probability prediction for new patients
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test


class CoxPHModel:
    """Cox Proportional Hazards survival model.

    Parameters
    ----------
    duration_col : str
        Column name for follow-up time.
    event_col : str
        Column name for event indicator (1 = event, 0 = censored).
    penalizer : float
        L2 regularisation strength.
    label : str
        Human-readable label for this model instance.
    """

    def __init__(
        self,
        duration_col: str = "duration",
        event_col: str = "event",
        penalizer: float = 0.1,
        label: str = "CoxPH",
    ) -> None:
        self.duration_col = duration_col
        self.event_col = event_col
        self.penalizer = penalizer
        self.label = label

        self.fitter: CoxPHFitter | None = None
        self.feature_cols: list[str] = []
        self._c_index: float | None = None
        self._ph_test_result = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        train: pd.DataFrame,
        feature_cols: list[str],
    ) -> "CoxPHModel":
        """Fit the Cox PH model on training data.

        Parameters
        ----------
        train : pd.DataFrame
            Training set containing duration, event, and feature columns.
        feature_cols : list[str]
            Numeric feature columns to include.

        Returns
        -------
        self
        """
        self.feature_cols = feature_cols
        cols = [self.duration_col, self.event_col] + list(feature_cols)

        df_fit = train[cols].copy()
        # Drop rows with any NaN in selected columns
        df_fit = df_fit.dropna(subset=cols)

        self.fitter = CoxPHFitter(penalizer=self.penalizer)
        self.fitter.fit(
            df_fit,
            duration_col=self.duration_col,
            event_col=self.event_col,
            show_progress=False,
        )
        self._c_index = float(self.fitter.concordance_index_)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def check_proportional_hazards(
        self, df: pd.DataFrame, p_threshold: float = 0.05
    ) -> pd.DataFrame:
        """Test proportional hazards assumption via Schoenfeld residuals.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to compute residuals on (training set recommended).
        p_threshold : float
            P-value below which the assumption is flagged as violated.

        Returns
        -------
        pd.DataFrame with columns [feature, test_statistic, p_value, assumption_holds]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before check_proportional_hazards().")

        cols = [self.duration_col, self.event_col] + list(self.feature_cols)
        df_test = df[cols].dropna(subset=cols)

        try:
            result = proportional_hazard_test(
                self.fitter, df_test, time_transform="rank"
            )
            ph_df = result.summary.copy()
            ph_df = ph_df.reset_index()
            ph_df.columns = [c.lower().replace(" ", "_") for c in ph_df.columns]
            if "covariate" not in ph_df.columns and ph_df.columns[0] != "covariate":
                ph_df = ph_df.rename(columns={ph_df.columns[0]: "covariate"})
            ph_df["assumption_holds"] = ph_df.get("p", ph_df.get("p_value", pd.Series())) > p_threshold
            self._ph_test_result = ph_df
            return ph_df
        except Exception as exc:
            warnings.warn(f"PH test failed: {exc}", stacklevel=2)
            return pd.DataFrame(columns=["covariate", "test_statistic", "p_value", "assumption_holds"])

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def hazard_ratios(self) -> pd.DataFrame:
        """Return hazard ratio table with CIs and p-values.

        Returns
        -------
        pd.DataFrame with columns [feature, HR, lower_95, upper_95, p_value]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before hazard_ratios().")

        summary = self.fitter.summary.copy().reset_index()
        summary.columns = [c.lower().replace(" ", "_") for c in summary.columns]

        # Rename to canonical names
        rename_map = {
            "covariate": "feature",
            "exp(coef)": "hr",
            "exp(coef)_lower_95%": "lower_95",
            "exp(coef)_upper_95%": "upper_95",
            "p": "p_value",
        }
        # lifelines uses various column names across versions
        for old, new in rename_map.items():
            if old in summary.columns:
                summary = summary.rename(columns={old: new})

        # Ensure we have feature column
        if "feature" not in summary.columns and len(summary.columns) > 0:
            summary = summary.rename(columns={summary.columns[0]: "feature"})

        keep = [c for c in ["feature", "hr", "coef", "lower_95", "upper_95", "p_value"] if c in summary.columns]
        return summary[keep].sort_values("p_value") if "p_value" in summary.columns else summary[keep]

    @property
    def c_index(self) -> float:
        """Concordance index on training data."""
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing c_index.")
        return self._c_index

    def score(self, df: pd.DataFrame) -> float:
        """Compute Harrell's concordance index (C-index) on a validation/test set.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        float  C-index in [0, 1]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before score().")
        from sksurv.metrics import concordance_index_censored
        risk = self.predict_risk(df)
        times = df[self.duration_col].astype(float).values
        events = df[self.event_col].astype(bool).values
        c, _, _, _, _ = concordance_index_censored(events, times, risk)
        return float(c)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_survival(
        self, df: pd.DataFrame, times: list[float]
    ) -> np.ndarray:
        """Predict survival probability at given times.

        Parameters
        ----------
        df : pd.DataFrame
            Patient feature rows (must contain feature_cols).
        times : list[float]
            Time points.

        Returns
        -------
        np.ndarray of shape (n_patients, n_times)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_survival().")
        feat_df = df[self.feature_cols].copy()
        sf = self.fitter.predict_survival_function(feat_df, times=times)
        # sf is (n_times, n_patients); transpose to (n_patients, n_times)
        return sf.values.T

    def predict_median(self, df: pd.DataFrame) -> np.ndarray:
        """Predict median survival time for each patient.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_patients,)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_median().")
        feat_df = df[self.feature_cols].copy()
        return self.fitter.predict_median(feat_df).values

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Predict log-partial hazard (risk score) for each patient.

        Higher values indicate higher risk.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_patients,)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_risk().")
        feat_df = df[self.feature_cols].copy()
        return self.fitter.predict_log_partial_hazard(feat_df).values
