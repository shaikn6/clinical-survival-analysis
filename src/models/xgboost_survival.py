"""
Gradient Boosting survival model using scikit-survival.

Implements:
- GradientBoostingSurvivalAnalysis with hyperparameter tuning
- SHAP values for feature importance
- Survival probability and risk score prediction
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv


class XGBoostSurvivalModel:
    """Gradient Boosted survival trees.

    Uses scikit-survival's GradientBoostingSurvivalAnalysis which
    implements a Cox partial likelihood loss, equivalent to the AFT/Cox
    GBM approach.

    Parameters
    ----------
    duration_col : str
    event_col : str
    random_state : int
    """

    def __init__(
        self,
        duration_col: str = "duration",
        event_col: str = "event",
        random_state: int = 42,
    ) -> None:
        self.duration_col = duration_col
        self.event_col = event_col
        self.random_state = random_state

        self.model: GradientBoostingSurvivalAnalysis | None = None
        self.feature_cols: list[str] = []
        self._best_params: dict = {}
        self._feature_importances: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_structured_array(self, df: pd.DataFrame) -> np.ndarray:
        return Surv.from_arrays(
            event=df[self.event_col].astype(bool),
            time=df[self.duration_col].astype(float),
        )

    # ------------------------------------------------------------------
    # Hyperparameter tuning
    # ------------------------------------------------------------------

    def tune_and_fit(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        feature_cols: list[str],
        param_grid: dict | None = None,
    ) -> "XGBoostSurvivalModel":
        """Grid-search over param_grid, fit best model on train+val.

        Parameters
        ----------
        train : pd.DataFrame
        val : pd.DataFrame
        feature_cols : list[str]
        param_grid : dict | None
            If None, uses a sensible default grid.

        Returns
        -------
        self
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [100, 200],
                "max_depth": [2, 3],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8],
                "min_samples_split": [10],
                "min_samples_leaf": [5],
            }

        self.feature_cols = list(feature_cols)
        X_train = train[self.feature_cols].values
        y_train = self._make_structured_array(train)
        X_val = val[self.feature_cols].values
        y_val = self._make_structured_array(val)

        best_score = -np.inf
        best_params = list(ParameterGrid(param_grid))[0]

        for params in ParameterGrid(param_grid):
            try:
                mdl = GradientBoostingSurvivalAnalysis(
                    random_state=self.random_state,
                    **params,
                )
                mdl.fit(X_train, y_train)
                score = mdl.score(X_val, y_val)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                continue

        self._best_params = best_params
        return self.fit(train, feature_cols, **best_params)

    def fit(
        self,
        train: pd.DataFrame,
        feature_cols: list[str],
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
    ) -> "XGBoostSurvivalModel":
        """Fit the gradient boosted survival model.

        Parameters
        ----------
        train : pd.DataFrame
        feature_cols : list[str]
        n_estimators, max_depth, learning_rate, subsample : model hyperparams

        Returns
        -------
        self
        """
        self.feature_cols = list(feature_cols)
        X = train[self.feature_cols].values
        y = self._make_structured_array(train)

        self.model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        self._feature_importances = self.model.feature_importances_
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> float:
        """Harrell's C-index on an external dataset."""
        if not self._fitted:
            raise RuntimeError("Call fit() or tune_and_fit() first.")
        X = df[self.feature_cols].values
        y = self._make_structured_array(df)
        return float(self.model.score(X, y))

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance_df(self) -> pd.DataFrame:
        """Return sorted MDI feature importances."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        df = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": self._feature_importances,
            }
        )
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def shap_values(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Compute SHAP values using TreeExplainer.

        Falls back to gradient-based approximation if SHAP is unavailable.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        (shap_values_array, shap_summary_df)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        X = df[self.feature_cols].values

        try:
            import shap

            explainer = shap.TreeExplainer(self.model)
            shap_vals = explainer.shap_values(X)
            mean_abs = np.abs(shap_vals).mean(axis=0)
            summary = pd.DataFrame(
                {"feature": self.feature_cols, "mean_abs_shap": mean_abs}
            ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            return shap_vals, summary

        except Exception:
            # Fallback: use built-in MDI importance as surrogate
            warnings.warn(
                "SHAP computation failed; using MDI importance as fallback.", stacklevel=2
            )
            imp = self._feature_importances
            n = len(df)
            shap_vals = np.tile(imp, (n, 1))  # (n, n_features)
            summary = pd.DataFrame(
                {"feature": self.feature_cols, "mean_abs_shap": np.abs(shap_vals).mean(axis=0)}
            ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            return shap_vals, summary

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_survival(
        self, df: pd.DataFrame, times: list[float]
    ) -> np.ndarray:
        """Predict survival probability at specified time points.

        Parameters
        ----------
        df : pd.DataFrame
        times : list[float]

        Returns
        -------
        np.ndarray of shape (n_patients, n_times)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = df[self.feature_cols].values
        surv_fns = self.model.predict_survival_function(X)
        n = len(surv_fns)
        result = np.zeros((n, len(times)))
        for i, fn in enumerate(surv_fns):
            for j, t in enumerate(times):
                t_clipped = float(np.clip(t, fn.x.min(), fn.x.max()))
                result[i, j] = float(fn(t_clipped))
        return result

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Predict risk score for each patient (higher = higher risk).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_patients,)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = df[self.feature_cols].values
        return self.model.predict(X)
