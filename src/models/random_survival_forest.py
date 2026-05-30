"""
Random Survival Forest model using scikit-survival.

Implements:
- RandomSurvivalForest with 100 trees
- Permutation-based feature importance
- Out-of-bag C-index
- Survival and cumulative hazard prediction
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


class RSFModel:
    """Random Survival Forest survival estimator.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the forest.
    min_samples_split : int
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int
        Minimum number of samples required at a leaf node.
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs).
    random_state : int
        Random seed for reproducibility.
    duration_col : str
        Column name for follow-up time.
    event_col : str
        Column name for event indicator.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        n_jobs: int = -1,
        random_state: int = 42,
        duration_col: str = "duration",
        event_col: str = "event",
    ) -> None:
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.duration_col = duration_col
        self.event_col = event_col

        self.model: RandomSurvivalForest | None = None
        self.feature_cols: list[str] = []
        self._oob_score: float | None = None
        self._feature_importances: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_structured_array(self, df: pd.DataFrame) -> np.ndarray:
        """Convert duration/event columns to scikit-survival structured array."""
        return Surv.from_arrays(
            event=df[self.event_col].astype(bool),
            time=df[self.duration_col].astype(float),
        )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        train: pd.DataFrame,
        feature_cols: list[str],
    ) -> "RSFModel":
        """Fit the RSF on training data.

        Parameters
        ----------
        train : pd.DataFrame
        feature_cols : list[str]

        Returns
        -------
        self
        """
        self.feature_cols = list(feature_cols)
        X = train[self.feature_cols].values
        y = self._make_structured_array(train)

        self.model = RandomSurvivalForest(
            n_estimators=self.n_estimators,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            oob_score=True,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self.model.fit(X, y)

        self._oob_score = float(self.model.oob_score_)
        # feature_importances_ is not implemented in all sksurv versions;
        # compute permutation importance on the training set as a fallback.
        try:
            self._feature_importances = self.model.feature_importances_
        except NotImplementedError:
            self._feature_importances = self._compute_perm_importance_train(X, y)
        self._fitted = True
        return self

    def _compute_perm_importance_train(
        self, X: np.ndarray, y: np.ndarray, n_repeats: int = 3
    ) -> np.ndarray:
        """Compute permutation-based feature importance on training data."""
        rng = np.random.default_rng(self.random_state)
        baseline = float(self.model.score(X, y))
        importances = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            drops = []
            for _ in range(n_repeats):
                X_p = X.copy()
                X_p[:, j] = rng.permutation(X_p[:, j])
                drops.append(baseline - float(self.model.score(X_p, y)))
            importances[j] = max(0.0, float(np.mean(drops)))
        # Normalise so importances sum to 1 (matches sklearn convention)
        total = importances.sum()
        if total > 0:
            importances = importances / total
        return importances

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @property
    def oob_c_index(self) -> float:
        """Out-of-bag concordance index (computed during training)."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        return self._oob_score

    def score(self, df: pd.DataFrame) -> float:
        """Harrell's C-index on an external dataset.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        float
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = df[self.feature_cols].values
        y = self._make_structured_array(df)
        return float(self.model.score(X, y))

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance_df(self) -> pd.DataFrame:
        """Return sorted feature importances as a DataFrame.

        Returns
        -------
        pd.DataFrame with columns [feature, importance]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        df = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": self._feature_importances,
            }
        )
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def permutation_importance(
        self, df: pd.DataFrame, n_repeats: int = 5, seed: int = 42
    ) -> pd.DataFrame:
        """Compute permutation-based feature importance.

        Each feature is permuted n_repeats times; importance is the mean
        drop in C-index versus the baseline score.

        Parameters
        ----------
        df : pd.DataFrame
        n_repeats : int
        seed : int

        Returns
        -------
        pd.DataFrame with columns [feature, importance_mean, importance_std]
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        rng = np.random.default_rng(seed)
        X = df[self.feature_cols].values.copy()
        y = self._make_structured_array(df)
        baseline = float(self.model.score(X, y))

        results = []
        for j, feat in enumerate(self.feature_cols):
            drops = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                score_perm = float(self.model.score(X_perm, y))
                drops.append(baseline - score_perm)
            results.append(
                {
                    "feature": feat,
                    "importance_mean": float(np.mean(drops)),
                    "importance_std": float(np.std(drops)),
                }
            )

        return (
            pd.DataFrame(results)
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

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
        surv_fns = self.model.predict_survival_function(X, return_array=False)
        n = len(surv_fns)
        result = np.zeros((n, len(times)))
        for i, fn in enumerate(surv_fns):
            for j, t in enumerate(times):
                t_clipped = float(np.clip(t, fn.x.min(), fn.x.max()))
                result[i, j] = float(fn(t_clipped))
        return result

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        """Predict risk score (negative expected survival time) for each patient.

        Higher score = higher risk.

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
