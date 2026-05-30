"""
Kaplan-Meier survival estimator with subgroup analysis.

Implements:
- KM curves per subgroup (age quartiles, comorbidity levels)
- Log-rank test between groups
- Median survival time with 95% CI
- Nelson-Aalen cumulative hazard estimator
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test


class KaplanMeierModel:
    """Kaplan-Meier survival estimator with subgroup comparison.

    Parameters
    ----------
    duration_col : str
        Column name for follow-up time.
    event_col : str
        Column name for event indicator (1 = event, 0 = censored).
    label : str
        Human-readable label for this dataset/cohort.
    """

    def __init__(
        self,
        duration_col: str = "duration",
        event_col: str = "event",
        label: str = "cohort",
    ) -> None:
        self.duration_col = duration_col
        self.event_col = event_col
        self.label = label

        self.overall_kmf: KaplanMeierFitter | None = None
        self.overall_naf: NelsonAalenFitter | None = None
        self.subgroup_kmfs: dict[str, KaplanMeierFitter] = {}
        self.logrank_results: dict[str, object] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "KaplanMeierModel":
        """Fit overall KM and Nelson-Aalen estimators.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing duration and event columns.

        Returns
        -------
        self
        """
        T = df[self.duration_col]
        E = df[self.event_col]

        self.overall_kmf = KaplanMeierFitter(label=self.label)
        self.overall_kmf.fit(T, event_observed=E)

        self.overall_naf = NelsonAalenFitter(label=self.label)
        self.overall_naf.fit(T, event_observed=E)

        self._fitted = True
        return self

    def fit_subgroups(
        self,
        df: pd.DataFrame,
        subgroup_col: str,
    ) -> "KaplanMeierModel":
        """Fit a KM curve per subgroup and run log-rank test.

        Parameters
        ----------
        df : pd.DataFrame
        subgroup_col : str
            Column whose unique values define the subgroups.

        Returns
        -------
        self
        """
        T = df[self.duration_col]
        E = df[self.event_col]
        groups = df[subgroup_col]
        unique_groups = sorted(df[subgroup_col].dropna().unique())

        self.subgroup_kmfs[subgroup_col] = {}
        for grp in unique_groups:
            mask = groups == grp
            kmf = KaplanMeierFitter(label=str(grp))
            kmf.fit(T[mask], event_observed=E[mask])
            self.subgroup_kmfs[subgroup_col][str(grp)] = kmf

        # Multi-group log-rank test
        result = multivariate_logrank_test(T, groups, event_observed=E)
        self.logrank_results[subgroup_col] = result

        return self

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    def median_survival(self) -> float:
        """Return overall median survival time."""
        if not self._fitted:
            raise RuntimeError("Call fit() before median_survival().")
        med = self.overall_kmf.median_survival_time_
        return float(med) if med is not None and not np.isnan(float(med)) else float("inf")

    def median_survival_ci(self) -> dict[str, float]:
        """Return 95% CI of the median survival time."""
        if not self._fitted:
            raise RuntimeError("Call fit() before median_survival_ci().")
        ci = self.overall_kmf.confidence_interval_cumulative_density_
        # lifelines stores timeline-indexed CI; extract median CI from table
        timeline = self.overall_kmf.timeline
        sf = self.overall_kmf.survival_function_.squeeze()

        ci_lower_col = [c for c in self.overall_kmf.confidence_interval_.columns if "lower" in c]
        ci_upper_col = [c for c in self.overall_kmf.confidence_interval_.columns if "upper" in c]

        def _find_crossing(series: pd.Series) -> float:
            idx = (series <= 0.5).idxmax()
            return float(idx) if series[idx] <= 0.5 else float("inf")

        lower_sf = self.overall_kmf.confidence_interval_[ci_upper_col[0]] if ci_upper_col else sf
        upper_sf = self.overall_kmf.confidence_interval_[ci_lower_col[0]] if ci_lower_col else sf

        return {
            "median": self.median_survival(),
            "ci_lower": _find_crossing(lower_sf),
            "ci_upper": _find_crossing(upper_sf),
        }

    def survival_at(self, times: list[float]) -> pd.DataFrame:
        """Return survival probability at specified times.

        Parameters
        ----------
        times : list[float]
            Time points to evaluate.

        Returns
        -------
        pd.DataFrame with columns ['time', 'survival_prob']
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before survival_at().")
        rows = []
        for t in times:
            prob = self.overall_kmf.predict(t)
            rows.append({"time": t, "survival_prob": float(prob)})
        return pd.DataFrame(rows)

    def logrank_summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all log-rank tests."""
        rows = []
        for subgroup_col, result in self.logrank_results.items():
            rows.append(
                {
                    "subgroup": subgroup_col,
                    "test_statistic": float(result.test_statistic),
                    "p_value": float(result.p_value),
                    "significant": result.p_value < 0.05,
                }
            )
        return pd.DataFrame(rows)

    def subgroup_medians(self, subgroup_col: str) -> pd.DataFrame:
        """Return median survival per subgroup for a fitted subgroup column."""
        if subgroup_col not in self.subgroup_kmfs:
            raise KeyError(f"Subgroup {subgroup_col!r} not fitted. Call fit_subgroups() first.")
        rows = []
        for grp, kmf in self.subgroup_kmfs[subgroup_col].items():
            med = kmf.median_survival_time_
            eo = kmf.event_observed
            n_subj = int(eo.size) if isinstance(eo, np.ndarray) else int(len(eo))
            n_ev = int(eo.sum())
            rows.append(
                {
                    "group": grp,
                    "median_survival": float(med) if med is not None and not np.isnan(float(med)) else float("inf"),
                    "n_subjects": n_subj,
                    "n_events": n_ev,
                }
            )
        return pd.DataFrame(rows).sort_values("group")

    def cumulative_hazard_at(self, times: list[float]) -> pd.DataFrame:
        """Nelson-Aalen cumulative hazard at given times."""
        if not self._fitted:
            raise RuntimeError("Call fit() before cumulative_hazard_at().")
        rows = []
        for t in times:
            h = self.overall_naf.predict(t)
            rows.append({"time": t, "cumulative_hazard": float(h)})
        return pd.DataFrame(rows)


def run_pairwise_logrank(
    df: pd.DataFrame,
    group_col: str,
    duration_col: str = "duration",
    event_col: str = "event",
) -> pd.DataFrame:
    """Run pairwise log-rank tests for all pairs of subgroups.

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str
        Column defining subgroups.
    duration_col, event_col : str

    Returns
    -------
    pd.DataFrame with columns [group_a, group_b, test_statistic, p_value]
    """
    groups = sorted(df[group_col].dropna().unique())
    rows = []
    for i, g_a in enumerate(groups):
        for g_b in groups[i + 1:]:
            mask_a = df[group_col] == g_a
            mask_b = df[group_col] == g_b
            result = logrank_test(
                df.loc[mask_a, duration_col],
                df.loc[mask_b, duration_col],
                event_observed_A=df.loc[mask_a, event_col],
                event_observed_B=df.loc[mask_b, event_col],
            )
            rows.append(
                {
                    "group_a": str(g_a),
                    "group_b": str(g_b),
                    "test_statistic": float(result.test_statistic),
                    "p_value": float(result.p_value),
                    "significant": result.p_value < 0.05,
                }
            )
    return pd.DataFrame(rows)
