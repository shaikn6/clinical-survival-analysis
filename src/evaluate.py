"""
Model evaluation metrics for survival analysis.

Implements:
- Concordance index (C-index / Harrell's C)
- Integrated Brier Score (IBS) at standard clinical time points
- Time-dependent AUC at standard clinical time points
- Calibration: expected vs observed survival
- Model comparison DataFrame → results/comparison.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv

CLINICAL_TIMES = [30, 90, 180, 365]
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Structured array helpers
# ---------------------------------------------------------------------------

def _to_surv_array(df: pd.DataFrame, event_col: str, duration_col: str) -> np.ndarray:
    """Build a scikit-survival structured array from a DataFrame."""
    return Surv.from_arrays(
        event=df[event_col].astype(bool).values,
        time=df[duration_col].astype(float).values,
    )


# ---------------------------------------------------------------------------
# C-index
# ---------------------------------------------------------------------------

def compute_c_index(
    y_true: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Compute Harrell's concordance index.

    Parameters
    ----------
    y_true : np.ndarray
        Structured array from ``_to_surv_array``.
    risk_scores : np.ndarray
        Risk scores per patient (higher = higher risk).

    Returns
    -------
    float in [0, 1]
    """
    events = y_true["event"]
    times = y_true["time"]
    c, _, _, _, _ = concordance_index_censored(events, times, risk_scores)
    return float(c)


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------

def compute_brier_scores(
    y_train: np.ndarray,
    y_test: np.ndarray,
    survival_probs: np.ndarray,
    times: list[float] | None = None,
) -> pd.DataFrame:
    """Compute Brier score at each time point and the Integrated Brier Score.

    Parameters
    ----------
    y_train : np.ndarray
        Structured array from training set (used for IPCW weight estimation).
    y_test : np.ndarray
        Structured array from test set.
    survival_probs : np.ndarray
        Shape (n_patients, n_times) — predicted survival probabilities.
    times : list[float] | None
        Time points.  Defaults to CLINICAL_TIMES.

    Returns
    -------
    pd.DataFrame with columns [time, brier_score] and one row for IBS.
    """
    if times is None:
        times = CLINICAL_TIMES

    max_time = y_test["time"].max()
    valid_times = [t for t in times if t < max_time]
    if not valid_times:
        warnings.warn("No valid times within observed follow-up range.", stacklevel=2)
        return pd.DataFrame(columns=["time", "brier_score"])

    # Align survival_probs columns to valid_times
    valid_idx = [i for i, t in enumerate(times) if t in valid_times]
    sp = survival_probs[:, valid_idx]

    bs_times, bs_scores = brier_score(y_train, y_test, sp, valid_times)
    ibs = integrated_brier_score(y_train, y_test, sp, valid_times)

    rows = [{"time": int(t), "brier_score": float(s)} for t, s in zip(bs_times, bs_scores)]
    rows.append({"time": "IBS", "brier_score": float(ibs)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Time-dependent AUC
# ---------------------------------------------------------------------------

def compute_time_auc(
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
    times: list[float] | None = None,
) -> pd.DataFrame:
    """Compute time-dependent AUC at each time point.

    Parameters
    ----------
    y_train : np.ndarray
    y_test : np.ndarray
    risk_scores : np.ndarray
        Shape (n_patients,) — monotone risk scores.
    times : list[float] | None

    Returns
    -------
    pd.DataFrame with columns [time, auc]
    """
    if times is None:
        times = CLINICAL_TIMES

    max_time = y_test["time"].max() * 0.99  # avoid exact boundary issues
    valid_times = [t for t in times if t < max_time]
    if not valid_times:
        warnings.warn("No valid times within observed follow-up range.", stacklevel=2)
        return pd.DataFrame(columns=["time", "auc"])

    try:
        aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, valid_times)
        rows = [{"time": int(t), "auc": float(a)} for t, a in zip(valid_times, aucs)]
        return pd.DataFrame(rows)
    except Exception as exc:
        warnings.warn(f"AUC computation failed: {exc}", stacklevel=2)
        return pd.DataFrame(columns=["time", "auc"])


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def compute_calibration(
    y_test: np.ndarray,
    survival_probs: np.ndarray,
    times: list[float] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute calibration (expected vs observed survival) at each time.

    Patients are binned by predicted survival probability; within each bin
    the observed KM survival at that time is compared to the mean prediction.

    Parameters
    ----------
    y_test : np.ndarray
    survival_probs : np.ndarray  shape (n_patients, n_times)
    times : list[float] | None
    n_bins : int

    Returns
    -------
    pd.DataFrame with columns [time, bin_mid, predicted, observed, n]
    """
    if times is None:
        times = CLINICAL_TIMES

    from lifelines import KaplanMeierFitter

    rows = []
    T = y_test["time"]
    E = y_test["event"]

    for ti, t in enumerate(times):
        if ti >= survival_probs.shape[1]:
            continue
        preds = survival_probs[:, ti]
        bins = np.percentile(preds, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (preds >= lo) & (preds <= hi)
            if mask.sum() < 5:
                continue
            mean_pred = float(preds[mask].mean())
            kmf = KaplanMeierFitter()
            kmf.fit(T[mask], event_observed=E[mask], timeline=[t])
            obs = float(kmf.survival_function_.iloc[-1, 0])
            rows.append(
                {
                    "time": int(t),
                    "bin_mid": round((lo + hi) / 2, 3),
                    "predicted": mean_pred,
                    "observed": obs,
                    "n": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    dataset_name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    risk_scores: np.ndarray,
    survival_probs: np.ndarray,
    times: list[float] | None = None,
) -> dict:
    """Run all evaluation metrics and return a summary dict.

    Parameters
    ----------
    model_name : str
    dataset_name : str
    y_train, y_test : np.ndarray  structured arrays
    risk_scores : np.ndarray  shape (n_patients,)
    survival_probs : np.ndarray  shape (n_patients, n_times)
    times : list[float] | None

    Returns
    -------
    dict with keys: model, dataset, c_index, ibs, auc_30, auc_90, auc_180, auc_365
    """
    if times is None:
        times = CLINICAL_TIMES

    c_idx = compute_c_index(y_test, risk_scores)
    bs_df = compute_brier_scores(y_train, y_test, survival_probs, times=times)
    auc_df = compute_time_auc(y_train, y_test, risk_scores, times=times)

    ibs_rows = bs_df[bs_df["time"] == "IBS"]
    ibs = float(ibs_rows["brier_score"].iloc[0]) if len(ibs_rows) else float("nan")

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "c_index": round(c_idx, 4),
        "ibs": round(ibs, 4),
    }
    for t in times:
        auc_row = auc_df[auc_df["time"] == t]
        result[f"auc_{t}d"] = round(float(auc_row["auc"].iloc[0]), 4) if len(auc_row) else float("nan")

    return result


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(results: list[dict]) -> pd.DataFrame:
    """Aggregate per-model/per-dataset evaluation results into one DataFrame.

    Parameters
    ----------
    results : list[dict]
        List of dicts from ``evaluate_model``.

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame(results).sort_values(["dataset", "c_index"], ascending=[True, False])


def save_comparison(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save the comparison DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
    path : Path | None
        Defaults to RESULTS_DIR / 'comparison.csv'.

    Returns
    -------
    Path where the file was saved.
    """
    if path is None:
        path = RESULTS_DIR / "comparison.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
