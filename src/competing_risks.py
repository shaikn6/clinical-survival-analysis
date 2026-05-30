"""
Competing risks analysis.

Implements:
- Cause-specific hazard via separate Cox PH models (one per cause)
- Aalen-Johansen estimator for the Cumulative Incidence Function (CIF)
- Visualisation of CIF curves for competing causes
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


# ---------------------------------------------------------------------------
# Cause-specific hazards
# ---------------------------------------------------------------------------

def cause_specific_hazard(
    time: np.ndarray,
    event: np.ndarray,
    covariates: np.ndarray,
) -> dict:
    """Fit separate Cox PH models for each competing cause.

    Treats each cause as the event of interest while all other events are
    treated as censored (the standard cause-specific hazard approach).

    Parameters
    ----------
    time : np.ndarray  shape (n,)  — observed time
    event : np.ndarray  shape (n,)  — 0=censored, 1=event of interest, 2=competing event
    covariates : np.ndarray  shape (n, p)  — numeric covariate matrix

    Returns
    -------
    dict  — {1: CoxPHFitter, 2: CoxPHFitter}
        Fitted model for each cause.
    """
    causes = [c for c in np.unique(event) if c != 0]
    if not causes:
        raise ValueError("No events found (all observations are censored).")

    models: dict[int, CoxPHFitter] = {}
    n_features = covariates.shape[1]
    feat_cols = [f"x{i}" for i in range(n_features)]

    base_df = pd.DataFrame(covariates, columns=feat_cols)
    base_df["duration"] = time.astype(float)

    for cause in causes:
        # Indicator: 1 if this cause occurred, 0 otherwise (censor competing events)
        base_df["event_cause"] = (event == cause).astype(int)

        df_fit = base_df[["duration", "event_cause"] + feat_cols].copy()
        df_fit = df_fit.dropna()

        if df_fit["event_cause"].sum() < 5:
            warnings.warn(
                f"Cause {cause} has fewer than 5 events; model may be unreliable.",
                stacklevel=2,
            )

        fitter = CoxPHFitter(penalizer=0.1)
        try:
            fitter.fit(
                df_fit,
                duration_col="duration",
                event_col="event_cause",
                show_progress=False,
            )
            models[int(cause)] = fitter
        except Exception as exc:
            warnings.warn(f"Cox model failed for cause {cause}: {exc}", stacklevel=2)

    return models


# ---------------------------------------------------------------------------
# Cumulative Incidence Function — Aalen-Johansen estimator
# ---------------------------------------------------------------------------

def cumulative_incidence_function(
    time: np.ndarray,
    event: np.ndarray,
    cause: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the Cumulative Incidence Function via Aalen-Johansen.

    CIF_k(t) = integral from 0 to t of [S(s-) * lambda_k(s)] ds

    where S(s-) is the overall survival (accounting for all causes) and
    lambda_k(s) is the cause-specific hazard at time s.

    Parameters
    ----------
    time : np.ndarray  shape (n,)
    event : np.ndarray  shape (n,)  — 0=censored, 1=cause 1, 2=cause 2, ...
    cause : int
        Cause of interest.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (time_points, CIF_values) — both shape (m,) with m unique event times.
    """
    n = len(time)
    unique_times = np.sort(np.unique(time[event > 0]))

    # Overall survival S(t-): proportion still at risk just before each time
    overall_survival = np.ones(len(unique_times), dtype=float)
    cif = np.zeros(len(unique_times), dtype=float)

    s_prev = 1.0
    for j, t in enumerate(unique_times):
        # Number at risk just before t
        n_at_risk = int((time >= t).sum())
        if n_at_risk == 0:
            overall_survival[j] = s_prev
            cif[j] = cif[j - 1] if j > 0 else 0.0
            continue

        # Cause-specific events at exactly t
        d_k = int(((time == t) & (event == cause)).sum())
        d_all = int(((time == t) & (event > 0)).sum())

        # Cause-specific hazard: d_k / n_at_risk
        lambda_k = d_k / n_at_risk
        # Overall hazard: d_all / n_at_risk
        lambda_all = d_all / n_at_risk

        # CIF increment = S(t-) * lambda_k(t)
        cif_increment = s_prev * lambda_k
        cif[j] = (cif[j - 1] if j > 0 else 0.0) + cif_increment

        # Update overall survival: S(t) = S(t-) * (1 - lambda_all)
        s_prev = s_prev * (1.0 - lambda_all)
        overall_survival[j] = s_prev

    return unique_times, np.clip(cif, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_competing_risks(
    time: np.ndarray,
    event: np.ndarray,
    output_path: str,
) -> "plt.Figure":
    """Plot CIF curves for all competing causes on the same axes.

    Parameters
    ----------
    time : np.ndarray  shape (n,)
    event : np.ndarray  shape (n,)  — 0=censored, 1=cause 1, 2=cause 2, ...
    output_path : str
        File path where the figure is saved (PNG recommended).

    Returns
    -------
    matplotlib.figure.Figure
    """
    causes = sorted(c for c in np.unique(event) if c != 0)
    colors = ["#4f8ef7", "#f97316", "#22c55e", "#a855f7", "#ef4444"]
    labels = {1: "Event of interest (cause 1)", 2: "Competing event (cause 2)"}

    fig, ax = plt.subplots(figsize=(9, 5))

    for cause, color in zip(causes, colors):
        t_pts, cif_vals = cumulative_incidence_function(time, event, cause=cause)
        label = labels.get(cause, f"Cause {cause}")
        ax.step(t_pts, cif_vals, where="post", color=color, linewidth=2.5, label=label)

    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Cumulative Incidence", fontsize=12)
    ax.set_title("Cumulative Incidence Functions — Competing Risks", fontsize=13)
    ax.set_ylim(0, None)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig
