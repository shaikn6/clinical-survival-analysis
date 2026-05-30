"""
Visualization for clinical survival analysis.

Produces:
- KM survival curves with confidence bands (matplotlib)
- KM interactive curves (plotly)
- Cox PH hazard ratio forest plot
- Feature importance bar charts (RSF + XGBoost)
- Calibration curves per model
- All outputs saved to results/figures/
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if TYPE_CHECKING:
    from lifelines import KaplanMeierFitter

FIGURES_DIR = Path(__file__).parent.parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

_PALETTE = ["#4f8ef7", "#f97316", "#22c55e", "#a855f7", "#ef4444", "#0ea5e9"]


# ---------------------------------------------------------------------------
# KM curves — matplotlib
# ---------------------------------------------------------------------------

def plot_km_curves(
    kmfs: dict[str, "KaplanMeierFitter"],
    title: str = "Kaplan-Meier Survival Curves",
    xlabel: str = "Time (days)",
    ylabel: str = "Survival Probability",
    figsize: tuple[int, int] = (10, 6),
    save_path: Path | None = None,
) -> Path:
    """Plot overlaid KM curves with confidence bands.

    Parameters
    ----------
    kmfs : dict[str, KaplanMeierFitter]
        Mapping of group label → fitted KaplanMeierFitter.
    title : str
    save_path : Path | None
        If None, saved to FIGURES_DIR / '<sanitised title>.png'.

    Returns
    -------
    Path where figure was saved.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = _PALETTE[: len(kmfs)]

    for (label, kmf), color in zip(kmfs.items(), colors):
        kmf.plot_survival_function(ax=ax, ci_show=True, color=color, label=label)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is None:
        fname = title.lower().replace(" ", "_").replace("/", "_")[:60] + ".png"
        save_path = FIGURES_DIR / fname
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_km_interactive(
    kmfs: dict[str, "KaplanMeierFitter"],
    title: str = "Kaplan-Meier Curves (Interactive)",
    save_path: Path | None = None,
) -> Path:
    """Build a Plotly interactive KM chart.

    Parameters
    ----------
    kmfs : dict
    title : str
    save_path : Path | None

    Returns
    -------
    Path to saved HTML file.
    """
    fig = go.Figure()
    colors = _PALETTE[: len(kmfs)]

    for (label, kmf), color in zip(kmfs.items(), colors):
        sf = kmf.survival_function_
        ci = kmf.confidence_interval_
        t = sf.index.tolist()

        ci_cols = ci.columns.tolist()
        upper_col = next((c for c in ci_cols if "upper" in c.lower()), None)
        lower_col = next((c for c in ci_cols if "lower" in c.lower()), None)

        # Main survival line
        fig.add_trace(
            go.Scatter(
                x=t,
                y=sf.iloc[:, 0].tolist(),
                mode="lines",
                name=label,
                line={"color": color, "width": 2},
            )
        )
        # CI band
        if upper_col and lower_col:
            fig.add_trace(
                go.Scatter(
                    x=t + t[::-1],
                    y=ci[upper_col].tolist() + ci[lower_col].tolist()[::-1],
                    fill="toself",
                    fillcolor=color,
                    opacity=0.15,
                    line={"color": "rgba(255,255,255,0)"},
                    showlegend=False,
                    name=f"{label} CI",
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Time (days)",
        yaxis_title="Survival Probability",
        yaxis={"range": [0, 1.05]},
        template="plotly_white",
        hovermode="x unified",
    )

    if save_path is None:
        fname = title.lower().replace(" ", "_").replace("/", "_")[:60] + ".html"
        save_path = FIGURES_DIR / fname
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(save_path))
    return save_path


# ---------------------------------------------------------------------------
# Cox PH Forest plot
# ---------------------------------------------------------------------------

def plot_forest(
    hr_df: pd.DataFrame,
    title: str = "Hazard Ratios (Cox PH)",
    figsize: tuple[int, int] = (10, 7),
    save_path: Path | None = None,
) -> Path:
    """Forest plot of hazard ratios with 95% confidence intervals.

    Parameters
    ----------
    hr_df : pd.DataFrame
        Must contain columns: feature, hr, lower_95, upper_95, p_value.
    save_path : Path | None

    Returns
    -------
    Path
    """
    # Detect required columns flexibly
    feature_col = next((c for c in hr_df.columns if c in ("feature", "covariate")), hr_df.columns[0])
    hr_col = next((c for c in hr_df.columns if c in ("hr", "exp(coef)")), None)
    lo_col = next((c for c in hr_df.columns if "lower" in c.lower()), None)
    hi_col = next((c for c in hr_df.columns if "upper" in c.lower()), None)
    p_col = next((c for c in hr_df.columns if "p_value" in c.lower() or c == "p"), None)

    if hr_col is None:
        # fallback: use coef column
        hr_col = next((c for c in hr_df.columns if "coef" in c.lower()), hr_df.columns[1])

    df = hr_df.copy()
    df = df.sort_values(hr_col, ascending=True).reset_index(drop=True)

    n = len(df)
    fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(n)
    hrs = df[hr_col].values.astype(float)
    colors = ["#ef4444" if h > 1 else "#22c55e" for h in hrs]

    # Error bars
    if lo_col and hi_col:
        lo = df[lo_col].values.astype(float)
        hi = df[hi_col].values.astype(float)
        xerr_lo = np.clip(hrs - lo, 0, None)
        xerr_hi = np.clip(hi - hrs, 0, None)
        ax.barh(y, hrs, xerr=[xerr_lo, xerr_hi], color=colors, alpha=0.75, ecolor="grey", capsize=4)
    else:
        ax.barh(y, hrs, color=colors, alpha=0.75)

    ax.axvline(x=1.0, color="black", linewidth=1.2, linestyle="--", label="HR=1 (no effect)")
    ax.set_yticks(y)
    ax.set_yticklabels(df[feature_col].astype(str).tolist(), fontsize=9)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")

    legend_patches = [
        mpatches.Patch(color="#ef4444", alpha=0.75, label="HR > 1 (increased risk)"),
        mpatches.Patch(color="#22c55e", alpha=0.75, label="HR < 1 (protective)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "forest_plot.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str = "Model",
    top_n: int = 15,
    figsize: tuple[int, int] = (9, 6),
    save_path: Path | None = None,
) -> Path:
    """Horizontal bar chart of feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Must contain columns 'feature' and one of 'importance',
        'importance_mean', or 'mean_abs_shap'.
    model_name : str
    top_n : int
    save_path : Path | None

    Returns
    -------
    Path
    """
    imp_col = next(
        (c for c in ["importance_mean", "mean_abs_shap", "importance"] if c in importance_df.columns),
        importance_df.columns[-1],
    )
    df = importance_df.sort_values(imp_col, ascending=False).head(top_n).copy()
    df = df.sort_values(imp_col, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df["feature"].astype(str), df[imp_col], color="#4f8ef7", alpha=0.85)

    # Value labels on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + max(df[imp_col]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}",
            va="center",
            fontsize=8,
        )

    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(imp_col.replace("_", " ").title(), fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_path is None:
        fname = f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
        save_path = FIGURES_DIR / fname
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Calibration curve
# ---------------------------------------------------------------------------

def plot_calibration(
    calibration_df: pd.DataFrame,
    model_name: str = "Model",
    figsize: tuple[int, int] = (8, 6),
    save_path: Path | None = None,
) -> Path:
    """Calibration curve: predicted vs observed survival probability.

    Parameters
    ----------
    calibration_df : pd.DataFrame
        From evaluate.compute_calibration().  Must contain columns
        [time, predicted, observed].
    model_name : str
    save_path : Path | None

    Returns
    -------
    Path
    """
    times = sorted(calibration_df["time"].unique())
    n_plots = len(times)
    ncols = min(2, n_plots)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1] * nrows // ncols + 1))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax, t in zip(axes, times):
        sub = calibration_df[calibration_df["time"] == t]
        ax.scatter(sub["predicted"], sub["observed"], alpha=0.7, color="#4f8ef7", s=30)
        lim = [0, 1]
        ax.plot(lim, lim, "k--", linewidth=1, label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Survival", fontsize=10)
        ax.set_ylabel("Observed Survival (KM)", fontsize=10)
        ax.set_title(f"t = {t} days", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    # Hide unused axes
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(f"Calibration — {model_name}", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path is None:
        fname = f"calibration_{model_name.lower().replace(' ', '_')}.png"
        save_path = FIGURES_DIR / fname
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric: str = "c_index",
    figsize: tuple[int, int] = (10, 6),
    save_path: Path | None = None,
) -> Path:
    """Grouped bar chart comparing models across datasets.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        From evaluate.build_comparison_table().
    metric : str
    save_path : Path | None

    Returns
    -------
    Path
    """
    datasets = comparison_df["dataset"].unique()
    models = comparison_df["model"].unique()

    x = np.arange(len(datasets))
    width = 0.8 / len(models)
    offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(models))

    fig, ax = plt.subplots(figsize=figsize)

    for offset, model, color in zip(offsets, models, _PALETTE[: len(models)]):
        vals = []
        for ds in datasets:
            row = comparison_df[(comparison_df["model"] == model) & (comparison_df["dataset"] == ds)]
            vals.append(float(row[metric].iloc[0]) if len(row) else float("nan"))
        ax.bar(x + offset, vals, width=width * 0.9, label=model, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(0.4, 1.0) if "c_index" in metric else ax.set_ylim(0, None)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / f"model_comparison_{metric}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
