"""
Extended evaluation for V2 deep learning survival models.

Adds:
- evaluate_deep_models: C-index + Brier score for DeepSurv and DeepHit
- build_full_comparison_table: merge V1 (KM, Cox, RSF, XGBoost) + V2 (DeepSurv, DeepHit)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Deep model evaluation
# ---------------------------------------------------------------------------

def evaluate_deep_models(
    models: dict,
    time_points: list[int],
) -> pd.DataFrame:
    """Evaluate DeepSurv and DeepHit trainers on test sets.

    Parameters
    ----------
    models : dict
        Mapping of model name → (trainer, X_test, time_test, event_test).
        - For DeepSurv trainers: must have ``concordance_index`` and ``predict_risk``.
        - For DeepHit trainers: must additionally have ``predict_survival``.
    time_points : list[int]
        Time points (days) at which to evaluate Brier score (DeepHit only).

    Returns
    -------
    pd.DataFrame
        Columns: model, c_index, brier_<t>d (DeepHit only), dataset
    """
    rows = []
    for name, payload in models.items():
        trainer, X_test, time_test, event_test = payload

        # C-index
        try:
            c_idx = trainer.concordance_index(X_test, time_test, event_test)
        except Exception as exc:
            warnings.warn(f"C-index failed for {name}: {exc}", stacklevel=2)
            c_idx = float("nan")

        row: dict = {"model": name, "c_index": round(float(c_idx), 4)}

        # Brier score — only for models that expose predict_survival
        if hasattr(trainer, "predict_survival"):
            try:
                surv = trainer.predict_survival(X_test, time_points)
                for j, tp in enumerate(time_points):
                    survival_col = surv[:, j]
                    # Observed proportion who survived past tp (crude estimate)
                    float((time_test > tp).mean())
                    float(survival_col.mean())
                    brier = float(np.mean((survival_col - (time_test > tp).astype(float)) ** 2))
                    row[f"brier_{tp}d"] = round(brier, 4)
            except Exception as exc:
                warnings.warn(f"Brier score failed for {name}: {exc}", stacklevel=2)
                for tp in time_points:
                    row[f"brier_{tp}d"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Full 6-model comparison table
# ---------------------------------------------------------------------------

def build_full_comparison_table(
    traditional_results: dict,
    deep_results: pd.DataFrame,
) -> pd.DataFrame:
    """Combine V1 (KM, Cox PH, RSF, XGBoost) and V2 (DeepSurv, DeepHit).

    Parameters
    ----------
    traditional_results : dict
        Keys are model names; values are dicts with at least 'c_index' and 'dataset'.
        Matches the output format of ``src.evaluate.evaluate_model`` converted to dicts.
        Example::

            {
                "KM":       {"c_index": 0.5,   "dataset": "icu"},
                "Cox PH":   {"c_index": 0.72,  "dataset": "icu"},
                "RSF":      {"c_index": 0.76,  "dataset": "icu"},
                "XGBoost":  {"c_index": 0.77,  "dataset": "icu"},
            }

    deep_results : pd.DataFrame
        Output of ``evaluate_deep_models``.

    Returns
    -------
    pd.DataFrame
        One row per model, columns: model, c_index, dataset (plus any brier columns).
    """
    trad_rows = []
    for model_name, info in traditional_results.items():
        row = {"model": model_name, "c_index": info.get("c_index", float("nan"))}
        row["dataset"] = info.get("dataset", "icu")
        # Carry forward any extra columns (ibs, auc_*, ...)
        for k, v in info.items():
            if k not in row:
                row[k] = v
        trad_rows.append(row)

    trad_df = pd.DataFrame(trad_rows)

    # Merge: align on common columns, fill missing with NaN
    combined = pd.concat([trad_df, deep_results], ignore_index=True, sort=False)
    combined = combined.reset_index(drop=True)

    # Ensure canonical column order
    front_cols = ["model", "c_index", "dataset"]
    other_cols = [c for c in combined.columns if c not in front_cols]
    combined = combined[[c for c in front_cols if c in combined.columns] + other_cols]

    return combined
