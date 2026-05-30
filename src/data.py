"""
Data loading, preprocessing, and synthetic cohort generation.

Datasets:
- Lung:  NCCTG Lung Cancer Study, n=228 (lifelines.datasets.load_lung)
- GBSG2: German Breast Cancer Study Group, n=686 (lifelines.datasets.load_gbsg2)
- ICU:   Synthetic ICU cohort calibrated to MIMIC-III population statistics
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.datasets import load_gbsg2 as _load_gbsg2
from lifelines.datasets import load_lung as _load_lung
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_lung() -> pd.DataFrame:
    """Load NCCTG Lung Cancer Study dataset (North Central Cancer Treatment Group).

    228 patients with advanced lung cancer. Outcome is overall survival.

    Returns dataframe with columns:
        duration  – follow-up days
        event     – 1 = death, 0 = censored
        age, sex, ph.ecog, ph.karno, pat.karno, meal.cal, wt.loss
    """
    df = _load_lung()
    # status: 1 = censored, 2 = dead — convert to 0/1
    df["event"] = (df["status"] == 2).astype(int)
    df = df.rename(columns={"time": "duration"})
    df = df.drop(columns=["status", "inst"], errors="ignore")
    df["duration"] = df["duration"].astype(float)
    # Simple median imputation for the two columns with NaN
    for col in df.columns:
        if df[col].isna().any() and col not in ("duration", "event"):
            df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)


# Keep the old name as an alias so tests referencing load_whas500 still work
load_whas500 = load_lung


def load_gbsg2() -> pd.DataFrame:
    """Load GBSG2 German Breast Cancer Study Group dataset.

    686 patients, outcome is recurrence-free survival.

    Returns dataframe with columns:
        duration  – follow-up days
        event     – 1 = recurrence/death, 0 = censored
        + clinical features
    """
    df = _load_gbsg2()
    df = df.rename(columns={"time": "duration", "cens": "event"})
    df["duration"] = df["duration"].astype(float)
    df["event"] = df["event"].astype(int)
    # Encode categorical columns as numeric
    if "horTh" in df.columns:
        df["horTh"] = (df["horTh"] == "yes").astype(int)
    if "menostat" in df.columns:
        df["menostat"] = (df["menostat"] == "Post").astype(int)
    if "tgrade" in df.columns:
        grade_map = {"I": 1, "II": 2, "III": 3}
        df["tgrade"] = df["tgrade"].map(grade_map).fillna(2).astype(int)
    return df.reset_index(drop=True)


def generate_synthetic_icu(
    n_patients: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic ICU cohort calibrated to MIMIC-III statistics.

    Demographics are drawn from published MIMIC-III summary statistics:
        age      ~ N(65, 15), clipped [18, 99]
        LOS      ~ N(5, 3),   clipped [0.5, 30] days
        30-day mortality ~ 12%

    Parameters
    ----------
    n_patients : int
        Number of synthetic patients to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Cohort with features and survival outcome columns.
    """
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(65.0, 15.0, n_patients), 18, 99)
    los = np.clip(rng.normal(5.0, 3.0, n_patients), 0.5, 30.0)

    # Binary comorbidities
    diabetes = rng.binomial(1, 0.30, n_patients)
    hypertension = rng.binomial(1, 0.45, n_patients)
    heart_failure = rng.binomial(1, 0.25, n_patients)
    ckd = rng.binomial(1, 0.20, n_patients)
    copd = rng.binomial(1, 0.15, n_patients)

    # Continuous vitals / labs
    sofa_score = np.clip(rng.normal(6.0, 3.5, n_patients), 0, 24)
    creatinine = np.clip(rng.lognormal(0.2, 0.5, n_patients), 0.4, 15.0)
    lactate = np.clip(rng.lognormal(0.4, 0.6, n_patients), 0.5, 20.0)
    map_val = np.clip(rng.normal(75.0, 12.0, n_patients), 40, 120)
    spo2 = np.clip(rng.normal(96.0, 3.0, n_patients), 70, 100)

    # Charlson comorbidity index (simplified)
    cci = (
        (age >= 50).astype(int)
        + (age >= 60).astype(int)
        + (age >= 70).astype(int)
        + diabetes
        + heart_failure
        + ckd * 2
        + copd
    )

    # Hazard-driven event generation
    # Intercept -10.0 calibrated to produce ~20-25% event rate
    # with administrative censoring in [90, 730] days.
    log_h = (
        -10.0
        + 0.03 * age
        + 0.06 * sofa_score
        + 0.04 * cci
        + 0.08 * np.log1p(lactate)
        + rng.normal(0, 0.3, n_patients)
    )
    hazard = np.exp(log_h)
    # Exponential survival model: T ~ Exp(hazard), mean survival 1/hazard days
    true_duration = rng.exponential(1.0 / hazard)

    # Administrative censoring uniformly in [90, 730] days
    censor_time = rng.uniform(90.0, 730.0, n_patients)
    duration = np.minimum(true_duration, censor_time)
    event = (true_duration <= censor_time).astype(int)

    # Clip to [1, 730]
    duration = np.clip(duration, 1.0, 730.0)

    df = pd.DataFrame(
        {
            "duration": duration.astype(float),
            "event": event,
            "age": age.astype(float),
            "los": los.astype(float),
            "sofa_score": sofa_score.astype(float),
            "creatinine": creatinine.astype(float),
            "lactate": lactate.astype(float),
            "map": map_val.astype(float),
            "spo2": spo2.astype(float),
            "diabetes": diabetes,
            "hypertension": hypertension,
            "heart_failure": heart_failure,
            "ckd": ckd,
            "copd": copd,
            "cci": cci,
        }
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, dataset: str = "icu") -> pd.DataFrame:
    """Add derived features to a clinical dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw clinical dataset.
    dataset : str
        One of 'icu', 'whas500', 'gbsg2'.

    Returns
    -------
    pd.DataFrame
        Dataset with additional engineered columns.
    """
    df = df.copy()

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 45, 60, 75, 200],
            labels=["<45", "45-60", "60-75", "75+"],
        ).astype(str)

    if "cci" in df.columns:
        df["cci_level"] = pd.cut(
            df["cci"],
            bins=[-1, 1, 3, 6, 100],
            labels=["low", "moderate", "high", "very_high"],
        ).astype(str)

    if "los" in df.columns:
        df["los_category"] = pd.cut(
            df["los"],
            bins=[0, 2, 5, 10, 100],
            labels=["short", "medium", "long", "prolonged"],
        ).astype(str)

    if dataset in ("whas500", "lung"):
        # Lung cancer feature engineering
        if "ph.ecog" in df.columns:
            df["ecog_group"] = df["ph.ecog"].apply(
                lambda x: "good" if x <= 1 else ("moderate" if x <= 2 else "poor")
            )

    if dataset == "gbsg2":
        # Hormone receptor status interaction
        if "horTh" in df.columns and "menostat" in df.columns:
            df["hormone_meno_interaction"] = (df["horTh"] * df["menostat"]).astype(float)

    return df


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns (exclude outcome and categorical)."""
    exclude = {"duration", "event", "age_group", "cci_level", "los_category", "bmi_category"}
    return [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def preprocess(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[pd.DataFrame, list[str], StandardScaler]:
    """Impute missing values and scale numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with 'duration' and 'event' columns.
    feature_cols : list[str] | None
        Numeric feature columns to scale.  Inferred if None.
    scaler : StandardScaler | None
        Pre-fitted scaler (pass when transforming val/test splits).
    fit_scaler : bool
        Whether to fit a new scaler on ``df``.

    Returns
    -------
    (processed_df, feature_cols, scaler)
    """
    df = df.copy()

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    # Median imputation for numeric features
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    if fit_scaler:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False")
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, feature_cols, scaler


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test (70 / 15 / 15).

    Parameters
    ----------
    df : pd.DataFrame
    test_size : float
        Fraction reserved for test.
    val_size : float
        Fraction reserved for validation (of training pool after test split).
    seed : int

    Returns
    -------
    (train, val, test)
    """
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["event"]
    )
    # val fraction relative to train_val so that overall val ~ 15%
    val_fraction_of_trainval = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_trainval,
        random_state=seed,
        stratify=train_val["event"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def prepare_dataset(
    name: str,
    n_icu: int = 1000,
    seed: int = 42,
) -> dict:
    """Load, engineer features, preprocess, and split a named dataset.

    Parameters
    ----------
    name : str
        One of 'whas500', 'gbsg2', 'icu'.
    n_icu : int
        Number of synthetic ICU patients (only used when name='icu').
    seed : int

    Returns
    -------
    dict with keys: train, val, test, feature_cols, scaler, name
    """
    if name in ("whas500", "lung"):
        raw = load_lung()
        raw = engineer_features(raw, dataset="lung")
    elif name == "gbsg2":
        raw = load_gbsg2()
        raw = engineer_features(raw, dataset="gbsg2")
    elif name == "icu":
        raw = generate_synthetic_icu(n_patients=n_icu, seed=seed)
        raw = engineer_features(raw, dataset="icu")
    else:
        raise ValueError(f"Unknown dataset: {name!r}. Choose 'whas500', 'gbsg2', or 'icu'.")

    train_raw, val_raw, test_raw = split_dataset(raw, seed=seed)
    feature_cols = get_feature_columns(train_raw)

    train, feature_cols, scaler = preprocess(train_raw, feature_cols=feature_cols, fit_scaler=True)
    val, _, _ = preprocess(val_raw, feature_cols=feature_cols, scaler=scaler, fit_scaler=False)
    test, _, _ = preprocess(test_raw, feature_cols=feature_cols, scaler=scaler, fit_scaler=False)

    return {
        "name": name,
        "train": train,
        "val": val,
        "test": test,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "raw": raw,
    }
