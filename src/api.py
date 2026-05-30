"""
FastAPI service for clinical survival predictions.

Endpoints:
- POST /predict   — patient features → survival probabilities at [30, 90, 180, 365] days
- GET  /models    — list available models and their C-index scores
- GET  /health    — liveness check
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Clinical Survival Analysis API",
    description="Survival probability predictions from trained clinical models.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# In-memory model registry
# Populated at startup from the pipeline (or fallback synthetic models)
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: dict[str, Any] = {}
_MODEL_METADATA: list[dict] = []
_ACTIVE_MODEL_NAME: str = ""
_FEATURE_COLS: list[str] = []
_SCALER: Any = None

PREDICT_TIMES = [30, 90, 180, 365]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class PatientFeatures(BaseModel):
    """ICU patient features for survival prediction."""

    age: float = Field(65.0, ge=18.0, le=110.0, description="Age in years")
    los: float = Field(5.0, ge=0.0, description="Length of stay in days")
    sofa_score: float = Field(6.0, ge=0.0, le=24.0, description="SOFA score")
    creatinine: float = Field(1.0, ge=0.0, description="Serum creatinine (mg/dL)")
    lactate: float = Field(1.5, ge=0.0, description="Blood lactate (mmol/L)")
    map: float = Field(75.0, ge=0.0, le=200.0, description="Mean arterial pressure (mmHg)")
    spo2: float = Field(96.0, ge=0.0, le=100.0, description="Oxygen saturation (%)")
    diabetes: int = Field(0, ge=0, le=1, description="Diabetes (0/1)")
    hypertension: int = Field(0, ge=0, le=1, description="Hypertension (0/1)")
    heart_failure: int = Field(0, ge=0, le=1, description="Heart failure (0/1)")
    ckd: int = Field(0, ge=0, le=1, description="Chronic kidney disease (0/1)")
    copd: int = Field(0, ge=0, le=1, description="COPD (0/1)")
    cci: float = Field(2.0, ge=0.0, description="Charlson comorbidity index")


class PredictionResponse(BaseModel):
    model: str
    survival_probabilities: dict[str, float]
    risk_score: float
    message: str


class ModelInfo(BaseModel):
    name: str
    c_index: float | None
    dataset: str | None
    is_active: bool


class HealthResponse(BaseModel):
    status: str
    timestamp: float
    models_loaded: int


# ---------------------------------------------------------------------------
# Startup: load or build fallback models
# ---------------------------------------------------------------------------

@app.on_event("startup")
def _startup() -> None:
    """Train a minimal fallback model if no pre-trained model is registered."""
    global _MODEL_REGISTRY, _MODEL_METADATA, _ACTIVE_MODEL_NAME, _FEATURE_COLS, _SCALER

    if _MODEL_REGISTRY:
        return  # already populated externally

    # Build a minimal RSF model on synthetic ICU data as fallback
    try:
        from src.data import prepare_dataset
        from src.models.random_survival_forest import RSFModel

        dataset = prepare_dataset("icu", n_icu=500, seed=42)
        _SCALER = dataset["scaler"]
        _FEATURE_COLS = dataset["feature_cols"]

        rsf = RSFModel(n_estimators=50, random_state=42)
        rsf.fit(dataset["train"], _FEATURE_COLS)
        c_idx = rsf.score(dataset["test"])

        _MODEL_REGISTRY["RSF_ICU"] = rsf
        _MODEL_METADATA = [
            {"name": "RSF_ICU", "c_index": round(c_idx, 4), "dataset": "icu", "is_active": True}
        ]
        _ACTIVE_MODEL_NAME = "RSF_ICU"
    except Exception as exc:
        # Graceful degradation — API still starts
        _MODEL_METADATA = []
        _ACTIVE_MODEL_NAME = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _features_to_df(patient: PatientFeatures) -> pd.DataFrame:
    """Convert PatientFeatures schema to a single-row DataFrame."""
    row = patient.model_dump()
    return pd.DataFrame([row])


def _scale_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the stored scaler to the patient feature DataFrame."""
    if _SCALER is None or not _FEATURE_COLS:
        return df
    available = [c for c in _FEATURE_COLS if c in df.columns]
    df = df.copy()
    df[available] = _SCALER.transform(df[available])
    return df


def register_model(
    name: str,
    model: Any,
    c_index: float | None,
    dataset: str,
    feature_cols: list[str],
    scaler: Any,
    make_active: bool = False,
) -> None:
    """Register a trained model with the API (call from pipeline code).

    Parameters
    ----------
    name : str
    model : fitted model with predict_survival(df, times) and predict_risk(df)
    c_index : float | None
    dataset : str
    feature_cols : list[str]
    scaler : fitted StandardScaler
    make_active : bool
    """
    global _MODEL_REGISTRY, _MODEL_METADATA, _ACTIVE_MODEL_NAME, _FEATURE_COLS, _SCALER

    _MODEL_REGISTRY[name] = model
    _MODEL_METADATA.append(
        {"name": name, "c_index": c_index, "dataset": dataset, "is_active": make_active}
    )
    if make_active or not _ACTIVE_MODEL_NAME:
        _ACTIVE_MODEL_NAME = name
        _FEATURE_COLS = feature_cols
        _SCALER = scaler


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse(
        status="ok",
        timestamp=time.time(),
        models_loaded=len(_MODEL_REGISTRY),
    )


@app.get("/models", response_model=list[ModelInfo], tags=["Models"])
def list_models() -> list[ModelInfo]:
    """List all available models and their C-index scores."""
    if not _MODEL_METADATA:
        return []
    return [ModelInfo(**m) for m in _MODEL_METADATA]


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientFeatures) -> PredictionResponse:
    """Predict survival probabilities at [30, 90, 180, 365] days.

    Accepts patient features and returns:
    - Survival probability at each standard clinical time point
    - Scalar risk score (higher = higher risk)
    """
    if not _ACTIVE_MODEL_NAME or _ACTIVE_MODEL_NAME not in _MODEL_REGISTRY:
        raise HTTPException(
            status_code=503,
            detail="No model is loaded. Start the pipeline first.",
        )

    model = _MODEL_REGISTRY[_ACTIVE_MODEL_NAME]
    df = _features_to_df(patient)
    df = _scale_patient(df)

    # Align columns: add missing features as 0
    if _FEATURE_COLS:
        for col in _FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        df = df[_FEATURE_COLS]

    try:
        surv = model.predict_survival(df, times=PREDICT_TIMES)
        # shape (1, n_times) → first row
        probs = surv[0].tolist()
        risk = float(model.predict_risk(df)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return PredictionResponse(
        model=_ACTIVE_MODEL_NAME,
        survival_probabilities={
            f"{t}d": round(float(p), 4) for t, p in zip(PREDICT_TIMES, probs)
        },
        risk_score=round(risk, 4),
        message="Survival probability at standard clinical time points.",
    )
