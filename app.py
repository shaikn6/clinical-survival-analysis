"""
Streamlit dashboard for Clinical Survival Analysis.

Tabs:
  1. Dataset Overview    — cohort statistics, event rate, follow-up
  2. KM Curves           — subgroup selector
  3. Model Comparison    — C-index / IBS table per dataset
  4. Patient Prediction  — features → survival curve + risk score
  5. Feature Importance  — RSF / XGBoost importance comparison
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Clinical Survival Analysis",
    page_icon="🩺",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_datasets() -> dict:
    from src.data import prepare_dataset
    return {
        "whas500": prepare_dataset("whas500"),
        "gbsg2": prepare_dataset("gbsg2"),
        "icu": prepare_dataset("icu", n_icu=1000),
    }


@st.cache_resource(show_spinner=False)
def _train_models(datasets: dict) -> dict:
    """Train all 4 models on ICU dataset for demo. Returns model objects."""
    from src.models.kaplan_meier import KaplanMeierModel
    from src.models.cox_ph import CoxPHModel
    from src.models.random_survival_forest import RSFModel
    from src.models.xgboost_survival import XGBoostSurvivalModel

    results = {}
    for ds_name, ds in datasets.items():
        train = ds["train"]
        test = ds["test"]
        feat = ds["feature_cols"]

        km = KaplanMeierModel(label=ds_name)
        km.fit(train)
        results[f"km_{ds_name}"] = km

        cox = CoxPHModel(penalizer=0.1)
        cox.fit(train, feat)
        results[f"cox_{ds_name}"] = cox

        rsf = RSFModel(n_estimators=50, random_state=42)
        rsf.fit(train, feat)
        results[f"rsf_{ds_name}"] = rsf

        xgb = XGBoostSurvivalModel(random_state=42)
        xgb.fit(train, feat, n_estimators=50)
        results[f"xgb_{ds_name}"] = xgb

    return results


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------

with st.spinner("Loading datasets and training models (first run only)..."):
    try:
        datasets = _load_datasets()
        models = _train_models(datasets)
        DATA_LOADED = True
    except Exception as exc:
        st.error(f"Error loading data: {exc}")
        DATA_LOADED = False
        datasets = {}
        models = {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Clinical Survival Analysis")
st.sidebar.markdown("**4 models · 3 datasets · Real clinical data**")
st.sidebar.markdown("---")
dataset_choice = st.sidebar.selectbox(
    "Active dataset", ["whas500", "gbsg2", "icu"], index=2
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs(
    [
        "Dataset Overview",
        "KM Curves",
        "Model Comparison",
        "Patient Prediction",
        "Feature Importance",
        "Deep Learning Models",
    ]
)


# ── Tab 1: Dataset Overview ─────────────────────────────────────────────────
with tabs[0]:
    st.header("Cohort Statistics")
    if not DATA_LOADED:
        st.warning("Data not available.")
    else:
        ds = datasets[dataset_choice]
        raw = ds["raw"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Patients", f"{len(raw):,}")
        col2.metric("Events", f"{int(raw['event'].sum()):,}")
        col3.metric("Event rate", f"{raw['event'].mean():.1%}")
        col4.metric("Median follow-up", f"{raw['duration'].median():.0f} d")

        st.subheader("Follow-up time distribution")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(raw["duration"], bins=40, color="#4f8ef7", alpha=0.8, edgecolor="white")
        ax.set_xlabel("Days", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Feature summary")
        num_cols = [c for c in raw.columns if pd.api.types.is_numeric_dtype(raw[c]) and c not in ("event",)]
        st.dataframe(raw[num_cols].describe().round(2), use_container_width=True)


# ── Tab 2: KM Curves ────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Kaplan-Meier Survival Curves")
    if not DATA_LOADED:
        st.warning("Data not available.")
    else:
        ds = datasets[dataset_choice]
        raw = ds["raw"]

        cat_cols = [c for c in raw.columns if raw[c].dtype == object or c in ("age_group", "cci_level", "los_category")]
        subgroup_col = st.selectbox("Subgroup by", cat_cols if cat_cols else ["None"])

        from src.models.kaplan_meier import KaplanMeierModel
        from lifelines import KaplanMeierFitter

        km = KaplanMeierModel(label=dataset_choice)
        km.fit(raw)

        if subgroup_col and subgroup_col != "None" and subgroup_col in raw.columns:
            km.fit_subgroups(raw, subgroup_col)
            groups = km.subgroup_kmfs.get(subgroup_col, {})
        else:
            groups = {"Overall": km.overall_kmf}

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#4f8ef7", "#f97316", "#22c55e", "#a855f7", "#ef4444"]
        for (label, kmf), color in zip(groups.items(), colors):
            kmf.plot_survival_function(ax=ax, ci_show=True, color=color, label=label)
        ax.set_xlabel("Days")
        ax.set_ylabel("Survival Probability")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Median survival by subgroup")
        if subgroup_col and subgroup_col in raw.columns and subgroup_col in km.subgroup_kmfs:
            st.dataframe(km.subgroup_medians(subgroup_col), use_container_width=True)

        if km.logrank_results:
            st.subheader("Log-rank test")
            st.dataframe(km.logrank_summary(), use_container_width=True)


# ── Tab 3: Model Comparison ──────────────────────────────────────────────────
with tabs[2]:
    st.header("Model Comparison — C-index")
    if not DATA_LOADED:
        st.warning("Data not available.")
    else:
        rows = []
        for ds_name in ["whas500", "gbsg2", "icu"]:
            ds = datasets[ds_name]
            test = ds["test"]
            feat = ds["feature_cols"]

            for model_key, model_label in [
                (f"cox_{ds_name}", "Cox PH"),
                (f"rsf_{ds_name}", "Random Survival Forest"),
                (f"xgb_{ds_name}", "Gradient Boosting"),
            ]:
                mdl = models.get(model_key)
                if mdl is None:
                    continue
                try:
                    c = mdl.score(test)
                    rows.append({"Dataset": ds_name, "Model": model_label, "C-index": round(c, 4)})
                except Exception:
                    pass

        if rows:
            df_cmp = pd.DataFrame(rows)
            pivot = df_cmp.pivot(index="Model", columns="Dataset", values="C-index")
            st.dataframe(pivot.style.highlight_max(axis=0, color="#d1fae5"), use_container_width=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(9, 4))
            datasets_list = df_cmp["Dataset"].unique()
            models_list = df_cmp["Model"].unique()
            x = np.arange(len(datasets_list))
            width = 0.8 / len(models_list)
            offsets = np.linspace(-0.4 + width / 2, 0.4 - width / 2, len(models_list))
            colors = ["#4f8ef7", "#f97316", "#22c55e"]
            for offset, model_label, color in zip(offsets, models_list, colors):
                vals = [df_cmp[(df_cmp["Model"] == model_label) & (df_cmp["Dataset"] == ds)]["C-index"].values[0]
                        for ds in datasets_list
                        if len(df_cmp[(df_cmp["Model"] == model_label) & (df_cmp["Dataset"] == ds)]) > 0]
                ax.bar(x[:len(vals)] + offset, vals, width=width * 0.9, label=model_label, color=color, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(datasets_list)
            ax.set_ylabel("C-index")
            ax.set_ylim(0.4, 1.0)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)


# ── Tab 4: Patient Prediction ────────────────────────────────────────────────
with tabs[3]:
    st.header("Patient-Level Survival Prediction")
    if not DATA_LOADED:
        st.warning("Data not available.")
    else:
        st.markdown("Enter patient features to get a personalised survival curve.")

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 18, 99, 65)
            sofa = st.slider("SOFA score", 0, 24, 6)
            cci = st.slider("CCI", 0, 15, 3)
        with c2:
            creat = st.slider("Creatinine (mg/dL)", 0.3, 15.0, 1.0, step=0.1)
            lactate = st.slider("Lactate (mmol/L)", 0.5, 15.0, 1.5, step=0.1)
            map_v = st.slider("MAP (mmHg)", 40, 120, 75)
        with c3:
            los = st.slider("LOS (days)", 0.5, 30.0, 5.0, step=0.5)
            spo2 = st.slider("SpO2 (%)", 70, 100, 96)
            diabetes = st.checkbox("Diabetes")
            heart_failure = st.checkbox("Heart Failure")

        patient_row = pd.DataFrame([{
            "age": age, "los": los, "sofa_score": sofa, "creatinine": creat,
            "lactate": lactate, "map": map_v, "spo2": spo2,
            "diabetes": int(diabetes), "hypertension": 0, "heart_failure": int(heart_failure),
            "ckd": 0, "copd": 0, "cci": cci,
        }])

        ds = datasets["icu"]
        feat = ds["feature_cols"]
        scaler = ds["scaler"]
        available_feat = [f for f in feat if f in patient_row.columns]
        for f in feat:
            if f not in patient_row.columns:
                patient_row[f] = 0.0

        patient_scaled = patient_row.copy()
        patient_scaled[feat] = scaler.transform(patient_row[feat])

        rsf_mdl = models.get("rsf_icu")
        if rsf_mdl:
            times = list(range(0, 370, 10))
            surv = rsf_mdl.predict_survival(patient_scaled[feat], times=times)
            risk = float(rsf_mdl.predict_risk(patient_scaled[feat])[0])

            st.metric("Risk score", f"{risk:.2f}", help="Higher = higher risk")

            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(times, surv[0], color="#4f8ef7", linewidth=2.5, label="Patient survival curve")
            ax.fill_between(times, surv[0] * 0.95, np.minimum(surv[0] * 1.05, 1.0), alpha=0.2, color="#4f8ef7")
            ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="50% survival")
            ax.set_xlabel("Days")
            ax.set_ylabel("Survival Probability")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

            # Survival at key time points
            surv_table = rsf_mdl.predict_survival(patient_scaled[feat], times=[30, 90, 180, 365])
            st.markdown("**Survival probability at key time points:**")
            st.table(pd.DataFrame({
                "Time": ["30d", "90d", "180d", "365d"],
                "Survival probability": [f"{p:.2%}" for p in surv_table[0]],
            }))


# ── Tab 5: Feature Importance ────────────────────────────────────────────────
with tabs[4]:
    st.header("Feature Importance Comparison")
    if not DATA_LOADED:
        st.warning("Data not available.")
    else:
        rsf_mdl = models.get(f"rsf_{dataset_choice}")
        xgb_mdl = models.get(f"xgb_{dataset_choice}")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Random Survival Forest")
            if rsf_mdl:
                imp = rsf_mdl.feature_importance_df().head(12)
                fig, ax = plt.subplots(figsize=(7, 5))
                imp_sorted = imp.sort_values("importance")
                ax.barh(imp_sorted["feature"].astype(str), imp_sorted["importance"], color="#4f8ef7", alpha=0.85)
                ax.set_xlabel("MDI Importance")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(True, axis="x", alpha=0.3, linestyle="--")
                st.pyplot(fig)
                plt.close(fig)

        with c2:
            st.subheader("Gradient Boosting")
            if xgb_mdl:
                imp = xgb_mdl.feature_importance_df().head(12)
                fig, ax = plt.subplots(figsize=(7, 5))
                imp_sorted = imp.sort_values("importance")
                ax.barh(imp_sorted["feature"].astype(str), imp_sorted["importance"], color="#f97316", alpha=0.85)
                ax.set_xlabel("MDI Importance")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(True, axis="x", alpha=0.3, linestyle="--")
                st.pyplot(fig)
                plt.close(fig)


# ── Tab 6: Deep Learning Models ──────────────────────────────────────────────
with tabs[5]:
    st.header("Deep Learning Survival Models")
    st.markdown("DeepSurv and DeepHit trained on synthetic ICU data (n=500, fast demo).")

    @st.cache_resource(show_spinner=False)
    def _train_deep_models():
        """Train DeepSurv and DeepHit on a small ICU cohort for dashboard demo."""
        from src.data import generate_synthetic_icu
        from src.models.deep_surv import DeepSurvNet, DeepSurvTrainer
        from src.models.deep_hit import DeepHitNet, DeepHitTrainer
        from sklearn.preprocessing import StandardScaler

        df = generate_synthetic_icu(n_patients=500, seed=7)
        feat_cols = [
            "age", "los", "sofa_score", "creatinine",
            "lactate", "map", "spo2", "diabetes",
            "hypertension", "heart_failure", "ckd", "copd", "cci",
        ]
        feat_cols = [c for c in feat_cols if c in df.columns]
        X_raw = df[feat_cols].values.astype("float32")
        time = df["duration"].values.astype("float32")
        event = df["event"].values.astype("float32")

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw).astype("float32")

        # DeepSurv
        ds_net = DeepSurvNet(input_dim=X.shape[1], hidden_dims=[64, 32], dropout=0.3)
        ds_trainer = DeepSurvTrainer(ds_net, lr=5e-3, l2_reg=1e-4)
        ds_losses = ds_trainer.fit(X, time, event, n_epochs=60, batch_size=64)

        # DeepHit
        dh_net = DeepHitNet(input_dim=X.shape[1], num_time_bins=40, num_risks=1, hidden_dim=64)
        dh_trainer = DeepHitTrainer(dh_net, alpha=0.2, sigma=0.1, lr=5e-3)
        dh_losses = dh_trainer.fit(X, time, event, n_epochs=60, batch_size=64)

        return ds_trainer, dh_trainer, ds_losses, dh_losses, X, time, event

    if not DATA_LOADED:
        st.warning("Data not available.")
    else:
        try:
            with st.spinner("Training deep models (first run only, ~30s)..."):
                ds_trainer, dh_trainer, ds_losses, dh_losses, X_dl, T_dl, E_dl = _train_deep_models()

            # ── C-index comparison ──
            st.subheader("C-index Comparison")
            ds_c = ds_trainer.concordance_index(X_dl, T_dl, E_dl)
            dh_c = dh_trainer.concordance_index(X_dl, T_dl, E_dl)
            col1, col2 = st.columns(2)
            col1.metric("DeepSurv C-index", f"{ds_c:.4f}")
            col2.metric("DeepHit C-index", f"{dh_c:.4f}")

            # ── Loss curves ──
            st.subheader("Training Loss Curves")
            fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
            epochs = list(range(1, len(ds_losses) + 1))
            ax_loss.plot(epochs, ds_losses, color="#4f8ef7", linewidth=2, label="DeepSurv")
            ax_loss.plot(epochs, dh_losses, color="#f97316", linewidth=2, label="DeepHit")
            ax_loss.set_xlabel("Epoch", fontsize=11)
            ax_loss.set_ylabel("Loss", fontsize=11)
            ax_loss.set_title("Training Loss Over Epochs", fontsize=12)
            ax_loss.legend()
            ax_loss.grid(True, alpha=0.3, linestyle="--")
            ax_loss.spines["top"].set_visible(False)
            ax_loss.spines["right"].set_visible(False)
            st.pyplot(fig_loss)
            plt.close(fig_loss)

            # ── Sample patient survival curves ──
            st.subheader("Sample Patient Survival Curve")
            time_pts = list(range(0, 370, 10))
            sample_idx = 0
            X_sample = X_dl[[sample_idx]]

            # DeepSurv: risk score → survival via baseline hazard approximation
            # We produce a simple exponential approximation: S(t) = exp(-risk * t / scale)
            ds_risk = float(ds_trainer.predict_risk(X_dl).mean())
            ds_scale = float(T_dl.mean())
            ds_risk_sample = float(ds_trainer.predict_risk(X_sample)[0])
            ds_surv_sample = np.exp(-np.exp(ds_risk_sample - ds_risk) * np.array(time_pts) / ds_scale)

            # DeepHit: direct survival output
            dh_surv_sample = dh_trainer.predict_survival(X_sample, time_pts)[0]

            fig_surv, ax_surv = plt.subplots(figsize=(10, 4))
            ax_surv.plot(time_pts, ds_surv_sample, color="#4f8ef7", linewidth=2.5, label="DeepSurv")
            ax_surv.plot(time_pts, dh_surv_sample, color="#f97316", linewidth=2.5, label="DeepHit")
            ax_surv.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="50% threshold")
            ax_surv.set_xlabel("Days", fontsize=11)
            ax_surv.set_ylabel("Survival Probability", fontsize=11)
            ax_surv.set_title("Survival Curve — Sample Patient (index 0)", fontsize=12)
            ax_surv.set_ylim(0, 1.05)
            ax_surv.legend()
            ax_surv.grid(True, alpha=0.3, linestyle="--")
            ax_surv.spines["top"].set_visible(False)
            ax_surv.spines["right"].set_visible(False)
            st.pyplot(fig_surv)
            plt.close(fig_surv)

        except Exception as dl_exc:
            st.error(f"Deep learning models failed to load: {dl_exc}")
