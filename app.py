# app.py — Streamlit frontend for NASA KOI preprocessed data (Overview, Explore, Predict)
# pip install streamlit pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm joblib

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from functools import lru_cache
from sklearn.pipeline import Pipeline  # used for detecting sklearn Pipelines

st.set_page_config(page_title="KOI Explorer", layout="wide")
sns.set(style="whitegrid")

# --------------------------------
# Known saved model candidates (Windows paths)
# --------------------------------
MODEL_CANDIDATES = [
    ("XGBoost", "./models/xg_boost_model.pkl"),
    ("Logistic Regression", "./models/logistic_regression_model.pkl"),
    ("LightGBM", "./models/lightgbm_model.pkl"),
]

# --------------------------------
# Data loading
# --------------------------------
DEFAULT_PATHS = [
    "data/kepler_koi.csv",
    "kepler_koi.csv",
    "data/kepler_koi_clean.csv",
    "/mnt/data/cumulative_2025.10.04_08.58.22_clean.csv",
    "/mnt/data/cumulative_2025.10.04_08.58.22.csv",
]

@lru_cache(maxsize=1)
def load_df(path: str, has_comments: bool) -> pd.DataFrame:
    if has_comments:
        return pd.read_csv(path, comment="#")
    return pd.read_csv(path)

def auto_load() -> pd.DataFrame:
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            try:
                if p.endswith(".csv"):
                    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                        first = fh.readline()
                    return load_df(p, has_comments=first.startswith("#"))
            except Exception:
                pass
    return pd.DataFrame()

st.sidebar.header("Load data")
uploaded = st.sidebar.file_uploader("Upload preprocessed CSV (optional)", type=["csv"])

if uploaded is not None:
    raw = uploaded.getvalue().decode("utf-8", errors="ignore")
    has_hash = raw.lstrip().startswith("#")
    df = pd.read_csv(io.StringIO(raw), comment="#" if has_hash else None)
    src_label = "Uploaded file"
else:
    df = auto_load()
    src_label = "Auto-loaded" if not df.empty else "No file found"

st.sidebar.caption(f"Source: **{src_label}**")

if df.empty:
    st.warning(
        "Could not find a dataset. Please upload `data/kepler_koi.csv` "
        "or a cleaned file with KOI columns."
    )
    st.stop()

# Normalize column names
df.columns = df.columns.str.strip()

# --------------------------------
# Sidebar filters & options
# --------------------------------
st.sidebar.markdown("### Filters")
dispo_vals = sorted(df["koi_pdisposition"].dropna().unique()) if "koi_pdisposition" in df.columns else []
pick_dispo = st.sidebar.multiselect("koi_pdisposition", dispo_vals, default=dispo_vals)

if pick_dispo and "koi_pdisposition" in df.columns:
    df = df[df["koi_pdisposition"].isin(pick_dispo)].copy()

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
default_x = "koi_period" if "koi_period" in num_cols else (num_cols[0] if num_cols else None)
default_y = "koi_prad" if "koi_prad" in num_cols else (num_cols[1] if len(num_cols) > 1 else None)

# --------------------------------
# Tabs (Modeling removed)
# --------------------------------
tab_overview, tab_explore, tab_predict = st.tabs(["Overview", "Explore", "Predict"])

# --------------------------------
# Overview
# --------------------------------
with tab_overview:
    st.title("KOI Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    if "koi_disposition" in df.columns:
        dispo_counts = df["koi_disposition"].value_counts()
        c3.metric("Confirmed", int(dispo_counts.get("CONFIRMED", 0)))
        c4.metric("Candidates", int(dispo_counts.get("CANDIDATE", 0)))

    with st.expander("Preview data", expanded=True):
        st.dataframe(df.head(25), use_container_width=True)

    with st.expander("Missingness (top 20)"):
        miss = df.isna().mean().sort_values(ascending=False).head(20)
        st.dataframe(miss.to_frame("null_fraction"), use_container_width=True)

    with st.expander("Descriptive stats (selected)"):
        keys = [c for c in ["koi_period", "koi_prad", "koi_teq", "koi_insol", "koi_model_snr", "koi_depth"] if c in df.columns]
        if keys:
            st.dataframe(df[keys].describe().T, use_container_width=True)
        else:
            st.info("No expected numeric columns found; showing full describe.")
            st.dataframe(df.describe(include="all").T, use_container_width=True)

# --------------------------------
# Explore
# --------------------------------
with tab_explore:
    st.header("Interactive EDA")

    ec1, ec2 = st.columns(2)
    with ec1:
        x_col = st.selectbox("X (numeric)", options=num_cols, index=max(num_cols.index(default_x), 0) if default_x in num_cols else 0)
    with ec2:
        y_col = st.selectbox("Y (numeric)", options=num_cols, index=max(num_cols.index(default_y), 1) if default_y in num_cols else 1)

    color_col = "koi_disposition" if "koi_disposition" in df.columns else None

    st.subheader("Scatter")
    fig, ax = plt.subplots()
    if color_col:
        for label, group in df[[x_col, y_col, color_col]].dropna().groupby(color_col):
            ax.scatter(group[x_col], group[y_col], label=str(label), alpha=0.6, s=18)
        ax.legend(loc="best", fontsize="small")
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.6, s=18)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Histogram")
    hist_col = st.selectbox("Histogram column", options=num_cols, index=max(num_cols.index(default_y), 0) if default_y in num_cols else 0)
    bins = st.slider("Bins", 10, 100, 40, step=5)
    fig2, ax2 = plt.subplots()
    ax2.hist(df[hist_col].dropna(), bins=bins)
    ax2.set_xlabel(hist_col); ax2.set_ylabel("Count"); ax2.set_title(f"Distribution of {hist_col}")
    st.pyplot(fig2, clear_figure=True)

    st.subheader("Correlation heatmap (select up to 15)")
    heat_cols = st.multiselect("Columns", options=num_cols, default=[c for c in ["koi_period","koi_prad","koi_teq","koi_insol","koi_model_snr","koi_depth"] if c in num_cols])
    if len(heat_cols) >= 2:
        fig3, ax3 = plt.subplots(figsize=(min(12, len(heat_cols)), min(8, len(heat_cols))))
        sns.heatmap(df[heat_cols].corr(numeric_only=True), annot=True, fmt=".2f", ax=ax3)
        ax3.set_title("Correlation Heatmap")
        st.pyplot(fig3, clear_figure=True)
    else:
        st.info("Pick at least 2 numeric columns for a heatmap.")

# --------------------------------
# Predict (use SAVED .pkl from disk or upload)
# --------------------------------
with tab_predict:
    st.header("Single-row Prediction")

    # Default 16 features (used if meta/features can't be inferred)
    DEFAULT_FEATURES = [
        "koi_pdisposition","koi_score","koi_fpflag_nt","koi_fpflag_ss",
        "koi_fpflag_co","koi_fpflag_ec","koi_period","koi_impact","koi_depth",
        "koi_prad","koi_teq","koi_insol","koi_model_snr","koi_steff","ra","dec"
    ]

    # If uploaded/saved model outputs 0/1, map using this switch:
    scheme = st.radio(
        "If your uploaded model outputs 0/1, map to:",
        ["Candidate vs False Positive"],
        horizontal=True
    )
    zero_one_map = ({1: "CONFIRMED", 0: "NOT"}
                    if scheme == "Confirmed vs Not"
                    else {1: "CANDIDATE", 0: "FALSE POSITIVE"})

    # ---------- Sidebar: choose model source ----------
    st.sidebar.markdown("### Predict tab — choose a saved model")

    existing_disk_models = [(name, path) for name, path in MODEL_CANDIDATES if os.path.exists(path)]
    model_source = st.sidebar.radio(
        "Model source",
        ["Pick from disk", "Upload .pkl"],
        index=0 if existing_disk_models else 1
    )

    loaded_model = None
    loaded_meta = None
    FEATURES = DEFAULT_FEATURES[:]  # start with default

    if model_source == "Pick from disk":
        if not existing_disk_models:
            st.sidebar.warning("No known .pkl model found at the preset paths. Use 'Upload .pkl' instead.")
        else:
            pick_name = st.sidebar.selectbox(
                "Select saved model",
                [f"{name} — {path}" for name, path in existing_disk_models],
                index=0
            )
            sel_idx = [f"{n} — {p}" for n, p in existing_disk_models].index(pick_name)
            chosen_name, chosen_path = existing_disk_models[sel_idx]
            try:
                loaded_model = joblib.load(chosen_path)
                st.sidebar.success(f"Loaded: {chosen_name}")
            except Exception as e:
                st.sidebar.error(f"Failed to load model: {e}")

            # Optional: attempt to load adjacent meta JSON
            meta_guess = chosen_path.replace(".pkl", "__meta.json")
            if os.path.exists(meta_guess):
                try:
                    loaded_meta = json.loads(open(meta_guess, "r", encoding="utf-8").read())
                    if isinstance(loaded_meta, dict) and "features" in loaded_meta and loaded_meta["features"]:
                        FEATURES = list(loaded_meta["features"])
                    st.sidebar.caption(f"Loaded meta: {meta_guess}")
                except Exception:
                    pass

    else:
        up_pkl = st.sidebar.file_uploader("Upload model .pkl", type=["pkl"], key="pkl_predict")
        up_meta = st.sidebar.file_uploader("Upload metadata .json (optional)", type=["json"], key="meta_predict")
        if up_pkl is not None:
            try:
                loaded_model = joblib.load(io.BytesIO(up_pkl.getvalue()))
                st.sidebar.success("Model loaded.")
            except Exception as e:
                st.sidebar.error(f"Failed to load .pkl: {e}")
        if up_meta is not None:
            try:
                loaded_meta = json.loads(up_meta.getvalue().decode("utf-8"))
                if isinstance(loaded_meta, dict) and "features" in loaded_meta and loaded_meta["features"]:
                    FEATURES = list(loaded_meta["features"])
                st.sidebar.success("Meta loaded.")
            except Exception as e:
                st.sidebar.error(f"Failed to parse meta: {e}")

    # ---------- Try to auto-detect expected features from model ----------
    def autodetect_features_from_model(model):
        try:
            if hasattr(model, "feature_names_in_"):
                return list(model.feature_names_in_)
            if isinstance(model, Pipeline):
                last = model.named_steps.get("clf", None) or list(model.named_steps.values())[-1]
                if hasattr(last, "feature_names_in_"):
                    return list(last.feature_names_in_)
        except Exception:
            pass
        return None

    auto_feats = autodetect_features_from_model(loaded_model) if loaded_model is not None else None
    if auto_feats:
        FEATURES = auto_feats

    # ---------- Allow manual override ----------
    # with st.expander("Advanced: override feature list (comma-separated)"):
    #     manual_feats = st.text_input("Feature columns", value="")
    #     if manual_feats.strip():
    #         FEATURES = [c.strip() for c in manual_feats.split(",") if c.strip()]

    # ---------- Friendly labels/tooltips ----------
    LABELS = {
        "koi_pdisposition": "Preliminary disposition (pipeline label)",
        "koi_score":        "KOI score (0–1)",
        "koi_fpflag_nt":    "FP flag — Not a transit (0/1)",
        "koi_fpflag_ss":    "FP flag — Stellar variability/systematic (0/1)",
        "koi_fpflag_co":    "FP flag — Centroid offset (0/1)",
        "koi_fpflag_ec":    "FP flag — Eclipsing binary candidate (0/1)",
        "koi_period":       "Orbital period (days)",
        "koi_impact":       "Transit impact parameter (0–1+)",
        "koi_depth":        "Transit depth (ppm)",
        "koi_prad":         "Planet radius (Earth radii)",
        "koi_teq":          "Equilibrium temperature (K)",
        "koi_insol":        "Insolation (× Earth)",
        "koi_model_snr":    "Transit model SNR",
        "koi_steff":        "Stellar effective temperature (K)",
        "ra":               "Right ascension (deg)",
        "dec":              "Declination (deg)",
    }
    HELP = {
        "koi_pdisposition": "Pipeline’s preliminary class for the signal (not the final disposition we’re predicting).",
        "koi_score":        "Confidence score from the KOI pipeline; higher can indicate a more planet-like signal.",
        "koi_fpflag_nt":    "1 if analysis suggests the event is not due to a planetary transit.",
        "koi_fpflag_ss":    "1 if stellar variability or systematics likely explain the signal.",
        "koi_fpflag_co":    "1 if transit source appears offset from target star.",
        "koi_fpflag_ec":    "1 if the signal resembles an eclipsing binary.",
        "koi_period":       "Time between transits (days).",
        "koi_impact":       "Transit chord distance from stellar center in R⋆.",
        "koi_depth":        "Transit depth in parts-per-million (ppm).",
        "koi_prad":         "Estimated planetary radius in Earth radii.",
        "koi_teq":          "Zero-albedo equilibrium temperature (K).",
        "koi_insol":        "Stellar irradiation relative to Earth (=1).",
        "koi_model_snr":    "Signal-to-noise of the fitted transit model.",
        "koi_steff":        "Host star effective temperature (K).",
        "ra":               "Right ascension (degrees).",
        "dec":              "Declination (degrees).",
    }

    # ---------- Build the input form dynamically ----------
    with st.form("predict_form"):
        cols = st.columns(4)
        inputs = {}

        for i, feat in enumerate(FEATURES):
            col = cols[i % 4]
            label = LABELS.get(feat, feat)
            help_txt = HELP.get(feat, "")

            if feat == "koi_pdisposition":
                inputs[feat] = col.selectbox(label, ["CANDIDATE","FALSE POSITIVE","CONFIRMED"], index=0, help=help_txt)
            elif feat.startswith("koi_fpflag_"):
                inputs[feat] = col.selectbox(label, [0, 1], index=0, help=help_txt)
            else:
                default_vals = {
                    "koi_score": 0.5, "koi_period": 10.0, "koi_impact": 0.5, "koi_depth": 500.0,
                    "koi_prad": 2.0, "koi_teq": 300.0, "koi_insol": 1.0, "koi_model_snr": 10.0,
                    "koi_steff": 5700.0, "ra": 290.0, "dec": 48.0
                }
                step = 0.01 if feat in ["koi_prad","koi_insol","koi_impact","koi_score"] else (0.1 if feat in ["koi_model_snr","koi_period"] else 1.0)
                if feat in ["ra", "dec"]:
                    inputs[feat] = col.number_input(label, value=float(default_vals.get(feat, 0.0)), step=step, help=help_txt)
                else:
                    inputs[feat] = col.number_input(label, min_value=0.0, value=float(default_vals.get(feat, 0.0)), step=step, help=help_txt)

        submitted = st.form_submit_button("Predict disposition")

    # ---------- Helper: basic single-row imputation for non-Pipeline models ----------
    def basic_clean_row(row_df: pd.DataFrame) -> pd.DataFrame:
        for c in row_df.columns:
            if row_df[c].dtype.kind in "biufc":  # numeric
                row_df[c] = row_df[c].fillna(0.0)
            else:
                row_df[c] = row_df[c].fillna("CANDIDATE")
        return row_df

    if submitted:
        if loaded_model is None:
            st.error("No model is loaded. Please pick one from disk or upload a .pkl.")
            st.stop()

        # Build a single-row DataFrame in the expected column order
        row = {}
        for feat in FEATURES:
            v = inputs[feat]
            if isinstance(v, str):
                row[feat] = v
            elif isinstance(v, (int, float, np.integer, np.floating)):
                row[feat] = float(v)
            else:
                row[feat] = v
        row_df = pd.DataFrame([row], columns=FEATURES)

        try:
            model_to_use = loaded_model

            # If it's a Pipeline, pass DataFrame; preprocessing should be inside.
            if isinstance(model_to_use, Pipeline):
                pred = model_to_use.predict(row_df)[0]
                # Try to keep string labels if model has them
                label_text = pred if isinstance(pred, str) else None
                if label_text is None:
                    try:
                        classes = model_to_use.named_steps.get("clf", model_to_use).classes_
                        if all(isinstance(c, str) for c in classes):
                            try:
                                label_text = classes[int(pred)]
                            except Exception:
                                label_text = str(pred)
                    except Exception:
                        label_text = str(pred)
                st.success(f"**Predicted koi_disposition:** {label_text}")

                # Probabilities
                if hasattr(model_to_use, "predict_proba"):
                    proba = model_to_use.predict_proba(row_df)[0]
                    try:
                        classes = model_to_use.named_steps.get("clf", model_to_use).classes_
                    except Exception:
                        classes = None
                    if classes is not None and all(isinstance(c, str) for c in classes) and len(classes) == len(proba):
                        order = np.argsort(proba)[::-1]
                        st.caption("Probabilities:")
                        for idx in order:
                            st.write(f"- {classes[idx]}: {proba[idx]:.3f}")
                    else:
                        st.caption(f"Probabilities: {proba}")

            else:
                # Bare estimator (no preprocessing inside). Do minimal cleaning.
                if "koi_pdisposition" in row_df.columns:
                    row_df["koi_pdisposition"] = row_df["koi_pdisposition"].astype(str)
                row_df = basic_clean_row(row_df)

                pred = model_to_use.predict(row_df)[0]

                # Convert to human-readable label
                label_text = None
                classes = getattr(model_to_use, "classes_", None)

                if isinstance(pred, str):
                    label_text = pred
                elif classes is not None and all(isinstance(c, str) for c in classes):
                    try:
                        label_text = classes[int(pred)]
                    except Exception:
                        label_text = str(pred)
                else:
                    # Fall back to radio-selected scheme for 0/1 outputs
                    try:
                        label_text = zero_one_map.get(int(pred), str(pred))
                    except Exception:
                        label_text = str(pred)

                st.success(f"**Predicted koi_disposition:** {label_text}")

                # Probabilities
                if hasattr(model_to_use, "predict_proba"):
                    proba = model_to_use.predict_proba(row_df)[0]
                    classes = getattr(model_to_use, "classes_", None)
                    if classes is not None and all(isinstance(c, str) for c in classes) and len(classes) == len(proba):
                        order = np.argsort(proba)[::-1]
                        st.caption("Probabilities:")
                        for idx in order:
                            st.write(f"- {classes[idx]}: {proba[idx]:.3f}")
                    else:
                        st.caption(f"Probabilities: {proba}")

        except Exception as e:
            st.error(
                "Prediction failed. If your saved model was trained on a specific set/order of features, "
                "please provide the same list (use the 'Advanced: override feature list' box), "
                "or save your model as a sklearn Pipeline that includes preprocessing.\n\n"
                f"Details: {e}"
            )

# --------------------------------
# Footer
# --------------------------------
st.caption("KOI Explorer • Streamlit frontend (Modeling tab removed).")
