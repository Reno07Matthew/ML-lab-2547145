"""Non-diagnostic Streamlit demonstration for the fixed selected pipeline."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL = joblib.load(ROOT / "models" / "extended_low_cost__logistic_regression.joblib")
SHAP_REFERENCE = json.loads((ROOT / "reports" / "shap_background.json").read_text())


def select(label, choices):
    shown = list(choices)
    return choices[st.selectbox(label, shown)]


def transformed_row(frame: pd.DataFrame) -> np.ndarray:
    intermediate = MODEL.named_steps["before_sampling"].transform(frame)
    matrix = MODEL.named_steps["after_sampling"].transform(intermediate)
    return matrix.toarray()[0] if hasattr(matrix, "toarray") else np.asarray(matrix)[0]


st.set_page_config(page_title="BRFSS screening-score demonstration", page_icon="🫀")
st.title("Premature CHD/MI profile screening-score demonstration")
st.error("Research demonstration only. This is not a diagnosis, medical probability, or substitute for clinical care. A low score does not rule out disease, and a high score does not establish disease.")
st.caption("The model was developed from cross-sectional, self-reported US BRFSS 2023 data and has not been validated for India or for individual clinical decisions.")
st.caption(f"Fixed validation-selected F2 threshold: {SHAP_REFERENCE['f2_threshold']:.6f} ({100 * SHAP_REFERENCE['f2_threshold']:.4f}/100 screening-score units).")

with st.form("profile"):
    left, right = st.columns(2)
    with left:
        age = st.slider("Age", 18, 39, 28)
        sex = select("Sex variable", {"Male": "1", "Female": "2"})
        general = select("General health", {"Good": "3", "Very good": "2", "Excellent": "1", "Fair": "4", "Poor": "5"})
        physical_days = st.slider("Physical-health days not good (past 30)", 0, 30, 0)
        mental_days = st.slider("Mental-health days not good (past 30)", 0, 30, 0)
        exercise = select("Any exercise in the past 30 days", {"Yes": "1", "No": "2"})
        bmi = st.number_input("BMI", 10.0, 70.0, 24.0, 0.1)
        smoking = select("Smoking status", {"Never": "4", "Former": "3", "Some days": "2", "Every day": "1"})
        alcohol = select("Any alcohol in the past 30 days", {"No": "2", "Yes": "1"})
    with right:
        activity = select("Calculated leisure-time physical activity", {"Active": "1", "Inactive": "2"})
        education = select("Education", {"Some college/technical school": "3", "College/technical graduate": "4", "High-school graduate": "2", "Did not graduate high school": "1"})
        income = select("Household income", {"$50k-<$100k": "5", "$35k-<$50k": "4", "$25k-<$35k": "3", "$15k-<$25k": "2", "< $15k": "1", "$100k-<$200k": "6", "$200k+": "7"})
        blood_pressure = select("Blood-pressure history", {"No": "3", "Borderline/elevated": "4", "High blood pressure": "1", "Pregnancy only": "2"})
        cholesterol = select("Cholesterol status/check history", {"Not high": "not_high", "High": "high", "Never checked": "never_checked", "Unknown/not sure": "unknown_not_sure", "Refused": "refused"})
        diabetes = select("Diabetes history", {"No": "3", "Pre-diabetes/borderline": "4", "Diabetes": "1", "Pregnancy only": "2"})
        kidney = select("Kidney-disease history", {"No": "2", "Yes": "1"})
        walking = select("Serious difficulty walking/climbing stairs", {"No": "2", "Yes": "1"})
    submitted = st.form_submit_button("Generate research screening score")

if submitted:
    frame = pd.DataFrame([{
        "_AGE80": age, "_SEX": sex, "GENHLTH": general, "PHYSHLTH": physical_days,
        "MENTHLTH": mental_days, "EXERANY2": exercise, "_BMI5": bmi, "_SMOKER3": smoking,
        "DRNKANY6": alcohol, "_TOTINDA": activity, "_EDUCAG": education, "_INCOMG1": income,
        "BPHIGH6": blood_pressure, "cholesterol_status": cholesterol, "DIABETE4": diabetes,
        "CHCKDNY2": kidney, "DIFFWALK": walking,
    }])
    raw_score = float(MODEL.predict_proba(frame)[0, 1])
    score_points = round(100 * raw_score)
    threshold_points = 100 * SHAP_REFERENCE["f2_threshold"]
    st.metric("Research screening score", f"{score_points} / 100")
    if raw_score >= SHAP_REFERENCE["f2_threshold"]:
        st.warning(f"At or above the research F2 operating point ({threshold_points:.4f}/100). This does not diagnose disease; appropriate professional assessment is still required.")
    else:
        st.info(f"Below the research F2 operating point ({threshold_points:.4f}/100). This does not rule out disease or mean that medical assessment is unnecessary.")

    transformed = transformed_row(frame)
    coefficients = MODEL.named_steps["model"].coef_[0]
    contributions = (transformed - np.asarray(SHAP_REFERENCE["background_mean"])) * coefficients
    grouped = pd.DataFrame({"feature": SHAP_REFERENCE["original_features"], "contribution": contributions}).groupby("feature", as_index=False).contribution.sum()
    grouped["magnitude"] = grouped.contribution.abs()
    grouped = grouped.nlargest(6, "magnitude")
    grouped["direction"] = np.where(grouped.contribution >= 0, "Raised model score", "Lowered model score")
    st.subheader("Top model-attribution factors")
    st.dataframe(grouped[["feature", "direction", "contribution"]].rename(columns={"contribution": "SHAP contribution (log-odds)"}), hide_index=True)
    st.caption("These are model attributions relative to the analysis background, not causal effects or medical advice. Weak calibration means the score must not be interpreted as a personal probability.")
