import streamlit as st
import numpy as np
import joblib

# Load model & scaler
model = joblib.load("phone_addiction_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="📱 Smartphone Addiction Predictor")

st.title("📱 Smartphone Addiction Prediction")
st.write("Fill in the details below to predict phone addiction level.")

st.divider()

# ---------------- USER INPUTS ----------------

daily_usage = st.number_input(
    "Daily Phone Usage (hours)",
    min_value=0.0,
    max_value=24.0,
    step=0.1
)

sleep_hours = st.number_input(
    "Sleep Hours per Day",
    min_value=0.0,
    max_value=24.0,
    step=0.1
)

phone_checks = st.number_input(
    "Phone Checks Per Day",
    min_value=0,
    step=1
)

apps_used = st.number_input(
    "Apps Used Daily",
    min_value=0,
    step=1
)

social_media_time = st.number_input(
    "Time on Social Media (hours/day)",
    min_value=0.0,
    max_value=24.0,
    step=0.1
)

gaming_time = st.number_input(
    "Time on Gaming (hours/day)",
    min_value=0.0,
    max_value=24.0,
    step=0.1
)

# ---------------- PREDICTION ----------------

if st.button("🔍 Predict Addiction Level"):

    # MUST match training feature order
    input_data = np.array([[
        daily_usage,
        sleep_hours,
        phone_checks,
        apps_used,
        social_media_time,
        gaming_time
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)


    # Predict
    prediction = model.predict(input_scaled)[0]

    # Output
    if prediction == 0:
        st.error("🔴 Addiction Level: HIGH")
    elif prediction == 1:
        st.warning("🟡 Addiction Level: MEDIUM")
    else:
        st.success("🟢 Addiction Level: LOW")
