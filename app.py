import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("water_quality_model.pkl")

# App title
st.set_page_config(page_title="Water Potability Predictor", page_icon="💧")
st.title("💧 Water Potability Predictor")
st.markdown("Enter chemical values to check if the water is **safe to drink**.")

st.sidebar.header("⚙️ Prediction Settings")
threshold = st.sidebar.slider("Set Prediction Threshold", 0.0, 1.0, 0.45, 0.01)
st.sidebar.caption("Lower threshold = more likely to classify as Safe.")

# Input fields
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=150.0)
solids = st.number_input("Solids (ppm)", min_value=0.0, value=20000.0)
chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0)
conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=500.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=60.0)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0)

# Predict button
if st.button("Predict Potability"):
    # Prepare input for prediction
    features = np.array([[ph, hardness, solids, chloramines, sulfate,
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    # Predict probability of being safe (class 1)
    proba = model.predict_proba(features)[0][1]

    # Compare with threshold
    if proba >= threshold:
        result = "✅ Safe to Drink"
    else:
        result = "🚱 Not Safe to Drink"

    # Display result and confidence
    st.subheader(f"Prediction: {result}")
    st.caption(f"Confidence (Safe): {proba:.2f}")
