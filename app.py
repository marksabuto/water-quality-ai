import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('water_quality_model.pkl')

# Page title
st.title("💧 Water Potability Predictor")
st.markdown("Enter water chemical parameters to check if it's safe to drink (Potable).")

# Input fields for features
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
solids = st.number_input("Solids", min_value=0.0, value=20000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=500.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

# Prediction
if st.button("Predict Potability"):
    features = np.array([[ph, hardness, solids, chloramines, sulfate,
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    prediction = model.predict(features)[0]
    result = "🚱 Not Safe to Drink" if prediction == 0 else "✅ Safe to Drink"
    st.subheader(f"Prediction: {result}")
