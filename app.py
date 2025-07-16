import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
import plotly.graph_objects as go

# Load the model
model = joblib.load("water_quality_model.pkl")

# App config
st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="centered")

# --- Custom Logo/Header ---
st.markdown("""
    <div style="text-align: center;">
        <img src="https://img.icons8.com/fluency/96/000000/water.png" width="80"/>
        <h1 style="color:#00BFFF;">Water Potability Predictor 💧</h1>
        <p>Using AI to support <strong>UN SDG 6: Clean Water and Sanitation</strong></p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Prediction Threshold", 0.0, 1.0, 0.45, 0.01)
show_dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")
show_input_chart = st.sidebar.checkbox("📊 Show Input Chart")

# Apply dark mode theme
if show_dark_mode:
    st.markdown(
        """
        <style>
        body { background-color: #111; color: #EEE; }
        .stApp { background-color: #111; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- User Inputs ---
st.subheader("🔢 Enter Water Chemical Properties")

ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=150.0)
solids = st.number_input("Solids (ppm)", min_value=0.0, value=20000.0)
chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0)
conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=500.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=60.0)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0)

features = np.array([[ph, hardness, solids, chloramines, sulfate,
                      conductivity, organic_carbon, trihalomethanes, turbidity]])

# --- Show Input Chart ---
if show_input_chart:
    st.markdown("### 🔍 Input Overview")
    input_df = pd.DataFrame(features, columns=[
        "pH", "Hardness", "Solids", "Chloramines", "Sulfate", 
        "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"
    ])
    st.bar_chart(input_df.T)

# --- Prediction ---
if st.button("🔍 Predict Potability"):
    proba = model.predict_proba(features)[0][1]
    is_safe = proba >= threshold
    result = "✅ Safe to Drink" if is_safe else "🚱 Not Safe to Drink"
    
    st.subheader(f"🧠 Prediction: {result}")
    st.caption(f"Confidence (Safe): **{proba:.2f}**")

    # --- Gauge Meter ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba,
        title={'text': "Probability Water is Safe"},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "#00BFFF" if is_safe else "#FF6347"},
            'steps': [
                {'range': [0, 0.5], 'color': "#FFD2D2"},
                {'range': [0.5, 1], 'color': "#D2FFF0"}
            ]
        }
    ))
    st.plotly_chart(fig)

# --- Download Model ---
def generate_download_link(file_path, file_label="Download Model"):
    with open(file_path, "rb") as f:
        bytes_data = f.read()
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="water_quality_model.pkl">{file_label}</a>'
    return href

st.sidebar.markdown("📦 Download")
st.sidebar.markdown(generate_download_link("water_quality_model.pkl", "⬇️ Download Trained Model"), unsafe_allow_html=True)
