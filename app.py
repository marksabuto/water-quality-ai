import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load("water_quality_model.pkl")

# Load dataset for visualizations
@st.cache_data
def load_data():
    return pd.read_csv("data/water_potability.csv")
data = load_data()

# App title
st.set_page_config(page_title="Water Potability Predictor", page_icon="💧")
st.title("💧 Water Potability Predictor")
st.markdown("Enter chemical values to check if the water is **safe to drink**.")

# Sidebar: Prediction settings and water safety tips
st.sidebar.header("⚙️ Prediction Settings")
threshold = st.sidebar.slider(
    "Set Prediction Threshold", 0.0, 1.0, 0.45, 0.01,
    help="Lower threshold = more likely to classify as Safe. Adjust to control model sensitivity."
)

st.sidebar.markdown("---")
st.sidebar.header("💡 Water Safety Tips")
st.sidebar.info(
    """
- WHO recommends pH 6.5–8.5 for drinking water.
- High turbidity or solids may indicate contamination.
- If unsure, always boil or treat water before drinking.
- [WHO Water Quality Guidelines](https://www.who.int/publications/i/item/9789241549950)
    """
)

st.sidebar.markdown("---")
st.sidebar.header("❓ FAQ")
st.sidebar.markdown(
    """
**Q: What does 'Safe to Drink' mean?**  
A: The model predicts if water is likely potable based on chemical properties. Always confirm with certified labs for critical use.

**Q: What if my value is out of range?**  
A: Enter values within typical ranges. Outliers may reduce prediction reliability.
    """
)

# Input fields with tooltips
ph = st.number_input(
    "pH", min_value=0.0, max_value=14.0, value=7.0,
    help="Acidity/alkalinity (0-14). WHO recommends 6.5–8.5."
)
hardness = st.number_input(
    "Hardness (mg/L)", min_value=0.0, value=150.0,
    help="Calcium and magnesium content. Typical: 80–300 mg/L."
)
solids = st.number_input(
    "Solids (ppm)", min_value=0.0, value=20000.0,
    help="Total dissolved solids. Typical: <500 ppm is desirable."
)
chloramines = st.number_input(
    "Chloramines (ppm)", min_value=0.0, value=7.0,
    help="Disinfectant. Typical: <4 ppm."
)
sulfate = st.number_input(
    "Sulfate (mg/L)", min_value=0.0, value=300.0,
    help="Sulfate ions. WHO guideline: <500 mg/L."
)
conductivity = st.number_input(
    "Conductivity (μS/cm)", min_value=0.0, value=500.0,
    help="Electrical conductivity. Typical: 50–1500 μS/cm."
)
organic_carbon = st.number_input(
    "Organic Carbon (ppm)", min_value=0.0, value=10.0,
    help="Organic carbon content. Typical: 2–15 ppm."
)
trihalomethanes = st.number_input(
    "Trihalomethanes (μg/L)", min_value=0.0, value=60.0,
    help="Byproducts of disinfection. WHO guideline: <100 μg/L."
)
turbidity = st.number_input(
    "Turbidity (NTU)", min_value=0.0, value=4.0,
    help="Cloudiness. WHO guideline: <5 NTU."
)

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

# ---
# Feature importance (placeholder if not available)
try:
    importances = model.feature_importances_
    feature_names = [
        "pH", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic Carbon", "Trihalomethanes", "Turbidity"
    ]
    st.markdown("### 🔍 Model Feature Importance")
    st.bar_chart(pd.Series(importances, index=feature_names))
except AttributeError:
    st.info("Feature importance is not available for this model type.")

# ---
# Dataset visualizations
with st.expander("📊 Explore Water Quality Data (Sample)"):
    st.write("**Sample of the dataset:**")
    st.dataframe(data.sample(10))
    st.write("**Potability Distribution:**")
    potability_counts = data["Potability"].value_counts().rename({0: "Not Safe", 1: "Safe"})
    st.bar_chart(potability_counts)
    st.write("**Feature Distributions:**")
    selected_feature = st.selectbox(
        "Select a feature to view its distribution:",
        [col for col in data.columns if col != "Potability"]
    )
    fig, ax = plt.subplots()
    data[selected_feature].hist(bins=30, ax=ax)
    ax.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig)
