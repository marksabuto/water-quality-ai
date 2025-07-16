from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load model and scaler
model_bundle = joblib.load("water_quality_model.pkl")
model = model_bundle['model']
scaler = model_bundle['scaler']

app = FastAPI(title="Water Potability Prediction API")

class WaterSample(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.post("/predict")
def predict(sample: WaterSample):
    features = np.array([[sample.ph, sample.Hardness, sample.Solids, sample.Chloramines,
                         sample.Sulfate, sample.Conductivity, sample.Organic_carbon,
                         sample.Trihalomethanes, sample.Turbidity]])
    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0][1]
    pred = int(proba >= 0.5)
    return {
        "prediction": pred,
        "probability_safe": proba,
        "label": "Safe" if pred == 1 else "Not Safe"
    }

"""
Example request:
POST /predict
{
  "ph": 7.0,
  "Hardness": 150.0,
  "Solids": 20000.0,
  "Chloramines": 7.0,
  "Sulfate": 300.0,
  "Conductivity": 500.0,
  "Organic_carbon": 10.0,
  "Trihalomethanes": 60.0,
  "Turbidity": 4.0
}

Example response:
{
  "prediction": 1,
  "probability_safe": 0.82,
  "label": "Safe"
}
""" 