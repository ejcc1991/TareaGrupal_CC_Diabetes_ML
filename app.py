from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

app = FastAPI(title="API Diabetes UAI", version="1.0")

# --- CORRECCIÓN: Rutas directas (están en la misma carpeta) ---
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# Cargar modelos
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    model = None
    scaler = None
    print(f"Error cargando modelos: {e}")

class PatientData(BaseModel):
    hba1c: float
    glucose_postprandial: float
    glucose_fasting: float
    age: int
    bmi: float
    systolic_bp: float
    cholesterol_total: float
    physical_activity_minutes_per_week: float

@app.get("/")
def home():
    return {"status": "ok", "message": "API Diabetes Online (GCP)"}

@app.post("/predict")
def predict(data: PatientData):
    if not model:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    
    df = pd.DataFrame([data.dict()])
    df_scaled = scaler.transform(df)
    
    pred = model.predict(df_scaled)[0]
    prob = model.predict_proba(df_scaled)[0][1]
    
    return {
        "prediccion": int(pred),
        "probabilidad": round(float(prob), 4),
        "mensaje": "Alto Riesgo" if pred == 1 else "Bajo Riesgo"
    }