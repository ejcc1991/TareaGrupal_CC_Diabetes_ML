from fastapi import FastAPI
from pydantic import BaseModel, Field # <--- IMPORTANTE: Agregar Field aquí
import joblib
import pandas as pd

app = FastAPI()

# Cargar el modelo (asegúrate de que el nombre del archivo sea el correcto)
model = joblib.load("random_forest_model.pkl")

class InputData(BaseModel):
    hba1c: float = Field(
        ..., 
        description="Hemoglobina Glicosilada (Promedio azúcar 3 meses). Rango normal: 4-5.6%", 
        ge=0, le=20, 
        example=5.5
    )
    glucose_postprandial: int = Field(
        ..., 
        description="Glucosa 2 horas después de comer (mg/dL).", 
        ge=0, le=500, 
        example=140
    )
    glucose_fasting: int = Field(
        ..., 
        description="Glucosa en ayunas (mg/dL).", 
        ge=0, le=500, 
        example=90
    )
    age: int = Field(
        ..., 
        description="Edad del paciente en años.", 
        ge=0, le=120, 
        example=35
    )
    bmi: float = Field(
        ..., 
        description="Índice de Masa Corporal (Peso/Altura²).", 
        ge=10, le=60, 
        example=24.5
    )
    systolic_bp: int = Field(
        ..., 
        description="Presión arterial sistólica (mm Hg).", 
        ge=50, le=250, 
        example=120
    )
    cholesterol_total: int = Field(
        ..., 
        description="Colesterol total (mg/dL).", 
        ge=50, le=500, 
        example=180
    )
    physical_activity_minutes_per_week: int = Field(
        ..., 
        description="Minutos de ejercicio a la semana.", 
        ge=0, le=10080, 
        example=150
    )

@app.get("/")
def home():
    return {"message": "API de Predicción de Diabetes funcionando correctamente"}

@app.post("/predict")
def predict(data: InputData):
    # Convertir los datos a DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Realizar la predicción
    prediction = model.predict(df)
    
    # Devolver el resultado
    result = "Positivo (Riesgo Alto)" if prediction[0] == 1 else "Negativo (Riesgo Bajo)"
    return {"prediccion": result, "datos_recibidos": data.dict()}