import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Configuración de la API
app = FastAPI(
    title="API de Diagnóstico de Diabetes (Detección Temprana)",
    description="Sistema experto con detección de Prediabetes, Diabetes y Factores de Riesgo.",
    version="3.0.0"
)

# 2. Carga de Modelos
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Sistema cargado: Modelo + Scaler listos.")
    else:
        print("Advertencia: No se encontraron archivos .pkl")
except Exception as e:
    print(f"Error al cargar modelos: {e}")

# 3. Input Data
class InputData(BaseModel):
    hba1c: float = Field(..., description="Hemoglobina Glicosilada (%)", example=5.8)
    glucose_postprandial: int = Field(..., description="Glucosa 2h post comida", example=145)
    glucose_fasting: int = Field(..., description="Glucosa en ayunas", example=110)
    age: int = Field(..., description="Edad", example=40)
    bmi: float = Field(..., description="IMC", example=27.5)
    systolic_bp: int = Field(..., description="Presión Sistólica", example=130)
    cholesterol_total: int = Field(..., description="Colesterol Total", example=200)
    physical_activity_minutes_per_week: int = Field(..., description="Minutos ejercicio", example=30)

# 4. Lógica de Explicabilidad (Factores de Riesgo)
def obtener_explicacion(data: InputData):
    razones = []
    
    # Análisis de HbA1c
    if data.hba1c >= 6.5:
        razones.append(f"HbA1c Crítica ({data.hba1c}%) - Rango diabético.")
    elif 5.7 <= data.hba1c < 6.5:
        razones.append(f"HbA1c Elevada ({data.hba1c}%) - Rango de Prediabetes.")

    # Análisis de Glucosa
    if data.glucose_fasting >= 126:
        razones.append(f"Glucosa Ayunas Alta ({data.glucose_fasting} mg/dL).")
    elif 100 <= data.glucose_fasting < 126:
        razones.append(f"Glucosa Ayunas Alterada ({data.glucose_fasting} mg/dL).")
        
    # Análisis de IMC
    if data.bmi >= 30:
        razones.append(f"Obesidad (IMC: {data.bmi}).")
    elif 25 <= data.bmi < 30:
        razones.append(f"Sobrepeso (IMC: {data.bmi}).")
    
    if not razones:
        razones.append("No se detectan factores de riesgo individuales críticos.")
        
    return razones


# 4. Endpoint de Health Check
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "diabetes-prediction-api",
        "version": "3.0.1",
        "model_ready": model is not None,
        "scaler_ready": scaler is not None,
        "message": "API lista para recibir datos en /predict"
    }


# 5. Endpoint de Predicción Inteligente
@app.post("/predict")
def predict(data: InputData):
    if not model or not scaler:
        raise HTTPException(status_code=503, detail="Modelo no disponible.")
    
    try:
        # A. Preprocesar
        cols = [
            "hba1c", "glucose_postprandial", "glucose_fasting", "age", 
            "bmi", "systolic_bp", "cholesterol_total", "physical_activity_minutes_per_week"
        ]
        df = pd.DataFrame([data.dict()])[cols]
        df_scaled = scaler.transform(df)
        
        # B. Probabilidad del Modelo
        probability = model.predict_proba(df_scaled)[0][1]
        
        # C. Lógica de 3 ZONAS (Prediabetes activado)
        # Umbral Alto (Diabetes): Usamos el 0.48 del entrenamiento
        UMBRAL_DIABETES = 0.4798
        # Umbral Bajo (Prediabetes): Definimos 0.30 para captar riesgo temprano
        UMBRAL_PREDIABETES = 0.30 
        
        if probability >= UMBRAL_DIABETES:
            # CASO ROJO: DIABETES
            diagnostico = "Positivo (Diabetes Tipo 2)"
            nivel_alerta = "ALTA"
            mensaje = "El modelo detecta patrones clínicos consistentes con diabetes."
            
        elif probability >= UMBRAL_PREDIABETES:
            # CASO AMARILLO: PREDIABETES (El nuevo punto medio)
            diagnostico = "Alerta: Prediabetes / Riesgo Elevado"
            nivel_alerta = "MEDIA"
            mensaje = "Zona de riesgo detectada. Se sugiere revisión médica preventiva."
            
        else:
            # CASO VERDE: SANO
            diagnostico = "Negativo (Sano)"
            nivel_alerta = "BAJA"
            mensaje = "No se detectan patrones de riesgo significativos."

        # D. Generar Explicación
        factores = obtener_explicacion(data)

        return {
            "resultado_diagnostico": diagnostico,
            "probabilidad_calculada": round(float(probability), 4),
            "nivel_alerta": nivel_alerta,
            "mensaje_clinico": mensaje,
            "factores_de_riesgo": factores,
            "meta_info": {
                "rango_prediabetes": f"{UMBRAL_PREDIABETES} - {UMBRAL_DIABETES}",
                "umbral_diabetes": UMBRAL_DIABETES
            }
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor.")