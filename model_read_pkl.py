import pickle
import pandas as pd
import numpy as np

# 1. Intentar cargar el Modelo y el Scaler
print("--- Iniciando prueba de carga ---")
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Archivos .pkl cargados correctamente.")
except FileNotFoundError:
    print("❌ ERROR: No se encuentran los archivos .pkl. Ejecuta entrenamiento.py primero.")
    exit()

# 2. Crear un dato de prueba (Un paciente ficticio)
# Usamos las mismas 8 variables que definimos
paciente_prueba = pd.DataFrame([{
    'hba1c': 8.5,                       # Alto (Diabetes probable)
    'glucose_postprandial': 200,        # Alto
    'glucose_fasting': 160,             # Alto
    'age': 50,
    'bmi': 32.0,                        # Obesidad
    'systolic_bp': 140,
    'cholesterol_total': 240,
    'physical_activity_minutes_per_week': 0
}])

print("\n--- Datos del Paciente de Prueba ---")
print(paciente_prueba)

# 3. Preprocesamiento (Escalar igual que en el entrenamiento)
try:
    paciente_scaled = scaler.transform(paciente_prueba)
    print("\n✅ Escalado exitoso.")
except Exception as e:
    print(f"\n❌ ERROR en escalado: {e}")
    exit()

# 4. Predicción
prediccion = model.predict(paciente_scaled)
probabilidad = model.predict_proba(paciente_scaled)[0][1]

print("\n--- Resultado del Modelo ---")
print(f"Clase Predicha: {prediccion[0]} (0=Sano, 1=Diabetes)")
print(f"Probabilidad de Diabetes: {probabilidad:.4f}")

if prediccion[0] == 1:
    print("CONCLUSIÓN: El modelo detecta RIESGO ALTO (Funciona OK)")
else:
    print("CONCLUSIÓN: El modelo detecta RIESGO BAJO (Funciona OK)")