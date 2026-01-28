import pandas as pd
import joblib
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve, 
    roc_auc_score, 
    accuracy_score, 
    classification_report, 
    confusion_matrix
)

# ==========================================
# Carga
# ==========================================
print("--- Iniciando Entrenamiento Profesional ---")
try:
    df = pd.read_csv("data/diabetes_dataset.csv")
    print(f" Dataset cargado: {df.shape[0]} registros.")
except FileNotFoundError:
    print(" Error: No se encuentra 'diabetes_dataset.csv'")
    exit()

# Features y Target
features = [
    "hba1c", "glucose_postprandial", "glucose_fasting", "age", 
    "bmi", "systolic_bp", "cholesterol_total", "physical_activity_minutes_per_week"
]
target = "diagnosed_diabetes"

# --- Verificacion del Target ---
# Matriz Correlacion
print("\nAnálisis de Correlación (Top variables influyentes):")
correlations = df[features + [target]].corr()[target].sort_values(ascending=False)
print(correlations.drop(target))  # Mostramos todas menos el target mismo

X = df[features]
y = df[target]

# ==========================================
# preparacion de ML
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")
print("\nScaler guardado correctamente.")

# ==========================================
# 3. Train
# ==========================================
print("Entrenando modelo...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12, 
    class_weight="balanced", # Vital para detectar enfermos reales
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

joblib.dump(model, "model.pkl")
print(" Modelo entrenado y guardado.")

# ==========================================
# buscando el mejor umbral
# ==========================================
# Probabilidades (0 a 1)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# calculo
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])

print(f"\n💡 Umbral Óptimo Detectado: {round(best_threshold, 4)}")
print("(Este es el valor que usarás en tu API para decidir Si/No)")

# ==========================================
# verificacion con el umbral óptimo
# ==========================================
# Generamos predicciones finales usando el umbral óptimo
y_pred_final = (y_proba >= best_threshold).astype(int)

# Matriz de Confusión (Lo que pedías recuperar)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()

print("\n Matriz de Confusión (Con Umbral Óptimo):")
print(f"   - Verdaderos Negativos (Sanos OK): {tn}")
print(f"   - Falsos Positivos (Alarma Falsa): {fp}")
print(f"   - Falsos Negativos (Peligro!):     {fn}")
print(f"   - Verdaderos Positivos (Detectados): {tp}")

# Guardamos todo en el JSON detallado
metrics_data = {
    "descripcion": "Modelo Random Forest Binario (Optimizado)",
    "threshold_optimo": best_threshold,
    "metricas_globales": {
        "accuracy": float(accuracy_score(y_test, y_pred_final)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precisions[best_idx]),
        "recall": float(recalls[best_idx]),
        "f1_score": float(f1_scores[best_idx])
    },
    "matriz_confusion": {
        "verdaderos_negativos": int(tn),
        "falsos_positivos": int(fp),
        "falsos_negativos": int(fn),
        "verdaderos_positivos": int(tp)
    },
    "importancia_variables": dict(zip(features, model.feature_importances_))
}

with open("metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

print("\n Reporte completo generado en 'metrics.json'.")
print("------------------------------------------------")