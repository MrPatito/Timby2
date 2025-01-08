import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# --- Cargar el modelo entrenado ---
try:
    modelo = load_model("best_model_1k_60x.h5")  # Cambia el nombre si tu modelo tiene otro
except OSError as e:
    print(f"Error loading model: {e}")
    modelo = None  # Ensure modelo is defined as None if loading fails

# --- Cargar los datos y encoders ---
data = pd.read_csv("datos_ruleta.csv")

# Cargar los encoders desde el archivo
with open('encoders.pickle', 'rb') as handle:
    encoders = pickle.load(handle)

color_encoder = encoders['color']
docena_encoder = encoders['docena']
fila_encoder = encoders['fila']
paridad_encoder = encoders['paridad']
alto_bajo_encoder = encoders['alto_bajo']

def get_features_for_number(num, data, encoders):
    """Obtiene las características codificadas para un número dado."""
    try:
        color = data.iloc[num]['Color']
        docena = data.iloc[num]['Docena']
        fila = data.iloc[num]['Fila']
        paridad = data.iloc[num]['Par_Non']
        alto_bajo = data.iloc[num]['Mayor_Menor']

        color_feat = encoders['color'].transform([[color]])
        docena_feat = encoders['docena'].transform([[docena]])
        fila_feat = encoders['fila'].transform([[fila]])
        paridad_feat = encoders['paridad'].transform([[paridad]])
        alto_bajo_feat = encoders['alto_bajo'].transform([[alto_bajo]])

        return np.concatenate([color_feat[0], docena_feat[0], fila_feat[0], paridad_feat[0], alto_bajo_feat[0]])
    except Exception as e:
        print(f"Error extracting features for number {num}: {e}")
        return None

def predecir_siguiente_tirada(secuencia_numeros, data, encoders):
    """Predice la siguiente tirada basada en una secuencia de números."""
    secuencia_features = []
    for numero in secuencia_numeros:
        features = get_features_for_number(numero, data, encoders)
        if features is not None:
            secuencia_features.append(features)

    if not secuencia_features:
        raise ValueError("Feature extraction failed for all numbers.")

    secuencia_features = np.array([secuencia_features])
    print(f"Shape of secuencia_features: {secuencia_features.shape}")

    # Hacer la predicción
    predicciones = modelo.predict(secuencia_features)
    print(f"Shape of predicciones: {len(predicciones)}")  # Debería ser 5 ahora (color, docena, fila, paridad, alto_bajo)

    return predicciones

def interpretar_prediccion(predicciones, encoders):
    """Interpreta las predicciones del modelo."""
    # Predicciones de color
    color_pred = encoders['color'].categories_[0][np.argmax(predicciones[0][0])]
    prob_color = np.max(predicciones[0][0]) * 100

    # Predicciones de docena
    docena_pred = encoders['docena'].categories_[0][np.argmax(predicciones[1][0])]
    prob_docena = np.max(predicciones[1][0]) * 100
    
    # Predicciones de fila
    fila_pred = encoders['fila'].categories_[0][np.argmax(predicciones[2][0])]
    prob_fila = np.max(predicciones[2][0]) * 100

    # Predicciones de paridad
    paridad_pred = encoders['paridad'].categories_[0][np.argmax(predicciones[3][0])]
    prob_paridad = np.max(predicciones[3][0]) * 100

    # Predicciones de alto/bajo
    alto_bajo_pred = encoders['alto_bajo'].categories_[0][np.argmax(predicciones[4][0])]
    prob_alto_bajo = np.max(predicciones[4][0]) * 100

    return {
        'color': (color_pred, prob_color),
        'docena': (docena_pred, prob_docena),
        'fila': (fila_pred, prob_fila),
        'paridad': (paridad_pred, prob_paridad),
        'alto_bajo': (alto_bajo_pred, prob_alto_bajo)
    }

# --- Ejemplo de uso ---
secuencia_numeros = [
    18, 25, 13, 9, 21, 4, 16, 24, 24, 26,
    18, 29, 14, 32, 6, 12, 14, 26, 11, 11,
    26, 26, 16, 4, 13, 16, 34, 31, 1, 1,
    28, 6, 24, 35, 5, 23, 26, 0, 24, 1,
    24, 29, 6, 23, 31, 15, 4, 21, 31, 12,
    15, 23, 16, 36, 32, 14, 2, 12, 10, 3]  # Secuencia de ejemplo

# Predecir la siguiente tirada
if modelo is not None:  # Check if the model was loaded successfully
    prediccion = predecir_siguiente_tirada(secuencia_numeros, data, encoders)
else:
    print("Model not loaded. Cannot make predictions.")

# Interpretar los resultados
resultado = interpretar_prediccion(prediccion, encoders)

print("\nPredicción de la siguiente tirada:")
print(f"Color: {resultado['color'][0]} (Probabilidad: {resultado['color'][1]:.2f}%)")
print(f"Docena: {resultado['docena'][0]} (Probabilidad: {resultado['docena'][1]:.2f}%)")
print(f"Fila: {resultado['fila'][0]} (Probabilidad: {resultado['fila'][1]:.2f}%)")
print(f"Paridad: {resultado['paridad'][0]} (Probabilidad: {resultado['paridad'][1]:.2f}%)")
print(f"Alto/Bajo: {resultado['alto_bajo'][0]} (Probabilidad: {resultado['alto_bajo'][1]:.2f}%)")