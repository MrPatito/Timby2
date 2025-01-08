import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model

# --- Cargar el modelo entrenado ---
modelo = load_model("modelo_ruleta_B70.h5")

# --- Cargar las características de los números ---
data = pd.read_csv("datos_ruleta.csv")

# --- Preprocesamiento de los datos ---
encoder = OneHotEncoder(handle_unknown='ignore')
features = encoder.fit_transform(data[['Par_Non', 'Color', 'Mayor_Menor', 'Docena', 'Fila']]).toarray()

# --- Función para predecir el siguiente número ---
def predecir_siguiente_numero(secuencia_numeros):
    """
    Predice el siguiente número de la ruleta basándose en una secuencia de números previos.

    Args:
      secuencia_numeros: Una lista de números previos de la ruleta.

    Returns:
      Una lista con las características predichas del siguiente número.
    """

    # Convertir la secuencia de números a características
    secuencia_features = []
    for numero in secuencia_numeros:
        caracteristicas_numero = features[numero]
        secuencia_features.append(caracteristicas_numero)
    secuencia_features = np.array([secuencia_features])

    # Hacer la predicción
    prediccion = modelo.predict(secuencia_features)

    # Decodificar la predicción
    caracteristicas_predichas = encoder.inverse_transform(prediccion)

    return caracteristicas_predichas[0]

# --- Ejemplo de uso ---

# Secuencia de números previos (ajusta esto con tus datos)
secuencia_numeros = [6, 19, 30, 4, 16, 4, 20, 5, 0, 15, 0, 36, 26, 9, 15, 18, 30, 0, 26, 5]

# Predecir el siguiente número
prediccion = predecir_siguiente_numero(secuencia_numeros)

# Imprimir la predicción
print("Predicción del siguiente número:")
print(prediccion)