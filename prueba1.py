import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model

# --- Cargar el modelo entrenado ---
modelo = load_model("modelo_ruleta_V2.h5")

# --- Cargar las características de los números ---
data = pd.read_csv("datos_ruleta.csv")

# Create separate encoders for each category
color_encoder = OneHotEncoder(sparse_output=False)
docena_encoder = OneHotEncoder(sparse_output=False)
fila_encoder = OneHotEncoder(sparse_output=False)
numero_encoder = OneHotEncoder(sparse_output=False)

# Prepare the data for each category
color_features = color_encoder.fit_transform(data[['Color']].values)
docena_features = docena_encoder.fit_transform(data[['Docena']].values)
fila_features = fila_encoder.fit_transform(data[['Fila']].values)
numero_features = numero_encoder.fit_transform([[i] for i in range(37)])

def get_features_for_number(num):
    numero_feat = numero_encoder.transform([[num]])
    color = data.iloc[num]['Color']
    docena = data.iloc[num]['Docena']
    fila = data.iloc[num]['Fila']
    
    color_feat = color_encoder.transform([[color]])
    docena_feat = docena_encoder.transform([[docena]])
    fila_feat = fila_encoder.transform([[fila]])
    
    # Concatenate all features
    return np.concatenate([color_feat[0], docena_feat[0], fila_feat[0], numero_feat[0]])

def predecir_siguiente_numero(secuencia_numeros):
    # Convertir la secuencia de números a características
    secuencia_features = []
    for numero in secuencia_numeros:
        caracteristicas_numero = get_features_for_number(numero)
        secuencia_features.append(caracteristicas_numero)
    secuencia_features = np.array([secuencia_features])
    
    # Hacer la predicción
    predicciones = modelo.predict(secuencia_features)
    
    # Las predicciones ahora son [color, docena, fila, numero]
    return predicciones

def interpretar_prediccion(predicciones):
    # Mapear colores
    colores = ['Rojo', 'Negro', 'Verde']
    color_pred = colores[np.argmax(predicciones[0][0])]
    prob_color = np.max(predicciones[0][0]) * 100
    
    # Mapear docena - CORREGIDO
    # Agrupar las probabilidades en 3 docenas
    docena_probs = predicciones[1][0]
    docena_1 = np.sum(docena_probs[0:4])    # Primera docena
    docena_2 = np.sum(docena_probs[4:8])    # Segunda docena
    docena_3 = np.sum(docena_probs[8:12])   # Tercera docena
    docenas_agrupadas = [docena_1, docena_2, docena_3]
    docena_pred = np.argmax(docenas_agrupadas) + 1
    prob_docena = np.max(docenas_agrupadas) * 100
    
    # Mapear fila
    filas = ['Primera', 'Segunda', 'Tercera']
    fila_pred = filas[np.argmax(predicciones[2][0])]
    prob_fila = np.max(predicciones[2][0]) * 100
    
    # Mapear número
    numero_pred = np.argmax(predicciones[3][0])
    prob_numero = np.max(predicciones[3][0]) * 100
    
    return {
        'numero': (numero_pred, prob_numero),
        'color': (color_pred, prob_color),
        'docena': (docena_pred, prob_docena),
        'fila': (fila_pred, prob_fila)
    }

# --- Ejemplo de uso ---

# Secuencia de números previos (ajusta esto con tus datos)
secuencia_numeros = [6, 19, 30, 4, 16, 4, 20, 5, 0, 15, 0, 36, 26, 9, 15, 18, 30, 0, 26, 5]

# Predecir el siguiente número
prediccion = predecir_siguiente_numero(secuencia_numeros)
resultado = interpretar_prediccion(prediccion)

print("\nPredicción del siguiente número:")
print(f"Número: {resultado['numero'][0]} (Probabilidad: {resultado['numero'][1]:.2f}%)")
print(f"Color: {resultado['color'][0]} (Probabilidad: {resultado['color'][1]:.2f}%)")
print(f"Docena: {resultado['docena'][0]} (Probabilidad: {resultado['docena'][1]:.2f}%)")
print(f"Fila: {resultado['fila'][0]} (Probabilidad: {resultado['fila'][1]:.2f}%)")