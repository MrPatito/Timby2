import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention

# --- Cargar los datos ---
data = pd.read_csv("datos_ruleta.csv")  # Asegúrate de que este archivo contiene los datos combinados

# --- Preprocesamiento de los datos ---

# 1. Codificar las características categóricas
encoder = OneHotEncoder(handle_unknown='ignore')
categorical_features = encoder.fit_transform(data[['Num', 'Par_Non', 'Color', 'Mayor_Menor', 'Docena', 'Fila']]).toarray()  # Ajustado para incluir todas las columnas

# 2. Escalar las características numéricas (si hay alguna)
# ... (Si tienes otras características numéricas además de 'Num', agrega el código aquí) ...

# 3. Combinar las características
features = categorical_features  # Si solo tienes características categóricas

# 4. Crear secuencias de datos
secuencia_longitud = 20
X, y = [], []
for i in range(len(features) - secuencia_longitud - 1):
    X.append(features[i:(i + secuencia_longitud)])
    y.append(features[i + secuencia_longitud])
X = np.array(X)
y = np.array(y)

# 5. Dividir los datos en conjuntos de entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- Construir la red neuronal ---
model = Sequential()
model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Bidirectional(LSTM(256, return_sequences=True)))  # Mantener return_sequences=True
model.add(Attention())  # Mover la capa de atención aquí
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))  # Ajustar num_outputs según tus características de salida

# --- Compilar el modelo ---
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# --- Entrenar el modelo ---
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# --- Guardar el modelo ---
model.save("modelo_ruleta_V2.h5")

# --- Función para predecir el siguiente número ---
def predecir_siguiente_numero(secuencia_numeros):
    """
    Predice el siguiente número de la ruleta basándose en una secuencia de números previos.

    Args:
      secuencia_numeros: Una lista de números previos de la ruleta.

    Returns:
      Una lista con las características predichas del siguiente número.
    """
    secuencia_features = []
    for numero in secuencia_numeros:
        # Obtener las características del número (debes adaptar esto a tu estructura de datos)
        # ...
        secuencia_features.append(caracteristicas_numero)
    secuencia_features = np.array([secuencia_features])

    # Hacer la predicción
    prediccion = model.predict(secuencia_features)

    # Decodificar la predicción
    # ...

    return prediccion

# --- Ejemplo de uso ---
# secuencia_numeros = [1, 5, 12, 24, 31, 8, 17, 22, 35, 10]
# prediccion = predecir_siguiente_numero(secuencia_numeros)
# print("Predicción del siguiente número:")
# print(prediccion)