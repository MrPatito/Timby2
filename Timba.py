import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dropout

# --- Preparación de los datos ---

# 1. Cargar los datos de la secuencia de la ruleta
secuencia_data = pd.read_csv("secuencia_ruleta.csv")

# 2. Cargar las características de los números
data = pd.read_csv("datos_ruleta.csv")

# 3. Preprocesamiento de los datos
# Convertir las características categóricas a numéricas usando OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')

# Ajustar el encoder a las columnas categóricas incluyendo "Docena" y "Fila"
features = encoder.fit_transform(data[['Par_Non', 'Color', 'Mayor_Menor', 'Docena', 'Fila']]).toarray()  # Cambio aquí

# 4. Combinar la secuencia de números con sus características
# Crear una lista para almacenar las características de la secuencia
secuencia_features = []

# Before the loop, let's add a debug line to see available columns
print("Available columns:", secuencia_data.columns)

# Then modify the loop to use the correct column name
# It might be 'Num' instead of 'Numero'
for numero in secuencia_data['Num']:  # Changed back to 'Num'
    caracteristicas_numero = features[numero]
    secuencia_features.append(caracteristicas_numero)

# Convertir la lista a un array NumPy
secuencia_features = np.array(secuencia_features)

# 5. Definir la longitud de la secuencia
secuencia_longitud = 20  # Puedes ajustar este valor

# 6. Crear secuencias de entrada (X) y salida (y)
X, y = [], []
for i in range(len(secuencia_features) - secuencia_longitud - 1):
    X.append(secuencia_features[i:(i + secuencia_longitud)])
    y.append(secuencia_features[i + secuencia_longitud])
X = np.array(X)
y = np.array(y)

# --- Construir la red neuronal ---
# 1. Definir el modelo pro
modelo = Sequential()
modelo.add(LSTM(256, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
modelo.add(LSTM(256))
modelo.add(Dropout(0.2))
modelo.add(Dense(128, activation='relu'))
modelo.add(Dense(y.shape[1]))
modelo.compile(loss='mse', optimizer='adam')

# 1. Definir el modelo
##
#modelo = Sequential()
#modelo.add(LSTM(256, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
#modelo.add(LSTM(256))  # Segunda capa LSTM
#modelo.add(Dense(128, activation='relu'))  # Capa densa con 128 neuronas y activación ReLU

modelo.add(Dropout(0.2))  # Dropout con una tasa del 20%

# 2. Compilar el modelo
modelo.compile(loss='mse', optimizer='adam')

# --- Entrenar la red neuronal ---

# 1. Entrenar el modelo
print("Iniciando entrenamiento...")
modelo.fit(X, y, epochs=2000, batch_size=70)
print("Entrenamiento finalizado.")
# --- Guardar el modelo ---

# 1. Guardar el modelo entrenado
modelo.save("modelo_ruleta_B70.h5")