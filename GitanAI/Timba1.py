import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization,
    LayerNormalization, MultiHeadAttention, Add, Concatenate
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# --- Preparación de los datos ---

# 1. Cargar los datos de la secuencia de la ruleta
secuencia_data = pd.read_csv("secuencia_ruleta.csv")

# 2. Cargar las características de los números
data = pd.read_csv("datos_ruleta.csv")

# 3. Preprocesamiento de los datos
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

# Function to get combined features for a number
def get_features_for_number(num):
    numero_feat = numero_encoder.transform([[num]])
    color = data.iloc[num]['Color']
    docena = data.iloc[num]['Docena']
    fila = data.iloc[num]['Fila']
    
    color_feat = color_encoder.transform([[color]])
    docena_feat = docena_encoder.transform([[docena]])
    fila_feat = fila_encoder.transform([[fila]])
    
    return np.concatenate([color_feat[0], docena_feat[0], fila_feat[0], numero_feat[0]])

# 4. Crear secuencias de características
secuencia_features = []
for numero in secuencia_data['Num']:
    caracteristicas_numero = get_features_for_number(numero)
    secuencia_features.append(caracteristicas_numero)

secuencia_features = np.array(secuencia_features)

# 5. Definir la longitud de la secuencia
secuencia_longitud = 20

# 6. Crear secuencias de entrada (X) y salida (y)
X, y = [], []
for i in range(len(secuencia_features) - secuencia_longitud - 1):
    X.append(secuencia_features[i:(i + secuencia_longitud)])
    y.append(secuencia_features[i + secuencia_longitud])
X = np.array(X)
y = np.array(y)

# 7. Preparar las etiquetas para cada categoría
y_color = y[:, :3]  # Primeros 3 valores son color
y_docena = y[:, 3:15]  # Siguientes 12 valores son docena
y_fila = y[:, 15:18]  # Siguientes 3 valores son fila
y_numero = y[:, 18:]  # Restantes 37 valores son número

# --- Parámetros de la arquitectura ---
LSTM_UNITS = [256, 128, 64]
DENSE_UNITS = [512, 256]
ATTENTION_HEADS = 8
DROPOUT_RATE = 0.3
L1_FACTOR = 1e-5
L2_FACTOR = 1e-4

# --- Construir el modelo ---
# Definir las entradas
inputs = Input(shape=(X.shape[1], X.shape[2]))
x = inputs

# Primera capa de normalización
x = LayerNormalization()(x)

# Bloque inicial de LSTM Bidireccional
lstm_out = Bidirectional(
    LSTM(LSTM_UNITS[0], 
         return_sequences=True,
         kernel_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR),
         recurrent_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR))
)(x)

# Mecanismo de atención multi-cabeza
attention_output = MultiHeadAttention(
    num_heads=ATTENTION_HEADS,
    key_dim=LSTM_UNITS[0]
)(lstm_out, lstm_out)

# Conexión residual post-atención
x = Add()([lstm_out, attention_output])
x = LayerNormalization()(x)

# Bloques LSTM profundos con conexiones residuales
for units in LSTM_UNITS[1:]:
    lstm_out = Bidirectional(
        LSTM(units, 
             return_sequences=True,
             kernel_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR),
             recurrent_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR))
    )(x)
    x = Add()([x, lstm_out])
    x = LayerNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

# Última capa LSTM para secuencia final
x = Bidirectional(
    LSTM(LSTM_UNITS[-1],
         kernel_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR),
         recurrent_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR))
)(x)

# Capas densas compartidas
for units in DENSE_UNITS:
    x = Dense(
        units,
        activation='relu',
        kernel_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR)
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(DROPOUT_RATE)(x)

# Salidas específicas para cada predicción
dense_color = Dense(
    3, 
    activation='softmax',
    name='color',
    kernel_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR)
)(x)

dense_docena = Dense(
    12,
    activation='softmax',
    name='docena',
    kernel_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR)
)(x)

dense_fila = Dense(
    3,
    activation='softmax',
    name='fila',
    kernel_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR)
)(x)

dense_numero = Dense(
    37,
    activation='softmax',
    name='numero',
    kernel_regularizer=l1_l2(l1=L1_FACTOR, l2=L2_FACTOR)
)(x)

# Crear el modelo
modelo = Model(inputs=inputs, outputs=[dense_color, dense_docena, dense_fila, dense_numero])

# Modify compilation with gradient clipping and adjusted learning rate
optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
modelo.compile(
    optimizer=optimizer,
    loss={
        'color': 'categorical_crossentropy',
        'docena': 'categorical_crossentropy',
        'fila': 'categorical_crossentropy',
        'numero': 'categorical_crossentropy'
    },
    metrics=['accuracy'],
    loss_weights={
        'color': 1.0,
        'docena': 1.0,
        'fila': 1.0,
        'numero': 1.0  # Reduced from 2.0 to balance learning
    }
)

# Add model summary for debugging
modelo.summary()

# Modify training parameters
history = modelo.fit(
    X,
    {
        'color': y_color,
        'docena': y_docena,
        'fila': y_fila,
        'numero': y_numero
    },
    epochs=1000,  # Reduced from 5000
    batch_size=64,  # Increased from 32
    validation_split=0.2,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Reduced from 50
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Changed from 0.2
            patience=10,  # Reduced from 20
            min_lr=1e-6
        ),
        ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
    ]
)

# Add basic training visualization
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Guardar el modelo
modelo.save("modelo_ruleta_CMulti_Cursor.h5")