import os
import tensorflow as tf
from tensorflow import keras
import warnings
import tensorflow_config
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Configuración de TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Actualizar las llamadas deprecadas
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suprimir advertencias específicas
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Si tu modelo usa sparse_softmax_cross_entropy, actualízalo así:
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE
)

# Si estás usando layer normalization, actualízalo así:
layer_norm = tf.keras.layers.LayerNormalization(
    axis=-1,
    epsilon=1e-12,
    dtype=tf.float32
)

# Configuración de GPU si está disponible
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- Cargar el modelo entrenado ---
modelo = load_model("Embedding_V5_Trained_1k_60x.h5")
# --- Cargar los datos y encoders ---
data = pd.read_csv("secuencia_ruleta_V2.csv")

with open('encoders.pickle', 'rb') as handle:
    encoders = pickle.load(handle)

color_encoder = encoders['color']
docena_encoder = encoders['docena']
fila_encoder = encoders['fila']
paridad_encoder = encoders['paridad']
alto_bajo_encoder = encoders['alto_bajo']

# Cargar datos de características de números
features_data = pd.read_csv("datos_ruleta.csv")

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

    predicciones = modelo.predict(secuencia_features)
    print(f"Shape of predicciones: {len(predicciones)}")

    return predicciones

def interpretar_prediccion(predicciones, encoders, secuencia=None):
    """Interpreta las predicciones del modelo."""
    # Original prediction logic
    color_pred = encoders['color'].categories_[0][np.argmax(predicciones[0][0])]
    prob_color = np.max(predicciones[0][0]) * 100

    docena_pred = encoders['docena'].categories_[0][np.argmax(predicciones[1][0])]
    prob_docena = np.max(predicciones[1][0]) * 100

    fila_pred = encoders['fila'].categories_[0][np.argmax(predicciones[2][0])]
    prob_fila = np.max(predicciones[2][0]) * 100

    paridad_pred = encoders['paridad'].categories_[0][np.argmax(predicciones[3][0])]
    prob_paridad = np.max(predicciones[3][0]) * 100

    alto_bajo_pred = encoders['alto_bajo'].categories_[0][np.argmax(predicciones[4][0])]
    prob_alto_bajo = np.max(predicciones[4][0]) * 100

    return {
        'sequence': secuencia,  # Add sequence to the return dictionary
        'color': (color_pred, prob_color),
        'docena': (docena_pred, prob_docena),
        'fila': (fila_pred, prob_fila),
        'paridad': (paridad_pred, prob_paridad),
        'alto_bajo': (alto_bajo_pred, prob_alto_bajo)
    }

def predecir_numero_final(secuencia_numeros, data, encoders, predicciones_caracteristicas):
    """
    Predice el número más probable basado en las predicciones de características
    y la secuencia histórica de números.
    """

    # Crear un diccionario para almacenar las probabilidades de cada número
    probabilidades_numeros = {num: 0 for num in range(37)}  # Incluir el 0

    # Ponderar las probabilidades basadas en las características predichas
    for num in range(37):
        caracteristicas_num = get_features_for_number(num, data, encoders)
        if caracteristicas_num is None:
            continue

        # Asignar pesos a cada característica
        peso_color = 0.23
        peso_docena = 0.16
        peso_fila = 0.1
        peso_paridad = 0.16
        peso_alto_bajo = 0.35

        # Calcular la probabilidad para el número actual
        if encoders['color'].categories_[0][np.argmax(caracteristicas_num[:3])] == predicciones_caracteristicas['color'][0]:
            probabilidades_numeros[num] += peso_color * predicciones_caracteristicas['color'][1] / 100
        if encoders['docena'].categories_[0][np.argmax(caracteristicas_num[3:6])] == predicciones_caracteristicas['docena'][0]:
            probabilidades_numeros[num] += peso_docena * predicciones_caracteristicas['docena'][1] / 100
        if encoders['fila'].categories_[0][np.argmax(caracteristicas_num[6:10])] == predicciones_caracteristicas['fila'][0]:
            probabilidades_numeros[num] += peso_fila * predicciones_caracteristicas['fila'][1] / 100
        if encoders['paridad'].categories_[0][np.argmax(caracteristicas_num[10:12])] == predicciones_caracteristicas['paridad'][0]:
            probabilidades_numeros[num] += peso_paridad * predicciones_caracteristicas['paridad'][1] / 100
        if encoders['alto_bajo'].categories_[0][np.argmax(caracteristicas_num[12:])] == predicciones_caracteristicas['alto_bajo'][0]:
            probabilidades_numeros[num] += peso_alto_bajo * predicciones_caracteristicas['alto_bajo'][1] / 100

    # Encontrar el número con la mayor probabilidad
    numero_predicho = max(probabilidades_numeros, key=probabilidades_numeros.get)
    probabilidad_predicha = probabilidades_numeros[numero_predicho] * 100

    return numero_predicho, probabilidad_predicha

def get_number_characteristics(numero):
    """Obtiene las características de un número de la ruleta desde datos_ruleta.csv."""
    try:
        row = features_data[features_data['Num'] == numero].iloc[0]
        return {
            'color': row['Color'],
            'docena': row['Docena'],
            'fila': row['Fila'],
            'paridad': row['Par_Non'],
            'alto_bajo': row['Mayor_Menor']
        }
    except Exception as e:
        print(f"Error accessing data for number {numero}: {e}")
        return None

def calculate_reward(prediction, actual):
    """Calcula la recompensa basada en la precisión de la predicción."""
    rewards = {
        'color': 0,
        'docena': 0,
        'fila': 0,
        'paridad': 0,
        'alto_bajo': 0
    }
    
    for feature in rewards.keys():
        if prediction[feature][0] == actual[feature]:
            rewards[feature] = 1
    
    return rewards

def retrain_model(model, experience_buffer, batch_size=100):
    """Reentrenar el modelo con las experiencias acumuladas."""
    if len(experience_buffer.buffer) < batch_size:
        return

    # Preparar datos de entrenamiento
    X_batch = []
    y_batch = {
        'color': [],
        'docena': [],
        'fila': [],
        'paridad': [],
        'alto_bajo': []
    }


    # Seleccionar batch aleatorio de experiencias
    experiences = np.random.choice(experience_buffer.buffer, batch_size)
    
    for exp in experiences:
        X_batch.append(exp['sequence'])
        for feature in y_batch.keys():
            y_batch[feature].append(exp['actual'][feature])

    # Convertir a arrays numpy
    X_batch = np.array(X_batch)
    for feature in y_batch.keys():
        y_batch[feature] = np.array(y_batch[feature])

    # Fine-tuning con un learning rate pequeño
    model.optimizer.learning_rate = 1e-5
    model.fit(
        X_batch,
        y_batch,
        epochs=1,
        batch_size=batch_size,
        verbose=0
    )

def validate_prediction(prediccion, actual_outcome, experience_buffer, model):
    """Validar predicción y actualizar el modelo."""
    # Calcular recompensas
    rewards = calculate_reward(prediccion, actual_outcome)
    
    # Agregar experiencia al buffer
    experience_buffer.add_experience(
        sequence=prediccion['sequence'],
        prediction=prediccion,
        actual_outcome=actual_outcome,
        reward=rewards
    )
    
    # Reentrenar si hay suficientes experiencias
    retrain_model(model, experience_buffer)
    
    return rewards

# Add ExperienceBuffer class definition before its first usage
class ExperienceBuffer:
    def __init__(self, max_size=1000):
        self.buffer = []
        self.max_size = max_size

    def add_experience(self, sequence, prediction, actual_outcome, reward):
        experience = {
            'sequence': sequence,
            'prediction': prediction,
            'actual': actual_outcome,
            'reward': reward
        }
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

# --- Ejemplo de uso ---
secuencia_numeros = [
    18, 25, 13, 9, 21, 4, 16, 24, 24, 26,
    18, 29, 14, 32, 6, 12, 14, 26, 11, 11,
    26, 26, 16, 4, 13, 16, 34, 31, 1, 1,
    28, 6, 24, 35, 5, 23, 26, 0, 24, 1,
    24, 29, 6, 23, 31, 15, 4, 21, 31, 12,
    15, 23, 16, 36, 32, 14, 2, 12, 10, 3
]  # Secuencia de ejemplo

# Create experience buffer before first use
experience_buffer = ExperienceBuffer()

# Predecir la siguiente tirada
predicciones_caracteristicas = predecir_siguiente_tirada(secuencia_numeros, data, encoders)

# Interpretar los resultados
resultado_caracteristicas = interpretar_prediccion(predicciones_caracteristicas, encoders, secuencia=secuencia_numeros)

print("\nPredicción de la siguiente tirada (características):")
print(f"Color: {resultado_caracteristicas['color'][0]} (Probabilidad: {resultado_caracteristicas['color'][1]:.2f}%)")
print(f"Docena: {resultado_caracteristicas['docena'][0]} (Probabilidad: {resultado_caracteristicas['docena'][1]:.2f}%)")
print(f"Fila: {resultado_caracteristicas['fila'][0]} (Probabilidad: {resultado_caracteristicas['fila'][1]:.2f}%)")
print(f"Paridad: {resultado_caracteristicas['paridad'][0]} (Probabilidad: {resultado_caracteristicas['paridad'][1]:.2f}%)")
print(f"Alto/Bajo: {resultado_caracteristicas['alto_bajo'][0]} (Probabilidad: {resultado_caracteristicas['alto_bajo'][1]:.2f}%)")

# Solicitar el número que salió
numero_real = int(input("\nIngrese el número que salió en la ruleta (0-36): "))
caracteristicas_reales = get_number_characteristics(numero_real)

# Validar predicción y obtener recompensas
rewards = validate_prediction(resultado_caracteristicas, caracteristicas_reales, experience_buffer, modelo)

print("\nNúmero que salió:", numero_real)
print("Características reales:", caracteristicas_reales)
print("Recompensas obtenidas:", rewards)

def train_on_historical_data(model, data, encoders, window_size=60):
    """Entrenar el modelo usando datos históricos."""
    experience_buffer = ExperienceBuffer()
    
    for i in range(len(data) - window_size - 1):
        # Obtener secuencia y resultado real
        sequence = data.iloc[i:i+window_size].index.tolist()
        actual_next = {
            'color': data.iloc[i+window_size]['Color'],
            'docena': data.iloc[i+window_size]['Docena'],
            'fila': data.iloc[i+window_size]['Fila'],
            'paridad': data.iloc[i+window_size]['Par_Non'],
            'alto_bajo': data.iloc[i+window_size]['Mayor_Menor']
        }
        
        # Hacer predicción
        predicciones = predecir_siguiente_tirada(sequence, data, encoders)
        resultado = interpretar_prediccion(predicciones, encoders, secuencia=sequence)
        
        # Validar y actualizar modelo
        validate_prediction(resultado, actual_next, experience_buffer, model)
        
        if i % 100 == 0:
            print(f"Procesado {i} secuencias históricas")

# Ejemplo de uso:
experience_buffer = ExperienceBuffer()

# Para entrenamiento histórico:
train_on_historical_data(modelo, data, encoders)

# Para validación en tiempo real:
predicciones = predecir_siguiente_tirada(secuencia_numeros, data, encoders)
resultado = interpretar_prediccion(predicciones, encoders, secuencia=secuencia_numeros)

# Mostrar predicciones
print("\nPredicción de la siguiente tirada (características):")
print(f"Color: {resultado['color'][0]} (Probabilidad: {resultado['color'][1]:.2f}%)")
print(f"Docena: {resultado['docena'][0]} (Probabilidad: {resultado['docena'][1]:.2f}%)")
print(f"Fila: {resultado['fila'][0]} (Probabilidad: {resultado['fila'][1]:.2f}%)")
print(f"Paridad: {resultado['paridad'][0]} (Probabilidad: {resultado['paridad'][1]:.2f}%)")
print(f"Alto/Bajo: {resultado['alto_bajo'][0]} (Probabilidad: {resultado['alto_bajo'][1]:.2f}%)")

# Solicitar el número que salió
numero_real = int(input("\nIngrese el número que salió en la ruleta (0-36): "))
caracteristicas_reales = get_number_characteristics(numero_real)

# Validar predicción y obtener recompensas
rewards = validate_prediction(resultado, caracteristicas_reales, experience_buffer, modelo)

print("\nNúmero que salió:", numero_real)
print("Características reales:", caracteristicas_reales)
print("Recompensas obtenidas:", rewards)