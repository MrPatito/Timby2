import os
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization,
    LayerNormalization, MultiHeadAttention, Add, Embedding, Concatenate
)
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pickle

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

# --- Funciones Auxiliares ---
def load_and_train_model(data_path="datos_ruleta.csv"):
    """Carga los datos de la ruleta y entrena el modelo."""
    try:
        data = pd.read_csv(data_path)
        print("Datos de la ruleta cargados correctamente.")
        
        # Preparar datos para entrenamiento
        X = data.drop(['Num'], axis=1)
        y = data['Num']
        
        # Entrenar modelo
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_test)
        print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.2f}")
        
        return model
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return None


def load_data(unified_data_path="secuencia_ruleta_V2.csv"):
    """Carga los datos unificados de la ruleta."""
    try:
        data = pd.read_csv(unified_data_path)
        print("Datos unificados cargados correctamente.")
        return data
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return None

def create_encoders(data):
    """Crea y entrena los codificadores OneHotEncoder."""
    encoders = {}
    try:
        encoders['color'] = OneHotEncoder(sparse_output=False).fit(data[['Color']].values)
        encoders['docena'] = OneHotEncoder(sparse_output=False).fit(data[['Docena']].values)
        encoders['fila'] = OneHotEncoder(sparse_output=False).fit(data[['Fila']].values)
        encoders['paridad'] = OneHotEncoder(sparse_output=False).fit(data[['Par_Non']].values)
        encoders['alto_bajo'] = OneHotEncoder(sparse_output=False).fit(data[['Mayor_Menor']].values)
        print("Codificadores creados correctamente.")
        return encoders
    except KeyError as e:
        print(f"Error processing columns: {e}")
        return None

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

def prepare_sequences(data, encoders, secuencia_longitud=60):
    """Prepara las secuencias de entrada y salida desde datos unificados."""
    # Codificar todas las características
    features = np.hstack([
        encoders['color'].transform(data[['Color']]),
        encoders['docena'].transform(data[['Docena']]),
        encoders['fila'].transform(data[['Fila']]),
        encoders['paridad'].transform(data[['Par_Non']]),
        encoders['alto_bajo'].transform(data[['Mayor_Menor']])
    ])
    
    X, y = [], []
    for i in range(len(features) - secuencia_longitud):
        X.append(features[i:(i + secuencia_longitud)])
        y.append(features[i + secuencia_longitud])

    if not X or not y:
        raise ValueError("Insufficient data to create training sequences.")

    X = np.array(X)
    y = np.array(y)
    print(f"Forma de X: {X.shape}")
    print(f"Forma de y: {y.shape}")
    return X, y

def split_output(y, encoders):
    """Divide la salida en sus componentes (color, docena, fila, paridad, alto_bajo)."""
    num_color_features = encoders['color'].categories_[0].shape[0]
    num_docena_features = encoders['docena'].categories_[0].shape[0]
    num_fila_features = encoders['fila'].categories_[0].shape[0]
    num_paridad_features = encoders['paridad'].categories_[0].shape[0]
    num_alto_bajo_features = encoders['alto_bajo'].categories_[0].shape[0]

    y_color = y[:, :num_color_features]
    y_docena = y[:, num_color_features:num_color_features + num_docena_features]
    y_fila = y[:, num_color_features + num_docena_features:num_color_features + num_docena_features + num_fila_features]
    y_paridad = y[:, num_color_features + num_docena_features + num_fila_features:num_color_features + num_docena_features + num_fila_features + num_paridad_features]
    y_alto_bajo = y[:, num_color_features + num_docena_features + num_fila_features + num_paridad_features:num_color_features + num_docena_features + num_fila_features + num_paridad_features + num_alto_bajo_features]

    print(f"Forma de y_color: {y_color.shape}")
    print(f"Forma de y_docena: {y_docena.shape}")
    print(f"Forma de y_fila: {y_fila.shape}")
    print(f"Forma de y_paridad: {y_paridad.shape}")
    print(f"Forma de y_alto_bajo: {y_alto_bajo.shape}")

    return y_color, y_docena, y_fila, y_paridad, y_alto_bajo

def build_model(input_shape, num_color_features, num_docena_features, num_fila_features, num_paridad_features, num_alto_bajo_features,
                lstm_units=[128, 64], dense_units=[256, 128], attention_heads=4, dropout_rate=0.7, l1_factor=1e-1, l2_factor=1e-1):
    """Construye el modelo de red neuronal.
    Se han ajustado los hiperparámetros para un conjunto de datos pequeño:
    - Reducción de unidades LSTM y neuronas en capas densas.
    - Reducción del número de cabezas de atención.
    - Aumento de la tasa de dropout.
    - Aumento de la regularización L1 y L2.
    """
    inputs = Input(shape=input_shape)
    print(f"Shape of inputs: {inputs.shape}")
    x = LayerNormalization()(inputs)
    print(f"Shape of x after LayerNormalization: {x.shape}")

    # Primera capa LSTM bidireccional
    lstm_out = Bidirectional(
        LSTM(lstm_units[0],
             return_sequences=True,
             kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor),
             recurrent_regularizer=l1_l2(l1=l1_factor, l2=l2_factor))
    )(x)
    print(f"Shape of lstm_out after first Bidirectional LSTM: {lstm_out.shape}")

    # Capa de atención
    attention_output = MultiHeadAttention(
        num_heads=attention_heads,
        key_dim=lstm_units[0]
    )(lstm_out, lstm_out)
    print(f"Shape of attention_output after MultiHeadAttention: {attention_output.shape}")

    # Capa densa después de la atención
    x = Dense(lstm_units[1] * 2)(attention_output)
    x = LayerNormalization()(x)
    print(f"Shape of x after Dense and LayerNormalization: {x.shape}")

    # Capas LSTM bidireccionales adicionales
    for units in lstm_units[1:]:
        lstm_out = Bidirectional(
            LSTM(units,
                 return_sequences= (units != lstm_units[-1]),
                 kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor),
                 recurrent_regularizer=l1_l2(l1=l1_factor, l2=l2_factor))
        )(x)
        print(f"Shape of lstm_out after additional Bidirectional LSTM with units {units}: {lstm_out.shape}")
        if units != lstm_units[-1]:
            x = Add()([x, lstm_out])
            print(f"Shape of x after Add: {x.shape}")
            x = LayerNormalization()(x)
            print(f"Shape of x after LayerNormalization: {x.shape}")
            x = Dropout(dropout_rate)(x)
            print(f"Shape of x after Dropout: {x.shape}")
        else:
            x = lstm_out

    # Capas densas
    for units in dense_units:
        x = Dense(
            units,
            activation='relu',
            kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)
        )(x)
        print(f"Shape of x after Dense with units {units}: {x.shape}")
        x = BatchNormalization()(x)
        print(f"Shape of x after BatchNormalization: {x.shape}")
        x = Dropout(dropout_rate)(x)
        print(f"Shape of x after Dropout: {x.shape}")

    # Capas de salida
    dense_color = Dense(
        num_color_features,
        activation='softmax',
        name='color',
        kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)
    )(x)
    print(f"Shape of dense_color: {dense_color.shape}")

    dense_docena = Dense(
        num_docena_features,
        activation='softmax',
        name='docena',
        kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)
    )(x)
    print(f"Shape of dense_docena: {dense_docena.shape}")

    dense_fila = Dense(
        num_fila_features,
        activation='softmax',
        name='fila',
        kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)
    )(x)
    print(f"Shape of dense_fila: {dense_fila.shape}")

    dense_paridad = Dense(
        num_paridad_features,
        activation='softmax',
        name='paridad',
        kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)
    )(x)
    print(f"Shape of dense_paridad: {dense_paridad.shape}")

    dense_alto_bajo = Dense(
        num_alto_bajo_features,
        activation='softmax',
        name='alto_bajo',
        kernel_regularizer=l1_l2(l1=l1_factor, l2=l2_factor)
    )(x)
    print(f"Shape of dense_alto_bajo: {dense_alto_bajo.shape}")

    modelo = Model(inputs=inputs, outputs=[dense_color, dense_docena, dense_fila, dense_paridad, dense_alto_bajo])
    print("Modelo creado correctamente.")
    return modelo

def compile_model(model, learning_rate=1e-9):
    """Compila el modelo."""
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss={
            'color': 'categorical_crossentropy',
            'docena': 'categorical_crossentropy',
            'fila': 'categorical_crossentropy',
            'paridad': 'categorical_crossentropy',
            'alto_bajo': 'categorical_crossentropy'
        },
        metrics=['accuracy'],
        loss_weights={
            'color': 1.0,
            'docena': 1.0,
            'fila': 1.0,
            'paridad': 1.0,
            'alto_bajo': 1.0
        }
    )
    print("Modelo compilado correctamente.")

def train_model(model, X_train, y_train, X_val, y_val, epochs=1000, batch_size=60):
    """Entrena el modelo."""
    try:
        # Cambiar la ruta del archivo para usar una ruta absoluta y nombre simple
        checkpoint_path = os.path.join(os.getcwd(), 'best_model_1k_60x.h5')
        
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=900,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,
                    patience=600,
                    min_lr=1e-5
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    checkpoint_path,  # Usar la nueva ruta
                    monitor='val_loss',
                    save_best_only=True
                )
            ],
            verbose=1  # Agregar verbose para ver el progreso
        )
        print("Modelo entrenado correctamente.")
        return history
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        return None

# --- Flujo Principal ---

if __name__ == "__main__":
    # Cargar datos unificados
    data = load_data()

    if data is not None:
        encoders = create_encoders(data)

        if encoders is not None:
            # Guardar los encoders
            with open('encoders.pickle', 'wb') as handle:
                pickle.dump(encoders, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Encoders guardados correctamente.")

            # Preparar secuencias directamente desde los datos unificados
            X, y = prepare_sequences(data, encoders)
            y_color, y_docena, y_fila, y_paridad, y_alto_bajo = split_output(y, encoders)

            # Dividir en conjuntos de entrenamiento y validación
            X_train, X_val, y_color_train, y_color_val, y_docena_train, y_docena_val, y_fila_train, y_fila_val, y_paridad_train, y_paridad_val, y_alto_bajo_train, y_alto_bajo_val = train_test_split(
                X, y_color, y_docena, y_fila, y_paridad, y_alto_bajo, test_size=0.2, random_state=42
            )

            y_train = {
                'color': y_color_train,
                'docena': y_docena_train,
                'fila': y_fila_train,
                'paridad': y_paridad_train,
                'alto_bajo': y_alto_bajo_train
            }

            y_val = {
                'color': y_color_val,
                'docena': y_docena_val,
                'fila': y_fila_val,
                'paridad': y_paridad_val,
                'alto_bajo': y_alto_bajo_val
            }

            # ... (resto del código) ...

            num_color_features = encoders['color'].categories_[0].shape[0]
            num_docena_features = encoders['docena'].categories_[0].shape[0]
            num_fila_features = encoders['fila'].categories_[0].shape[0]
            num_paridad_features = encoders['paridad'].categories_[0].shape[0]
            num_alto_bajo_features = encoders['alto_bajo'].categories_[0].shape[0]

            model = build_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_color_features=num_color_features,
                num_docena_features=num_docena_features,
                num_fila_features=num_fila_features,
                num_paridad_features=num_paridad_features,
                num_alto_bajo_features=num_alto_bajo_features
            )
            model.summary()
            compile_model(model, learning_rate=1e-4)

            # Imprimir información de formas y tipos de datos
            print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
            print(f"y_train type: {type(y_train)}")
            print(f"y_train['color'] shape: {y_train['color'].shape}, dtype: {y_train['color'].dtype}")
            print(f"y_train['docena'] shape: {y_train['docena'].shape}, dtype: {y_train['docena'].dtype}")
            print(f"y_train['fila'] shape: {y_train['fila'].shape}, dtype: {y_train['fila'].dtype}")
            print(f"y_train['paridad'] shape: {y_train['paridad'].shape}, dtype: {y_train['paridad'].dtype}")
            print(f"y_train['alto_bajo'] shape: {y_train['alto_bajo'].shape}, dtype: {y_train['alto_bajo'].dtype}")
            print(f"X_val shape: {X_val.shape}, dtype: {X_val.dtype}")
            print(f"y_val type: {type(y_val)}")
            print(f"y_val['color'] shape: {y_val['color'].shape}, dtype: {y_val['color'].dtype}")
            print(f"y_val['docena'] shape: {y_val['docena'].shape}, dtype: {y_val['docena'].dtype}")
            print(f"y_val['fila'] shape: {y_val['fila'].shape}, dtype: {y_val['fila'].dtype}")
            print(f"y_val['paridad'] shape: {y_val['paridad'].shape}, dtype: {y_val['paridad'].dtype}")
            print(f"y_val['alto_bajo'] shape: {y_val['alto_bajo'].shape}, dtype: {y_val['alto_bajo'].dtype}")

            try:
                history = train_model(model, X_train, y_train, X_val, y_val)
                
                if history is not None:
                    # Graficar la pérdida
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['loss'], label='Training Loss')
                    plt.plot(history.history['val_loss'], label='Validation Loss')
                    plt.title('Model Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.legend()
                    plt.show()
                else:
                    print("El entrenamiento no se completó correctamente.")
            except Exception as e:
                print(f"Error al graficar los resultados: {e}")

            model.save("modelo_ruleta_Embedding_V2.h5")

            # Imprimir información de formas y tipos de datos
            print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
            print(f"y_train type: {type(y_train)}")
            print(f"y_train['color'] shape: {y_train['color'].shape}, dtype: {y_train['color'].dtype}")
            print(f"y_train['docena'] shape: {y_train['docena'].shape}, dtype: {y_train['docena'].dtype}")
            print(f"y_train['fila'] shape: {y_train['fila'].shape}, dtype: {y_train['fila'].dtype}")
            print(f"y_train['paridad'] shape: {y_train['paridad'].shape}, dtype: {y_train['paridad'].dtype}")
            print(f"y_train['alto_bajo'] shape: {y_train['alto_bajo'].shape}, dtype: {y_train['alto_bajo'].dtype}")
            print(f"X_val shape: {X_val.shape}, dtype: {X_val.dtype}")
            print(f"y_val type: {type(y_val)}")
            print(f"y_val['color'] shape: {y_val['color'].shape}, dtype: {y_val['color'].dtype}")
            print(f"y_val['docena'] shape: {y_val['docena'].shape}, dtype: {y_val['docena'].dtype}")
            print(f"y_val['fila'] shape: {y_val['fila'].shape}, dtype: {y_val['fila'].dtype}")
            print(f"y_val['paridad'] shape: {y_val['paridad'].shape}, dtype: {y_val['paridad'].dtype}")
            print(f"y_val['alto_bajo'] shape: {y_val['alto_bajo'].shape}, dtype: {y_val['alto_bajo'].dtype}")

            try:
                history = train_model(model, X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"An error occurred during training: {e}")
                
            model = build_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                num_color_features=num_color_features,
                num_docena_features=num_docena_features,
                num_fila_features=num_fila_features,
                num_paridad_features=num_paridad_features,
                num_alto_bajo_features=num_alto_bajo_features
            )