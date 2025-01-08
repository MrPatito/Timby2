import os
import tensorflow as tf
import logging

# Deshabilitar oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configurar nivel de logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR

# Deshabilitar advertencias deprecadas
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configurar warnings de Python
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configurar memoria de GPU si está disponible
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)