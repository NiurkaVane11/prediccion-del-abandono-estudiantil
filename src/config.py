"""
config.py
Rutas y constantes centralizadas del proyecto, basadas en la ubicación
de este archivo (no en el directorio desde donde se ejecute el proceso).
Esto evita que las rutas relativas se rompan según desde dónde corras
la API o el notebook.
"""

import os

# Carpeta donde vive este archivo (src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta modelos/ está un nivel arriba de src/
MODELOS_DIR = os.path.join(BASE_DIR, '..', 'modelos')

# Modelo original de entrenamiento (Keras) y su versión para producción
# (ONNX), generada con src/convertir_a_onnx.py. La API usa la ONNX.
RUTA_MODELO_KERAS = os.path.join(MODELOS_DIR, 'modelo_dropout.keras')
RUTA_MODELO = os.path.join(MODELOS_DIR, 'modelo_dropout.onnx')
RUTA_SCALER = os.path.join(MODELOS_DIR, 'scaler.pkl')
RUTA_COLUMNAS = os.path.join(MODELOS_DIR, 'columnas_features.pkl')

# Umbral de decisión Dropout vs Graduate (el mismo usado en la evaluación)
UMBRAL_DECISION = 0.5