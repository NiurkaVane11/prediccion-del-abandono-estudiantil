"""
preprocessing.py
Replica la limpieza y transformación de datos usada en el notebook de
entrenamiento, para que la API prediga con el mismo formato que el modelo
aprendió.
"""

import pandas as pd
import joblib

# Columnas eliminadas por data leakage (info que no existe antes de que
# el estudiante decida abandonar)
COLUMNAS_LEAKAGE = [
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)',
]


def cargar_scaler(ruta_scaler: str = '../modelos/scaler.pkl'):
    """Carga el StandardScaler ya ajustado en el entrenamiento."""
    return joblib.load(ruta_scaler)


def cargar_columnas_esperadas(ruta_columnas: str = '../modelos/columnas_features.pkl'):
    """Carga el orden exacto de columnas que el modelo espera."""
    return joblib.load(ruta_columnas)


def preparar_input(datos: dict, scaler, columnas_esperadas: list):
    """
    Convierte un input crudo (ej: JSON de la API) en un array listo
    para el modelo: mismas columnas, mismo orden, mismo escalado.

    Parámetros:
        datos: dict con las features del estudiante (sin Target, sin
               columnas de leakage)
        scaler: StandardScaler cargado con cargar_scaler()
        columnas_esperadas: lista cargada con cargar_columnas_esperadas()

    Retorna:
        array numpy escalado, listo para modelo.predict()
    """
    df_input = pd.DataFrame([datos])

    # Validación: que no falte ninguna columna esperada
    faltantes = [c for c in columnas_esperadas if c not in df_input.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")

    # Reordenamos exactamente igual que en el entrenamiento
    df_input = df_input[columnas_esperadas]

    # Escalamos con el mismo scaler del entrenamiento
    return scaler.transform(df_input)