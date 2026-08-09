"""
model.py
Carga el modelo Keras entrenado y expone una función simple de predicción,
para que la API (main.py) no tenga que preocuparse por los detalles de
TensorFlow/Keras ni del preprocesamiento.
"""

from tensorflow import keras

from preprocessing import cargar_scaler, cargar_columnas_esperadas, preparar_input
from config import RUTA_MODELO, RUTA_SCALER, RUTA_COLUMNAS, UMBRAL_DECISION


class ModeloDropout:
    """
    Envuelve el modelo entrenado + su scaler + sus columnas esperadas,
    para que predecir sea una sola llamada simple.
    """

    def __init__(
        self,
        ruta_modelo: str = RUTA_MODELO,
        ruta_scaler: str = RUTA_SCALER,
        ruta_columnas: str = RUTA_COLUMNAS,
    ):
        self.modelo = keras.models.load_model(ruta_modelo)
        self.scaler = cargar_scaler(ruta_scaler)
        self.columnas_esperadas = cargar_columnas_esperadas(ruta_columnas)

    def predecir(self, datos: dict, umbral: float = UMBRAL_DECISION) -> dict:
        X = preparar_input(datos, self.scaler, self.columnas_esperadas)

        probabilidad = float(self.modelo.predict(X, verbose=0).ravel()[0])
        prediccion = "Dropout" if probabilidad >= umbral else "Graduate"

        if probabilidad >= 0.7:
            riesgo = "alto"
        elif probabilidad >= 0.4:
            riesgo = "medio"
        else:
            riesgo = "bajo"

        return {
            "probabilidad_dropout": round(probabilidad, 4),
            "prediccion": prediccion,
            "riesgo": riesgo,
        }