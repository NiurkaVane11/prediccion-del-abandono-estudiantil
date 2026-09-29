"""
model.py
Carga el modelo entrenado (en formato ONNX) y expone una función simple de
predicción, para que la API (main.py) no tenga que preocuparse por los
detalles de onnxruntime ni del preprocesamiento.

El modelo se entrenó con Keras y se convirtió con src/convertir_a_onnx.py:
en producción solo se necesita onnxruntime, no TensorFlow.
"""

import numpy as np
import onnxruntime as ort

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
        self.modelo = ort.InferenceSession(ruta_modelo)
        self.scaler = cargar_scaler(ruta_scaler)
        self.columnas_esperadas = cargar_columnas_esperadas(ruta_columnas)

    def predecir(self, datos: dict, umbral: float = UMBRAL_DECISION) -> dict:
        X = preparar_input(datos, self.scaler, self.columnas_esperadas)

        # ONNX espera float32 (el scaler devuelve float64)
        salida = self.modelo.run(None, {"input": X.astype(np.float32)})[0]
        probabilidad = float(salida.ravel()[0])
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