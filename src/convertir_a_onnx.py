"""
convertir_a_onnx.py
Convierte el modelo Keras entrenado (modelo_dropout.keras) a formato ONNX
(modelo_dropout.onnx), para que la API pueda predecir con onnxruntime
(~50 MB) en vez de TensorFlow (~1.5 GB). Así la imagen Docker es mucho más
liviana y cabe en planes gratuitos con poca RAM (ej: Render free, 512 MB).

Se ejecuta solo cuando se reentrena el modelo:
    python src/convertir_a_onnx.py

La red es un MLP secuencial (Dense + Dropout), así que en vez de depender
de un conversor automático (tf2onnx), se arma el grafo ONNX directamente
con los pesos: cada Dense es un Gemm (X·W + b) seguido de su activación.
Los Dropout no se incluyen porque en inferencia no hacen nada.
"""

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from tensorflow import keras

from config import RUTA_MODELO_KERAS, RUTA_MODELO

# Activación de Keras -> operador ONNX equivalente
ACTIVACIONES_ONNX = {"relu": "Relu", "sigmoid": "Sigmoid"}


def keras_a_onnx(modelo) -> onnx.ModelProto:
    n_features = modelo.input_shape[-1]
    nodos, pesos = [], []
    actual = "input"

    capas_dense = [c for c in modelo.layers if isinstance(c, keras.layers.Dense)]
    for i, capa in enumerate(capas_dense):
        W, b = capa.get_weights()
        pesos += [
            numpy_helper.from_array(W.astype(np.float32), f"W{i}"),
            numpy_helper.from_array(b.astype(np.float32), f"b{i}"),
        ]
        nodos.append(helper.make_node("Gemm", [actual, f"W{i}", f"b{i}"], [f"z{i}"]))

        activacion = capa.get_config()["activation"]
        if activacion not in ACTIVACIONES_ONNX:
            raise ValueError(f"Activación no soportada en la conversión: {activacion}")
        actual = "output" if i == len(capas_dense) - 1 else f"a{i}"
        nodos.append(helper.make_node(ACTIVACIONES_ONNX[activacion], [f"z{i}"], [actual]))

    grafo = helper.make_graph(
        nodos,
        "modelo_dropout",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 1])],
        initializer=pesos,
    )
    # IR 9 / opset 17: formato fijo y conservador para que el archivo lo lea
    # cualquier onnxruntime reciente, aunque la librería onnx sea más nueva.
    modelo_onnx = helper.make_model(
        grafo, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.checker.check_model(modelo_onnx)
    return modelo_onnx


def verificar_equivalencia(modelo_keras, ruta_onnx: str, n: int = 1000, tolerancia: float = 1e-5):
    """Compara las predicciones de Keras y ONNX sobre inputs aleatorios."""
    import onnxruntime as ort

    X = np.random.default_rng(42).normal(size=(n, modelo_keras.input_shape[-1])).astype(np.float32)
    pred_keras = modelo_keras.predict(X, verbose=0)
    pred_onnx = ort.InferenceSession(ruta_onnx).run(None, {"input": X})[0]
    diferencia = float(np.abs(pred_keras - pred_onnx).max())
    if diferencia > tolerancia:
        raise AssertionError(f"ONNX difiere de Keras: diferencia máxima {diferencia}")
    return diferencia


if __name__ == "__main__":
    modelo_keras = keras.models.load_model(RUTA_MODELO_KERAS)
    onnx.save(keras_a_onnx(modelo_keras), RUTA_MODELO)
    diferencia = verificar_equivalencia(modelo_keras, RUTA_MODELO)
    print(f"Modelo guardado en {RUTA_MODELO} (diferencia máxima vs Keras: {diferencia:.2e})")
