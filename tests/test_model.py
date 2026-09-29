from model import ModeloDropout

FEATURES_ESTUDIANTE = {
    "Marital status": 1,
    "Application mode": 1,
    "Application order": 1,
    "Course": 9500,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 1,
    "Father's qualification": 1,
    "Mother's occupation": 1,
    "Father's occupation": 1,
    "Displaced": 0,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 0,
    "Scholarship holder": 0,
    "Age at enrollment": 19,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 6,
    "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 14.0,
    "Curricular units 1st sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74,
}


def test_modelo_carga_correctamente():
    modelo = ModeloDropout()
    assert modelo.modelo is not None
    assert modelo.scaler is not None
    assert len(modelo.columnas_esperadas) == 28


def test_prediccion_devuelve_estructura_correcta():
    modelo = ModeloDropout()
    resultado = modelo.predecir(FEATURES_ESTUDIANTE)

    assert "probabilidad_dropout" in resultado
    assert "prediccion" in resultado
    assert "riesgo" in resultado
    assert 0 <= resultado["probabilidad_dropout"] <= 1
    assert resultado["prediccion"] in ["Dropout", "Graduate"]
    assert resultado["riesgo"] in ["alto", "medio", "bajo"]


def test_prediccion_lanza_error_si_faltan_columnas():
    import pytest
    modelo = ModeloDropout()
    datos_incompletos = {"Marital status": 1}

    with pytest.raises(ValueError):
        modelo.predecir(datos_incompletos)

def test_modelo_onnx_equivale_a_keras():
    # Garantiza que modelo_dropout.onnx (producción) predice lo mismo que el
    # modelo Keras original. Si alguien reentrena y olvida reconvertir, falla.
    from tensorflow import keras
    from config import RUTA_MODELO_KERAS, RUTA_MODELO
    from convertir_a_onnx import verificar_equivalencia

    modelo_keras = keras.models.load_model(RUTA_MODELO_KERAS)
    assert verificar_equivalencia(modelo_keras, RUTA_MODELO) < 1e-5
