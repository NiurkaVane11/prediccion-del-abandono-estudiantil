import os

import pytest
from fastapi.testclient import TestClient
from api.main import app, limiter, EJEMPLO_ESTUDIANTE

client = TestClient(app)

# Header con la clave que conftest.py definió para los tests
HEADERS = {"X-API-Key": os.environ["API_KEY"]}


@pytest.fixture(autouse=True)
def reiniciar_rate_limit():
    """Reinicia el contador de requests antes de cada test para que el
    límite de 10/minuto de un test no afecte a los demás."""
    limiter.reset()
    yield

# Mismo estudiante real (test set) que aparece de ejemplo en /docs
FEATURES_ESTUDIANTE = EJEMPLO_ESTUDIANTE


def test_raiz_responde_ok():
    response = client.get("/")
    assert response.status_code == 200
    assert "mensaje" in response.json()


def test_salud_responde_ok():
    response = client.get("/salud")
    assert response.status_code == 200
    assert response.json()["estado"] == "ok"
    assert response.json()["modelo_cargado"] is True


def test_predecir_con_datos_validos():
    response = client.post("/predecir", json={"features": FEATURES_ESTUDIANTE}, headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "probabilidad_dropout" in data
    assert "prediccion" in data
    assert "riesgo" in data


def test_predecir_estudiante_real_es_graduate():
    # Este estudiante se graduó; el modelo le da ~4% de probabilidad de
    # abandono. Si el modelo o la conversión ONNX se rompen, esto falla.
    response = client.post("/predecir", json={"features": FEATURES_ESTUDIANTE}, headers=HEADERS)
    data = response.json()
    assert data["prediccion"] == "Graduate"
    assert data["riesgo"] == "bajo"
    assert data["probabilidad_dropout"] < 0.1


def test_predecir_con_columnas_faltantes():
    # Pydantic rechaza el input antes de llegar al modelo -> 422
    response = client.post("/predecir", json={"features": {"Marital status": 1}}, headers=HEADERS)
    assert response.status_code == 422
    assert "Faltan features requeridas" in response.text


def test_predecir_con_feature_desconocida():
    features = {**FEATURES_ESTUDIANTE, "columna_inventada": 1}
    response = client.post("/predecir", json={"features": features}, headers=HEADERS)
    assert response.status_code == 422
    assert "Features no reconocidas" in response.text


def test_predecir_sin_api_key():
    response = client.post("/predecir", json={"features": FEATURES_ESTUDIANTE})
    assert response.status_code == 401


def test_predecir_con_api_key_incorrecta():
    response = client.post(
        "/predecir",
        json={"features": FEATURES_ESTUDIANTE},
        headers={"X-API-Key": "clave-incorrecta"},
    )
    assert response.status_code == 401


def test_predecir_supera_rate_limit():
    # Las primeras 10 requests del minuto pasan, la 11 se bloquea con 429
    for _ in range(10):
        response = client.post("/predecir", json={"features": FEATURES_ESTUDIANTE}, headers=HEADERS)
        assert response.status_code == 200
    response = client.post("/predecir", json={"features": FEATURES_ESTUDIANTE}, headers=HEADERS)
    assert response.status_code == 429


def test_respuesta_incluye_request_id():
    response = client.get("/salud")
    assert len(response.headers["X-Request-ID"]) == 12


def test_request_id_del_cliente_se_respeta_si_es_valido():
    response = client.get("/salud", headers={"X-Request-ID": "mi-id-123"})
    assert response.headers["X-Request-ID"] == "mi-id-123"


def test_request_id_malicioso_se_reemplaza():
    response = client.get("/salud", headers={"X-Request-ID": "x status=200 FALSO"})
    assert response.headers["X-Request-ID"] != "x status=200 FALSO"
