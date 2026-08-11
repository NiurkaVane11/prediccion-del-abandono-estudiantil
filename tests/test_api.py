from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

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
    response = client.post("/predecir", json={"features": FEATURES_ESTUDIANTE})
    assert response.status_code == 200
    data = response.json()
    assert "probabilidad_dropout" in data
    assert "prediccion" in data
    assert "riesgo" in data


def test_predecir_con_columnas_faltantes():
    response = client.post("/predecir", json={"features": {"Marital status": 1}})
    assert response.status_code == 400
    assert "Faltan columnas requeridas" in response.json()["detail"]