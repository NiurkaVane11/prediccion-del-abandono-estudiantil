import os

import httpx
import joblib
import pytest
from fastapi.testclient import TestClient

from cliente import ErrorAPI, construir_features, predecir
from config import RUTA_COLUMNAS

DATOS_FORMULARIO = dict(
    carrera=12,
    edad=18,
    genero_masculino=False,
    desplazado=True,
    becado=True,
    deudor=False,
    matricula_al_dia=True,
    materias_inscritas=6,
    materias_evaluadas=6,
    materias_aprobadas=6,
    materias_convalidadas=0,
    materias_sin_evaluacion=0,
    nota_promedio=13.0,
)


def test_construir_features_tiene_las_28_columnas_del_modelo():
    features = construir_features(**DATOS_FORMULARIO)
    assert set(features) == set(joblib.load(RUTA_COLUMNAS))


def test_carrera_de_turno_noche_marca_asistencia_nocturna():
    assert construir_features(**{**DATOS_FORMULARIO, "carrera": 17})["Daytime/evening attendance"] == 0
    assert construir_features(**{**DATOS_FORMULARIO, "carrera": 12})["Daytime/evening attendance"] == 1


def test_mas_aprobadas_que_inscritas_es_error():
    with pytest.raises(ValueError):
        construir_features(**{**DATOS_FORMULARIO, "materias_aprobadas": 7})


def test_formulario_de_punta_a_punta_contra_la_api():
    # El formulario -> construir_features -> API real (en memoria) -> predicción
    from api.main import app, limiter

    limiter.reset()
    features = construir_features(**DATOS_FORMULARIO)
    resultado = predecir(features, "http://testserver", os.environ["API_KEY"], cliente=TestClient(app))
    assert resultado["prediccion"] in ("Dropout", "Graduate")
    assert 0 <= resultado["probabilidad_dropout"] <= 1


@pytest.mark.parametrize("codigo,texto", [(401, "autorizada"), (429, "Demasiadas"), (500, "inesperado")])
def test_errores_http_se_traducen_a_mensajes_claros(codigo, texto):
    transporte = httpx.MockTransport(lambda request: httpx.Response(codigo))
    with pytest.raises(ErrorAPI, match=texto):
        predecir({}, "http://api", "clave", cliente=httpx.Client(transport=transporte))


def test_timeout_se_traduce_a_mensaje_claro():
    def lanzar_timeout(request):
        raise httpx.ReadTimeout("lento", request=request)

    with pytest.raises(ErrorAPI, match="tardó demasiado"):
        predecir({}, "http://api", "clave", cliente=httpx.Client(transport=httpx.MockTransport(lanzar_timeout)))
