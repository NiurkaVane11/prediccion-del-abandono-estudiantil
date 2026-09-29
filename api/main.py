"""
main.py
API REST para predicción de deserción estudiantil.
Expone el modelo entrenado (model.py) como un endpoint HTTP.
"""

import sys
import os
import secrets
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Dict

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from model import ModeloDropout

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_dropout")

# ---------- Rate limiter ----------
# Limita por IP. Ajusta el límite según tu caso de uso real.
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="API de Predicción de Deserción Estudiantil",
    description="Predice el riesgo de abandono de un estudiante universitario",
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

modelo_dropout = ModeloDropout()

# ---------- API Key ----------
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY no está configurada. Define la variable de entorno API_KEY.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verificar_api_key(api_key: str = Security(api_key_header)):
    if api_key is None or not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(status_code=401, detail="API Key inválida o ausente")
    return api_key


# ---------- Features esperadas ----------
# Se toman del mismo archivo con el que se entrenó el modelo
# (modelos/columnas_features.pkl), así no hay una lista duplicada a mano
# que pueda desincronizarse si el modelo se reentrena.
FEATURES_ESPERADAS = frozenset(modelo_dropout.columnas_esperadas)


class EstudianteInput(BaseModel):
    features: Dict[str, float] = Field(..., description="Diccionario de 28 features numéricas")

    @field_validator("features")
    @classmethod
    def validar_columnas(cls, v: Dict[str, float]) -> Dict[str, float]:
        recibidas = set(v.keys())
        faltantes = FEATURES_ESPERADAS - recibidas
        sobrantes = recibidas - FEATURES_ESPERADAS

        if faltantes:
            raise ValueError(f"Faltan features requeridas: {sorted(faltantes)}")
        if sobrantes:
            raise ValueError(f"Features no reconocidas: {sorted(sobrantes)}")

        return v


# ---------- Manejo global de excepciones ----------
@app.exception_handler(Exception)
async def manejador_global(request: Request, exc: Exception):
    logger.error(
        "Error no manejado en %s %s: %s",
        request.method,
        request.url.path,
        repr(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Ocurrió un error interno. Intenta más tarde."},
    )


@app.get("/")
def raiz():
    return {"mensaje": "API de predicción de deserción estudiantil activa"}


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo_dropout.modelo is not None}


@app.post("/predecir")
@limiter.limit("10/minute")  # ajusta según tu caso: 10 requests por minuto por IP
def predecir(
    request: Request,  # requerido por slowapi para leer la IP del cliente
    estudiante: EstudianteInput,
    api_key: str = Depends(verificar_api_key),
):
    try:
        resultado = modelo_dropout.predecir(estudiante.features)
        return resultado
    except ValueError as e:
        logger.warning("ValueError en /predecir: %s", e)
        raise HTTPException(status_code=400, detail=str(e))