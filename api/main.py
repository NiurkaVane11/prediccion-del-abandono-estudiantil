"""
main.py
API REST para predicción de deserción estudiantil.
Expone el modelo entrenado (model.py) como un endpoint HTTP.
"""

import sys
import os
import secrets
import logging
import re
import time
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Literal

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from model import ModeloDropout

# ---------- Logging ----------
# El nivel se controla con la variable de entorno LOG_LEVEL (DEBUG, INFO,
# WARNING, ERROR). Por defecto INFO. Los logs van a stdout, que es donde
# Docker y las plataformas de deploy los recogen.
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("api_dropout")

# ---------- Rate limiter ----------
# Limita por IP. Ajusta el límite según tu caso de uso real.
limiter = Limiter(key_func=get_remote_address)

DESCRIPCION_API = """
Predice el riesgo de que un estudiante universitario **abandone** sus estudios,
usando información disponible al terminar el **1er semestre** (red neuronal,
AUC-ROC 0.946 en test).

### Cómo usarla
1. Pulsa **Authorize** e ingresa tu API key (header `X-API-Key`).
2. Abre `POST /predecir` → **Try it out**: el body ya trae un estudiante real
   de ejemplo. Cambia los valores y pulsa **Execute**.

### Límites
- `POST /predecir`: máximo **10 requests por minuto** por IP (luego responde `429`).
- Las variables categóricas usan la **codificación del dataset de entrenamiento**
  (ej: `Course` va de 1 a 17). Ver la model card en el repositorio.

Código y documentación: https://github.com/NiurkaVane11/prediccion-del-abandono-estudiantil
"""

app = FastAPI(
    title="API de Predicción de Deserción Estudiantil",
    description=DESCRIPCION_API,
    version="1.0.0",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

modelo_dropout = ModeloDropout()
logger.info("Modelo cargado con %d features esperadas", len(modelo_dropout.columnas_esperadas))

# ---------- API Key ----------
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY no está configurada. Define la variable de entorno API_KEY.")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verificar_api_key(api_key: str = Security(api_key_header)):
    if api_key is None or not secrets.compare_digest(api_key, API_KEY):
        # Nunca se loguea la key recibida, solo si venía o no
        logger.warning("Intento de acceso con API key %s", "ausente" if api_key is None else "inválida")
        raise HTTPException(status_code=401, detail="API Key inválida o ausente")
    return api_key


# ---------- Features esperadas ----------
# Se toman del mismo archivo con el que se entrenó el modelo
# (modelos/columnas_features.pkl), así no hay una lista duplicada a mano
# que pueda desincronizarse si el modelo se reentrena.
FEATURES_ESPERADAS = frozenset(modelo_dropout.columnas_esperadas)


# Estudiante real del test set (Graduate, probabilidad de abandono ~4%).
# Aparece prellenado en /docs para que se pueda probar la API sin armar el JSON.
EJEMPLO_ESTUDIANTE = {
    "Marital status": 1,
    "Application mode": 1,
    "Application order": 2,
    "Course": 10,
    "Daytime/evening attendance": 1,
    "Previous qualification": 1,
    "Nacionality": 1,
    "Mother's qualification": 13,
    "Father's qualification": 27,
    "Mother's occupation": 10,
    "Father's occupation": 10,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 0,
    "Scholarship holder": 1,
    "Age at enrollment": 18,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 6,
    "Curricular units 1st sem (approved)": 6,
    "Curricular units 1st sem (grade)": 13.17,
    "Curricular units 1st sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74,
}


class EstudianteInput(BaseModel):
    features: Dict[str, float] = Field(
        ...,
        description="Las 28 features del estudiante (nombre exacto de la columna → valor numérico)",
        examples=[EJEMPLO_ESTUDIANTE],
    )

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


class PrediccionOutput(BaseModel):
    probabilidad_dropout: float = Field(..., ge=0, le=1, description="Probabilidad estimada de abandono (0 a 1)")
    prediccion: Literal["Dropout", "Graduate"] = Field(..., description="Clase predicha con umbral 0.5")
    riesgo: Literal["bajo", "medio", "alto"] = Field(
        ..., description="bajo: < 0.4 · medio: 0.4 a 0.7 · alto: ≥ 0.7"
    )


class SaludOutput(BaseModel):
    estado: str = Field(..., examples=["ok"])
    modelo_cargado: bool


# ---------- Log de cada request ----------
# Solo se acepta un X-Request-ID del cliente si es alfanumérico (con - o _),
# para que nadie pueda inyectar saltos de línea o texto falso en los logs.
REQUEST_ID_VALIDO = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Cada request recibe un ID (o reutiliza el que mande el cliente) para
    # poder seguirla en los logs y que el cliente lo pueda reportar.
    request_id = request.headers.get("X-Request-ID", "")
    if not REQUEST_ID_VALIDO.match(request_id):
        request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    inicio = time.perf_counter()

    response = await call_next(request)

    duracion_ms = (time.perf_counter() - inicio) * 1000
    logger.info(
        "request_id=%s metodo=%s ruta=%s status=%d duracion_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duracion_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# ---------- Manejo global de excepciones ----------
@app.exception_handler(Exception)
async def manejador_global(request: Request, exc: Exception):
    logger.error(
        "request_id=%s Error no manejado en %s %s: %s",
        getattr(request.state, "request_id", "-"),
        request.method,
        request.url.path,
        repr(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Ocurrió un error interno. Intenta más tarde."},
    )


@app.get("/", tags=["General"], summary="Mensaje de bienvenida")
def raiz():
    return {"mensaje": "API de predicción de deserción estudiantil activa"}


@app.get(
    "/salud",
    tags=["General"],
    summary="Estado de la API",
    description="Indica si la API está viva y el modelo cargado. Lo usa Render como health check.",
    response_model=SaludOutput,
)
def salud():
    return {"estado": "ok", "modelo_cargado": modelo_dropout.modelo is not None}


@app.post(
    "/predecir",
    tags=["Predicción"],
    summary="Predecir riesgo de abandono",
    description="Recibe las 28 features de un estudiante y devuelve la probabilidad de abandono, "
    "la clase predicha y un nivel de riesgo. Requiere el header `X-API-Key`.",
    response_model=PrediccionOutput,
    responses={
        401: {"description": "API key ausente o inválida"},
        422: {"description": "Faltan features, sobran features o algún valor no es numérico"},
        429: {"description": "Se superó el límite de 10 requests por minuto"},
    },
)
@limiter.limit("10/minute")  # ajusta según tu caso: 10 requests por minuto por IP
def predecir(
    request: Request,  # requerido por slowapi para leer la IP del cliente
    estudiante: EstudianteInput,
    api_key: str = Depends(verificar_api_key),
):
    try:
        resultado = modelo_dropout.predecir(estudiante.features)
        # Se loguea solo el resultado, no las features: son datos personales
        # del estudiante y no deben quedar guardados en los logs.
        logger.info(
            "request_id=%s prediccion=%s probabilidad=%.4f riesgo=%s",
            request.state.request_id,
            resultado["prediccion"],
            resultado["probabilidad_dropout"],
            resultado["riesgo"],
        )
        return resultado
    except ValueError as e:
        logger.warning("ValueError en /predecir: %s", e)
        raise HTTPException(status_code=400, detail=str(e))