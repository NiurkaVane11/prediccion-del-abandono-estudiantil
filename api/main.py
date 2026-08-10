"""
main.py
API REST para predicción de deserción estudiantil.
Expone el modelo entrenado (model.py) como un endpoint HTTP.
"""

import sys
import os

# Permite importar model.py y preprocessing.py desde src/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

from model import ModeloDropout

app = FastAPI(
    title="API de Predicción de Deserción Estudiantil",
    description="Predice el riesgo de abandono de un estudiante universitario",
    version="1.0.0",
)

# El modelo se carga UNA sola vez al iniciar la API, no en cada request
modelo_dropout = ModeloDropout()


class EstudianteInput(BaseModel):
    """
    Representa las features de un estudiante. Usamos Dict porque son
    28 features y sería muy largo declararlas una por una — Pydantic
    igual valida que sea un dict de valores numéricos.
    """
    features: Dict[str, float]


@app.get("/")
def raiz():
    return {"mensaje": "API de predicción de deserción estudiantil activa"}


@app.get("/salud")
def salud():
    """Endpoint simple para verificar que la API y el modelo están vivos."""
    return {"estado": "ok", "modelo_cargado": modelo_dropout.modelo is not None}


@app.post("/predecir")
def predecir(estudiante: EstudianteInput):
    try:
        resultado = modelo_dropout.predecir(estudiante.features)
        return resultado
    except ValueError as e:
        # Ej: si faltan columnas requeridas
        raise HTTPException(status_code=400, detail=str(e))