# Predicción del Abandono Estudiantil

![CI](https://github.com/NiurkaVane11/prediccion-del-abandono-estudiantil/actions/workflows/ci.yml/badge.svg)

Modelo de aprendizaje automático para predecir la deserción estudiantil utilizando análisis exploratorio de datos (EDA) y algoritmos de clasificación.

**Modelo:** red neuronal (Keras) con AUC-ROC 0.9443, servido vía API REST con FastAPI.

## Estructura del proyecto

```
├── api/            # Endpoints FastAPI (/, /salud, /predecir)
├── src/             # Preprocesamiento (preprocessing.py) y clase del modelo (model.py)
├── modelos/         # Modelo entrenado, scaler y columnas de features
├── notebooks/       # EDA y entrenamiento
├── deploy/          # Script y README del deploy a Hugging Face Spaces
├── tests/           # Suite de pytest (17 tests)
├── Dockerfile
└── requirements.txt
```

## Instalación local

```
python -m venv venv
source venv/Scripts/activate  # Git Bash en Windows
pip install -r requirements.txt
```

## Docker

Construir la imagen:
```
docker build -t dropout-api .
```

Correr el contenedor (la API no arranca sin `API_KEY`):
```
docker run -p 8000:8000 -e API_KEY=tu-clave-secreta dropout-api
```

Luego abrir http://localhost:8000/docs

## Seguridad

- **API Key:** `/predecir` exige el header `X-API-Key` con el valor de la variable de entorno `API_KEY`. Sin ella, o con una incorrecta, responde `401`. La clave nunca se escribe en el código.
- **Rate limiting:** máximo 10 requests por minuto por IP en `/predecir`; al superarlo responde `429`.
- **Validación de entrada:** el body debe traer exactamente las 28 features con las que se entrenó el modelo (`modelos/columnas_features.pkl`); si falta o sobra alguna responde `422`.
- **Errores internos:** se registran en el log, pero al cliente solo se le devuelve un mensaje genérico (`500`), sin detalles internos.
- **Contenedor sin root:** la imagen Docker corre con un usuario sin privilegios (`appuser`).

Ejemplo de request:
```
curl -X POST http://localhost:8000/predecir   -H "Content-Type: application/json"   -H "X-API-Key: tu-clave-secreta"   -d '{"features": {"Marital status": 1, "...": 0}}'
```

## Logging

La API escribe sus logs en stdout (Docker y las plataformas de deploy los recogen desde ahí):

- Una línea por request: `request_id`, método, ruta, status y duración en ms.
- Una línea por predicción con el resultado (predicción, probabilidad, riesgo). **Las features del estudiante no se loguean**, porque son datos personales.
- Avisos (`WARNING`) de intentos de acceso sin API key o con una inválida (la key recibida nunca se escribe).
- Cada respuesta incluye el header `X-Request-ID` para buscar esa request en los logs.

El nivel se ajusta con la variable de entorno `LOG_LEVEL` (`DEBUG`, `INFO`, `WARNING`, `ERROR`; por defecto `INFO`):
```
docker run -p 8000:8000 -e API_KEY=tu-clave-secreta -e LOG_LEVEL=WARNING dropout-api
```

## Tests

```
pytest -v
```

17 tests: preprocesamiento (3), modelo (3), API (11, incluye autenticación, validación, rate limiting y request ID).

Los tests usan una `API_KEY` de prueba definida en `tests/conftest.py`, por eso no hace falta configurar ninguna clave para correrlos (ni en local ni en GitHub Actions).

## CI/CD

Cada push o pull request a `main` corre automáticamente la suite de tests vía GitHub Actions y valida el build de Docker. En los push a `main`, si todo pasa, se despliega en Hugging Face Spaces.

## Deploy (Hugging Face Spaces)

Cada push a `main` que pasa los tests y el build de Docker se despliega automáticamente en un Hugging Face Space (job `deploy` de GitHub Actions, script `deploy/subir_a_hf.py`). Solo se suben `Dockerfile`, `requirements.txt`, `api/`, `src/` y `modelos/`.

Configuración necesaria en GitHub (Settings → Secrets and variables → Actions):

| Tipo | Nombre | Valor |
|---|---|---|
| Secret | `HF_TOKEN` | Token de Hugging Face con permiso *write* |
| Secret | `API_KEY` | Clave que exigirá la API en producción |
| Variable | `HF_SPACE_ID` | `usuario-hf/nombre-del-space` |

Mientras `HF_SPACE_ID` no exista, el job de deploy se omite y el CI sigue en verde.

## Roadmap

- [x] Fase 1: Modularización
- [x] Fase 2: Dockerización
- [x] Fase 3: Testing
- [x] Fase 4: CI/CD
- [x] Fase 5: Seguridad
- [ ] Fase 6: Deploy + Logging
- [ ] Fase 7: Documentación