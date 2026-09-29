# Predicción del Abandono Estudiantil

![CI](https://github.com/NiurkaVane11/prediccion-del-abandono-estudiantil/actions/workflows/ci.yml/badge.svg)

Modelo de aprendizaje automático para predecir la deserción estudiantil utilizando análisis exploratorio de datos (EDA) y algoritmos de clasificación.

**API en producción:** https://prediccion-abandono-api.onrender.com/docs

**Modelo:** red neuronal (Keras) con AUC-ROC 0.9443, exportada a ONNX y servida vía API REST con FastAPI + onnxruntime.

## Estructura del proyecto

```
├── api/            # Endpoints FastAPI (/, /salud, /predecir)
├── src/             # Preprocesamiento, clase del modelo y conversión Keras -> ONNX
├── modelos/         # Modelo Keras (.keras), versión de producción (.onnx), scaler y columnas
├── notebooks/       # EDA y entrenamiento
├── tests/           # Suite de pytest (18 tests)
├── Dockerfile
├── render.yaml      # Configuración del servicio en Render
├── requirements.txt       # Dependencias de producción (sin TensorFlow)
└── requirements-dev.txt   # Producción + TensorFlow, ONNX y herramientas de test
```

## Instalación local

```
python -m venv venv
source venv/Scripts/activate  # Git Bash en Windows
pip install -r requirements-dev.txt
```

## Modelo en producción: ONNX

La red se entrena con Keras, pero la API predice con **onnxruntime** (~50 MB) en vez de TensorFlow (~1.5 GB): la imagen Docker es mucho más liviana, arranca más rápido y cabe en planes gratuitos con 512 MB de RAM.

Si se reentrena el modelo, hay que regenerar el `.onnx`:
```
python src/convertir_a_onnx.py
```
El script verifica que ONNX y Keras den la misma predicción, y el test `test_modelo_onnx_equivale_a_keras` lo vuelve a comprobar en cada CI.

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

18 tests: preprocesamiento (3), modelo (4, incluye equivalencia ONNX vs Keras), API (11, incluye autenticación, validación, rate limiting y request ID).

Los tests usan una `API_KEY` de prueba definida en `tests/conftest.py`, por eso no hace falta configurar ninguna clave para correrlos (ni en local ni en GitHub Actions).

## CI/CD

Cada push o pull request a `main` corre automáticamente la suite de tests vía GitHub Actions y valida el build de Docker. En los push a `main`, si todo pasa, se despliega en Hugging Face Spaces.

## Deploy (Render)

La API está desplegada en [Render](https://render.com) (plan gratuito) en https://prediccion-abandono-api.onrender.com usando el `Dockerfile` y la configuración de `render.yaml`:

- **Auto-deploy:** Render redespliega en cada push a `main`, solo después de que pasan los checks de GitHub Actions.
- **Health check:** Render consulta `/salud` antes de enviar tráfico a una versión nueva.
- **`API_KEY`:** se configura como variable de entorno en el panel de Render (nunca en el repo).
- **Plan gratuito:** el servicio se duerme tras ~15 min sin uso; la primera request después tarda unos segundos en responder.

## Roadmap

- [x] Fase 1: Modularización
- [x] Fase 2: Dockerización
- [x] Fase 3: Testing
- [x] Fase 4: CI/CD
- [x] Fase 5: Seguridad
- [x] Fase 6: Deploy + Logging
- [ ] Fase 7: Documentación