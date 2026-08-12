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
├── tests/           # Suite de pytest (10 tests)
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

Correr el contenedor:
```
docker run -p 8000:8000 dropout-api
```

Luego abrir http://localhost:8000/docs

## Tests

```
pytest -v
```

10 tests: preprocesamiento (3), modelo (3), API (4).

## CI/CD

Cada push o pull request a `main` corre automáticamente la suite de tests vía GitHub Actions. El build de Docker se valida en cada push directo a `main`.

## Roadmap

- [x] Fase 1: Modularización
- [x] Fase 2: Dockerización
- [x] Fase 3: Testing
- [x] Fase 4: CI/CD
- [ ] Fase 5: Seguridad
- [ ] Fase 6: Deploy + Logging
- [ ] Fase 7: Documentación