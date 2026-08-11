FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema necesarias para tensorflow/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiamos requirements primero para aprovechar la cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código y los artefactos del modelo
COPY src/ ./src/
COPY api/ ./api/
COPY modelos/ ./modelos/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]