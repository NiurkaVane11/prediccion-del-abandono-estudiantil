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

# Creamos un usuario sin privilegios y le damos dueño del directorio de trabajo.
# UID 1000 porque Hugging Face Spaces ejecuta el contenedor con ese UID; así
# el usuario coincide y tiene un home donde Keras puede escribir su caché.
RUN useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app

# Cambiamos al usuario no-root para el resto de la ejecución
USER appuser

# Por defecto 8000, pero las plataformas de deploy (Render, Cloud Run,
# Railway, etc.) asignan su propio puerto en la variable PORT.
ENV PORT=8000
EXPOSE 8000

# Forma "shell" para que ${PORT} se expanda; exec hace que uvicorn reciba
# directamente las señales de apagado del contenedor.
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}
