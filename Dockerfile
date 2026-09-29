FROM python:3.12-slim

WORKDIR /app

# Ya no se instala build-essential: sin TensorFlow, todas las dependencias
# vienen precompiladas (wheels) y la imagen queda mucho más liviana.

# Copiamos requirements primero para aprovechar la cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código y los artefactos del modelo
COPY src/ ./src/
COPY api/ ./api/
COPY modelos/ ./modelos/

# Creamos un usuario sin privilegios y le damos dueño del directorio de trabajo
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
# --proxy-headers: en Render las requests llegan a través de su proxy; así
# uvicorn toma la IP real del cliente de X-Forwarded-For y el rate limit
# funciona por usuario (si no, todos compartirían la IP del proxy).
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips="*"
