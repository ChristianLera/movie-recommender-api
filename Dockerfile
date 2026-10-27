FROM python:3.11-slim

# Evita .pyc y fuerza salida sin buffer (logs en tiempo real)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias del sistema mínimas (scipy/pyarrow a veces necesitan compilar)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1) Copiar solo requirements primero -> mejor caché
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Copiar el código y los datos/modelo (van en el repo porque son pequeños)
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 8000

# Healthcheck usando /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Arrancamos uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
