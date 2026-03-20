FROM python:3.11-slim

# deps nativas mínimas (git a veces lo usan libs, y build essentials por si algo compila)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias primero para aprovechar cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Importante: que Python encuentre "src"
ENV PYTHONPATH=/app

# Por defecto: ejecuta el chat CLI
CMD ["python", "main.py"]