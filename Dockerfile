# Imagen base
FROM python:3.11

# Directorio de trabajo
WORKDIR /app

# Copiar dependencias y código
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "MLops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
