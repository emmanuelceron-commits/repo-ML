# Imagen base
FROM python:3.11

# Directorio de trabajo
WORKDIR /app

# Copiar dependencias y código
COPY requirements_docker.txt .
COPY MLops_pipeline/src/RandomForest_model.pkl .

RUN pip install --upgrade pip \ 
&& pip install --no-cache-dir -r requirements_docker.txt

COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "MLops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
