# Imagen base
FROM python:3.11-slim

# Crear un usuario no root (mejora el security hotspot)
RUN useradd -m appuser

# Directorio de trabajo
WORKDIR /app

# Copiar dependencias y código
COPY requirements_docker.txt .
COPY MLops_pipeline/src/RandomForest_model.pkl .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_docker.txt

# Copiar solo lo necesario
COPY MLops_pipeline/src/model_deploy.py /app/MLops_pipeline/src/model_deploy.py
COPY MLops_pipeline/src/RandomForest_model.pkl /app/MLops_pipeline/src/RandomForest_model.pkl

# Cambiar el dueño de los archivos y usar un usuario sin privilegios
RUN chown -R appuser:appuser /app
USER appuser

# Exponer el puerto
EXPOSE 8000

# Comando de arranque
CMD ["uvicorn", "MLops_pipeline.src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
