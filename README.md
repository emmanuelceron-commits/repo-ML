# 💻 Proyecto de Machine Learning:

## 🐶🐱🐾 Clasificación de adoptadibilidad de mascotas 🐰🐹🎲

Este es un **proyecto para el curso de Machine Learning**, en el que se principalmente se busca realizar el **desarrollo y despliegue de un modelo supervisado predictivo** bajo una comprensión del negocio al que se brinda el proyecto como una solución. 

En este caso, la idea es desarrollar un modelo capaz predecir la **probabilidad de adopción de mascotas**, lo cual podría ayudar a entidades como los refugios de mascotas a plantear nuevas estrategias para priorizar y optimizar las adopciones.

Todo esto se puede lograr con la ayuda de una **base de datos de mascotas** (en este caso, un dataset de Kaggle) acompañada con una **variable objetivo** (como lo es AdoptionLikelihood en nuestro dataset), que permita diferenciar a mascotas más fáciles de adoptar todo esto **dentro del esquema de MLops**.

[Link del dataset original en Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-pet-adoption-status-dataset/data)

## 📁 Estructura del repositorio

(estructura recomendada y planteada en clase)
```
repo-ML/
└── MLops_pipeline/
│   └── src/
│        ├── Cargar_datos.ipynb       # Carga de dataset
│        ├── comprension_eda.ipynb    # Análisis exploratorio
│        ├── ft_engineering.py        # Generación de features y creación de datasets
│        ├── heuristic_model.py       # Modelo base
│        ├── model_training.ipynb     # Entrenamiento y comparación de modelos
│        ├── model_deploy.ipynb       # Despliegue
│        ├── model_evaluation.ipynb   # Evaluación
│        └── model_monitoring.ipynb   # Monitoreo
│
├── config.json                       # Archivo de configuración de pipeline
├── Base_de_datos.csv                 # Dataset de ejm
├── requirements.txt                  # Librerías y dependencias
├── .gitignore                        # Exclusiones de git
├── readme.md                         # Documentación del proyecto
└── set_up.bat                        # Script para preparar el entorno
```

(estructura de archivos usados en el proyecto)
```
repo-ML/
└── MLops_pipeline/
│   └── src/                        
│        ├── Cargar_datos.ipynb             # Carga de dataset
│        ├── comprension_eda.ipynb          # Análisis exploratorio
│        ├── ft_engineering.py              # Generación de features
│        ├── model_training_evualation.py   # Entrenamiento y comparación de modelos
│        ├── model_monitoring.py            # Monitoreo
│        ├── model_deploy.py                # Despliegue (API)
│        └── app_streamlit.py               # Interfaz visual de streamlit
│
├── config.json                             # Archivo de configuración de pipeline
├── Base_de_datos.csv                       # Dataset
├── requirements.txt                        # Librerías y dependencias
├── .gitignore                              # Exclusiones de git
├── readme.md                               # Documentación del proyecto
└── set_up.bat                              # Script para preparar el entorno
```

## 🛤️ Flujos de ejecución

Transformaciones, modelamiento y generación de métricas:

```
python ft_engineering.py
python model_training_evaluation.py
python model_monitoring.py
```
Despliegue de API:

- uvicorn src.model_deploy:app --reload
- Enlace de pruebas: http://127.0.0.1:8000/docs

Ejecución de interfaz de Streamlit:

- streamlit run app_streamlit.py

## 🕵️ Algunos hallazgos durante la exploración de datos

### ℹ️ Descripción general de los datos:

Este dataset de Kaggle contiene datos de mascotas en adopción, el cuál es sintético, recolectado en un periodo específico de tiempo con propósitos educacionales. 

> Si bien esto **no lo hace ideal para generalizar el comportamiento de las adopciones**, termina siendo ideal para proyectos de Machine Learning o Data Science con interés de aprender, predecir y entender tendencias de adopciones. 

Estos datos se pueden usar para:

- Modelamiento predictivo para determinar la adoptabilidad de una mascota

- Análisis de impacto de varios factores en las tasas de adopción

- Desarrollo de estrategias para incrementar las adopciones.

[Abrir notebook para ver más detalles](./MLops_pipeline/src/comprension_eda.ipynb)
