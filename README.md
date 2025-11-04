## 💻 Proyecto de Machine Learning:

### 🐶🐱 Clasificación de adoptadibilidad de mascotas 🐾

Este es un proyecto del curso de Machine Learning, en el que se busca desarrollar y desplegar un modelo capaz predecir la probabilidad de adopción (AdoptionLikelihood) de mascotas, todo esto dentro del esquema de MLops.

[Link del dataset original](https://www.kaggle.com/datasets/rabieelkharoua/predict-pet-adoption-status-dataset/data)

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
│        ├── model_deploy.py                # Despliegue
│        └── app_streamlit.py               # Interfaz de streamlit
│
├── config.json                             # Archivo de configuración de pipeline
├── Base_de_datos.csv                       # Dataset
├── requirements.txt                        # Librerías y dependencias
├── .gitignore                              # Exclusiones de git
├── readme.md                               # Documentación del proyecto
└── set_up.bat                              # Script para preparar el entorno
```
