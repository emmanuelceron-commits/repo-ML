# 💻 Proyecto de Machine Learning:

## 🐶🐱🐾 Clasificación de adoptadibilidad de mascotas 🐰🐹🎲

Este es un **proyecto para el curso de Machine Learning**, en el que se principalmente se busca realizar el **desarrollo y despliegue de un modelo supervisado predictivo** bajo una comprensión del negocio al que se brinda el proyecto como una solución. 

En este caso, la idea es desarrollar un modelo capaz predecir la **probabilidad de adopción de mascotas**, lo cual podría ayudar a entidades como los refugios de mascotas a plantear nuevas estrategias para priorizar y optimizar las adopciones.

Todo esto se puede lograr con la ayuda de una **base de datos de mascotas** (en este caso, un dataset de Kaggle) acompañada con una **variable objetivo** (como lo es AdoptionLikelihood en nuestro dataset), que permita diferenciar a mascotas más fáciles de adoptar todo esto **dentro del esquema de MLops**.

[Link del dataset original en Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/predict-pet-adoption-status-dataset/data)

## 🕵️ Algunos hallazgos del dataset durante la exploración del dataset

### ℹ️ Descripción general de los datos:

Este dataset de Kaggle contiene 2007 datos de mascotas en adopción, el cuál es sintético y fue recolectado en un periodo específico de tiempo con propósitos educacionales. 

> Si bien esto **no lo hace ideal para generalizar el comportamiento de las adopciones**, termina siendo ideal para proyectos de Machine Learning o Data Science con interés de aprender, predecir y entender tendencias de adopciones. 

Estos datos se pueden usar para:

- Modelamiento predictivo para determinar la adoptabilidad de una mascota

- Análisis de impacto de varios factores en las tasas de adopción

- Desarrollo de estrategias para incrementar las adopciones.

⛔ No hay nulos en el dataset

### 🔎📑📊 Hallazgos relevantes en la exploración de datos (EDA)

- Las mascotas con menos de 50 meses tienden a ser más adoptadas
- Las mascotas con más de 100 meses tienden a ser menos adoptadas

- La diferencia entre mascotas adoptadas y sin adoptar en el dataset es de 1 a 3 (un 33% aprox son adoptadas, un 66% están sin adoptar), lo cual puede ser un desbalanceo que deba considerarse en el modelamiento

Luego de revisar la relación entre variables categóricas y la variable objetivo:

- Si p ≈ 0 y Cramer's V > 0.3, hay relación real y relevante. Las variables Size y Vaccinated entran en esta categoría

- Si p ≈ 0 pero Cramer's V < 0.2 → relación estadísticamente detectable pero débil (Breed, PetType, HealthCondition).

- Si p es grande (ej. 0.37 en Color) → no hay casi evidencia de relación, y además V confirma que es irrelevante. Por lo que PreviousOwner y Color pueden no influir mucho en la adoptabilidad.

Posibles reglas de validación de datos:

- AgeMonths debe ser >= 0 y < 240.

- WeightKg > 0 y < 100.

- Categorías con muy pocos registros agrupar en 'Other' (ej. razas raras).

- Especie que coincida con raza

[Abrir notebook de comprensión_eda.ipynb para ver más detalles](./MLops_pipeline/src/comprension_eda.ipynb)

---

## 📁 Estructura del repositorio

<details><summary>(Desplegar para ver la estructura recomendada y planteada en clase)</summary>


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
</details>

---


(Estructura de archivos usados en el proyecto)
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
├── Base_de_datos.csv                       # Ubicación del dataset
├── config.json                             # Archivo de configuración de pipeline
├── requirements.txt                        # Librerías y dependencias
├── Dockerfile                              # Configuración docker
├── .dockerignore                              # Exclusiones de docker
├── .gitignore                              # Exclusiones de git
├── README.md                               # Documentación del proyecto
└── set_up.bat                              # Script para preparar el entorno
```
---
## 🛤️ Flujos de ejecución del repositorio

- Transformaciones, modelamiento y generación de métricas:

(Ubicarse en MLops_pipeline\src)

```
python ft_engineering.py
python model_training_evaluation.py
python model_monitoring.py
```
---
Despliegue de API con uvicorn:
```
uvicorn src.model_deploy:app --reload
```

- Enlace de pruebas: http://127.0.0.1:8000/docs
---
Ejecución de interfaz de Streamlit:
```
streamlit run app_streamlit.py
```

## 🐋 Construcción y ejecución de imagen de Docker

```
docker build -t pet-adoption-api .
docker run -p 8000:8000 pet-adoption-api
```
- Luego de ejecutar la imagen se pueden testear los endpoints en localhost:8000/docs

<details><summary>Datos de prueba para los endpoints (desplegar para ver)</summary>


Mascota con baja adoptabilidad:
```
{
  "PetType": "Rabbit",
  "Breed": "Rabbit",
  "AgeMonths": 70,
  "Color": "Gray",
  "Size": "Small",
  "WeightKg": 3.0,
  "Vaccinated": 0,
  "HealthCondition": 1,
  "TimeInShelterDays": 120,
  "AdoptionFee": 400,
  "PreviousOwner": 0
}
```
Mascota con alta probabilidad:
```
{
  "PetType": "Dog",
  "Breed": "Labrador",
  "AgeMonths": 10,
  "Color": "Brown",
  "Size": "Medium",
  "WeightKg": 18.0,
  "Vaccinated": 1,
  "HealthCondition": 0,
  "TimeInShelterDays": 15,
  "AdoptionFee": 150,
  "PreviousOwner": 1
}
```
Varias mascotas:
```
[
  {
    "PetType": "Dog",
    "Breed": "Golden Retriever",
    "AgeMonths": 8,
    "Color": "White",
    "Size": "Large",
    "WeightKg": 25.0,
    "Vaccinated": 1,
    "HealthCondition": 0,
    "TimeInShelterDays": 10,
    "AdoptionFee": 200,
    "PreviousOwner": 1
  },
  {
    "PetType": "Cat",
    "Breed": "Persian",
    "AgeMonths": 36,
    "Color": "Gray",
    "Size": "Small",
    "WeightKg": 5.0,
    "Vaccinated": 1,
    "HealthCondition": 0,
    "TimeInShelterDays": 60,
    "AdoptionFee": 300,
    "PreviousOwner": 0
  },
  {
    "PetType": "Rabbit",
    "Breed": "Rabbit",
    "AgeMonths": 72,
    "Color": "Brown",
    "Size": "Small",
    "WeightKg": 3.2,
    "Vaccinated": 0,
    "HealthCondition": 1,
    "TimeInShelterDays": 150,
    "AdoptionFee": 450,
    "PreviousOwner": 0
  }
]
```
en .csv:
```
PetType,Breed,AgeMonths,Color,Size,WeightKg,Vaccinated,HealthCondition,TimeInShelterDays,AdoptionFee,PreviousOwner
Dog,Golden Retriever,8,White,Large,25.0,1,0,10,200,1
Cat,Persian,36,Gray,Small,5.0,1,0,60,300,0
Rabbit,Rabbit,72,Brown,Small,3.2,0,1,150,450,0

```
</details>
---

## 📝 Resultados de pruebas en sonarcloud.io

1. Calidad del código: Evalúa la mantenibilidad del código fuente. 

- Código duplicado: 0.0% Duplications
- Mantenibilidad: A, significa que el ratio de deuda técnica es menor al 5%

- Complejidad ciclomática excesiva 
- Funciones demasiado largas o difíciles de entender 
- Malas prácticas de programación 

2. Seguridad: Detecta vulnerabilidades y puntos débiles que podrían ser explotados por  atacantes. 

- Puntuación de seguridad: A, 0 issues abiertos

3. Cobertura de Pruebas: Mide qué porcentaje del código está cubierto por pruebas unitarias o de integración.  

- Líneas de código ejecutadas durante las pruebas 
- Métodos y funciones validadas 

4. Integridad y Estilo: Verifica que el código siga convenciones de estilo y buenas prácticas.

- Nombres de variables y funciones 
- Uso adecuado de espacios, indentación y estructuras - Consistencia en la escritura del código
