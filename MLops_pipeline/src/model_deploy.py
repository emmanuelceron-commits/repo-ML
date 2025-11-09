from fastapi import FastAPI, UploadFile, File, Request, Body # type: ignore
from fastapi.responses import JSONResponse # type: ignore
import pandas as pd
import joblib
import io
from typing import List, Dict, Any

# Inicializar la app FastAPI
app = FastAPI(
    title="API de Predicción de Probabilidad de Adopciones de Mascotas 🐾",
    description="Servicio que utiliza el mejor modelo entrenado para predecir la probabilidad de adopción de una mascota.",
    version="1.0.0"
)

# Cargar modelo entrenado
model_path = "RandomForest_model.pkl"  
model = joblib.load(model_path)

# Mapas ordinales (deben coincidir con los usados en feature_engineering)
SIZE_MAP = {'Small': 0, 'Medium': 1, 'Large': 2}
COLOR_MAP = {'Black': 0, 'Brown': 1, 'Gray': 2, 'Orange': 3, 'White': 4}


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que el DataFrame tenga las columnas con tipos esperados por el pipeline:
    - Convierte Size y Color a numérico si vienen como texto.
    - Reordena las columnas según el preprocessor del modelo.
    """
    # Si columnas ordinales vienen como texto, mapearlas
    if 'Size' in df.columns:
        df['Size'] = df['Size'].map(SIZE_MAP).astype(float)
    if 'Color' in df.columns:
        df['Color'] = df['Color'].map(COLOR_MAP).astype(float)

    # Si hay columnas binarias como 'Sí'/'No', convertir a 1/0 (por seguridad)
    for col in ['Vaccinated', 'HealthCondition', 'PreviousOwner']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].replace({'Sí': 1, 'Si': 1, 'No': 0, 'no': 0, 'sí': 1, 'si': 1}).astype(float)

    # Reordenar columnas según el preprocessor (si está disponible en el pipeline)
    try:
        expected = list(model.named_steps['preprocessor'].feature_names_in_)
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas requeridas para la predicción: {missing}")
        df = df[expected]
    except Exception as e:
        # Si no hay modelo o no existe feature_names_in_, devolvemos el df como está y el error será manejado arriba
        print(f"Error: {e}")
        raise e

    return df


@app.get("/")
def home():
    return {"message": "API funcionando correctamente. Usa /predict o /predict_batch para hacer predicciones."}


@app.post("/predict")
async def predict(payload: Any = Body(...)):
    """
    Recibe un JSON con datos de una o varias mascotas.
    """
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Modelo no disponible en el servidor."})

    try:
        # Normalizar payload a DataFrame
        if isinstance(payload, dict):
            df = pd.DataFrame([payload])
        elif isinstance(payload, list):
            df = pd.DataFrame(payload)
        else:
            return JSONResponse(status_code=400, content={"error": "JSON inválido: enviar objeto o lista de objetos."})

        # Preparar df (mapear ordinals y reordenar)
        df_prepared = _prepare_dataframe(df.copy())

        # Predicciones
        preds = model.predict(df_prepared)
        probs = model.predict_proba(df_prepared)[:, 1]

        # Formatear salida
        results = [
            {"prediction": int(preds[i]), "probability": float(probs[i])}
            for i in range(len(df_prepared))
        ]

        if isinstance(payload, dict):
            return results[0]
        return results

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"{type(e).__name__}: {e}"})
    
@app.post("/predict_batch")
async def predict_batch(
    file: UploadFile = None,
    payload: Any = Body(None)
):
    """
    Permite subir un archivo CSV o enviar una lista de JSON con varios registros.
    Devuelve las predicciones y probabilidades para cada mascota.
    """
    try:
        # --- Opción 1: Archivo CSV ---
        if file:
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
            df_prepared = _prepare_dataframe(df.copy())

        # --- Opción 2: Lista de JSON ---
        elif payload:
            if isinstance(payload, list):
                df = pd.DataFrame(payload)
            elif isinstance(payload, dict):
                df = pd.DataFrame([payload])
            else:
                return JSONResponse(status_code=400, content={"error": "Formato JSON inválido."})
            df_prepared = _prepare_dataframe(df.copy())

        else:
            return JSONResponse(status_code=400, content={"error": "Debes subir un archivo CSV o enviar una lista de JSON."})

        # Predicciones
        preds = model.predict(df_prepared)
        probs = model.predict_proba(df_prepared)[:, 1]

        df["Prediction"] = preds
        df["Probability"] = probs

        return df.to_dict(orient="records")

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"error": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"{type(e).__name__}: {e}"})
