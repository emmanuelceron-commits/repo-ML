from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io

# Inicializar la app FastAPI
app = FastAPI(
    title="API de Predicción de Probabilidad de Adopciones de Mascotas 🐾",
    description="Servicio que utiliza el mejor modelo entrenado para predecir la probabilidad de adopción de una mascota.",
    version="1.0.0"
)

# Cargar modelo entrenado
model_path = "RandomForest_model.pkl"  
model = joblib.load(model_path)

@app.get("/")
def home():
    return {"message": "API funcionando correctamente. Usa /predict o /predict_batch para hacer predicciones."}


@app.post("/predict")
def predict(data: dict):
    """
    Recibe un JSON con los datos de una sola mascota y devuelve la predicción y probabilidad.
    """
    try:
        df = pd.DataFrame([data])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]
        return {
            "prediction": int(pred),
            "probability": round(float(prob), 4)
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Permite subir un archivo CSV con múltiples registros.
    Devuelve las predicciones en formato JSON.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        preds = model.predict(df)
        probs = model.predict_proba(df)[:, 1]
        df["Prediction"] = preds
        df["Probability"] = probs
        return df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)
