# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="🐾 Predicción de Adopción", layout="centered")

st.title("🐶 Predicción de Adopción de Mascotas")
st.markdown("Ingrese la información de la mascota para estimar la probabilidad de adopción.")

# Cargar modelo entrenado
@st.cache_resource
def load_model():
    return joblib.load("RandomForest_model.pkl")

model = load_model()

# --- Entrada de usuario ---
col1, col2 = st.columns(2)
with col1:
    pet_type = st.selectbox("Tipo de mascota", ["Dog", "Cat", "Rabbit", "Bird"])
    breed = st.text_input("Raza", "Labrador")
    color = st.selectbox("Color", ["Black", "Brown", "Gray", "Orange", "White"])
    size = st.selectbox("Tamaño", ["Small", "Medium", "Large"])
    vaccinated = st.selectbox("¿Vacunado?", ["Sí", "No"])
with col2:
    age = st.slider("Edad (meses)", 1, 180, 12)
    weight = st.number_input("Peso (kg)", 1.0, 50.0, 10.0)
    shelter_days = st.number_input("Días en refugio", 1, 200, 15)
    adoption_fee = st.number_input("Cuota de adopción ($)", 0, 500, 100)
    prev_owner = st.selectbox("¿Tuvo dueño previo?", ["Sí", "No"])
    health = st.selectbox("Condición médica", ["Sano", "Con condición médica"])

# --- Botón de predicción ---
if st.button("🔮 Predecir"):
    df = pd.DataFrame([{
        "PetType": pet_type,
        "Breed": breed,
        "AgeMonths": age,
        "Color": color,
        "Size": size,
        "WeightKg": weight,
        "Vaccinated": 1 if vaccinated == "Sí" else 0,
        "HealthCondition": 1 if health == "Con condición médica" else 0,
        "TimeInShelterDays": shelter_days,
        "AdoptionFee": adoption_fee,
        "PreviousOwner": 1 if prev_owner == "Sí" else 0
    }])

    prediction = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    st.subheader("Resultado:")
    st.metric(label="Probabilidad de adopción", value=f"{prob*100:.1f}%")
    if prediction == 1:
        st.success("🐾 Alta probabilidad de adopción")
    else:
        st.error("💤 Baja probabilidad de adopción")
