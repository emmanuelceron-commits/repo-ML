# ============================================================
# 🐾 App de Monitoreo y Predicción de Adopción de Mascotas
# ============================================================
# Ejecutar con:  streamlit run app_streamlit.py
# ============================================================

import streamlit as st # type: ignore
import pandas as pd
import joblib
import numpy as np

# ==============================
# Configuración general
# ==============================
st.set_page_config(page_title="ML App Mascotas 🐶🐱", layout="wide")
st.title("🐾 Sistema Predictivo y Monitoreo - Adopción de Mascotas")

# ==============================
# Cargar modelo y preprocesador
# ==============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("RandomForest_model.pkl")
        return model
    except:
        st.error("❌ No se encontró el modelo entrenado (.pkl).")
        return None

model = load_model()

# ==============================
# Crear pestañas
# ==============================
# tab1, tab2 = st.tabs(["📈 Monitoreo del modelo", "🐕 Predicción de adopción"])
tab2, tab1 = st.tabs(["🐕 Predicción de adopción", "📈 Monitoreo del modelo"])

# ============================================================
# 🧮 TAB 1: Monitoreo
# ============================================================
with tab1:
    st.header("📊 Monitoreo y detección de Data Drift")

    try:
        df_drift = pd.read_csv("drift_results.csv")
        st.dataframe(df_drift, width="stretch")

        st.subheader("📉 Métricas de Drift (PSI y KS Test)")
        num_drift = df_drift[df_drift["Tipo"] == "Numérica"][["Variable", "Valor", "Alerta"]]
        st.bar_chart(num_drift.set_index("Variable")["Valor"])

        # Alertas generales
        if any(df_drift["Alerta"].str.contains("⚠️")):
            st.warning("⚠️ Se detectaron posibles cambios en la distribución de algunas variables.")
        else:
            st.success("✅ El modelo se mantiene estable. No se detecta drift significativo.")
    except FileNotFoundError as e:
        st.error("❌ No se encontró el archivo 'drift_results.csv'. Ejecuta primero model_monitoring.py.")
        raise e
# ============================================================
# 🐕 TAB 2: Predicción del modelo
# ============================================================
with tab2:
    st.header("🖥️🎯 Predicción de probabilidad de adopción")

    st.markdown("Completa la información de la mascota para estimar la **probabilidad de adopción**:")

    # Formularios divididos en columnas
    col1, col2, col3 = st.columns(3)

    with col1:
        pet_type = st.selectbox("Tipo de mascota", ["Dog", "Cat", "Rabbit", "Bird"])
        breed = st.selectbox("Raza", ["Labrador", "Golden Retriever", "Persian", "Siamese", "Poodle", "Parakeet", "Rabbit"])
        color = st.selectbox("Color", ["Black", "Brown", "Gray", "Orange", "White"])
        size = st.selectbox("Tamaño", ["Small", "Medium", "Large"])

    with col2:
        age = st.slider("Edad (meses)", 1, 180, 12)
        weight = st.number_input("Peso (kg)", min_value=0.5, max_value=40.0, value=10.0, step=0.5)
        adoption_fee = st.number_input("Tarifa de adopción ($)", min_value=0, max_value=500, value=100, step=10)
        time_in_shelter = st.number_input("Días en refugio", min_value=0, max_value=200, value=30)

    with col3:
        vaccinated = st.selectbox("¿Vacunado?", ["Sí", "No"])
        health_condition = st.selectbox("Condición médica", ["Saludable", "Con condición médica"])
        prev_owner = st.selectbox("¿Tuvo dueño previo?", ["Sí", "No"])

    # Preparar entrada
    if st.button("🔍 Predecir probabilidad de adopción"):
        if model is None:
            st.error("❌ No hay modelo cargado.")
        else:
            # Mapas usados en el feature engineering original
            size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
            color_map = {'Black': 0, 'Brown': 1, 'Gray': 2, 'Orange': 3, 'White': 4}

            # Construir DataFrame con columnas y valores esperados
            input_data = pd.DataFrame([{
                "AgeMonths": age,
                "WeightKg": weight,
                "TimeInShelterDays": time_in_shelter,
                "AdoptionFee": adoption_fee,
                "Size": size_map.get(size, 1),
                "Color": color_map.get(color, 2),
                "PetType": pet_type,
                "Breed": breed,
                "Vaccinated": 1 if vaccinated == "Sí" else 0,
                "HealthCondition": 1 if health_condition == "Con condición médica" else 0,
                "PreviousOwner": 1 if prev_owner == "Sí" else 0
            }])

            # Reordenar columnas según el preprocessor
            try:
                expected_cols = list(model.named_steps['preprocessor'].feature_names_in_)
                input_data = input_data[expected_cols]
            except Exception as e:
                st.warning("⚠️ No se pudieron reordenar las columnas automáticamente.")
                st.write(e)

            # Predicción
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

            st.subheader("📋 Resultado:")
            if pred == 1:
                st.success(f"🐶 Alta probabilidad de adopción ({prob:.2%})")
            else:
                st.warning(f"🐾 Baja probabilidad de adopción ({prob:.2%})")

            st.markdown("### Datos ingresados:")
            st.dataframe(input_data)


# ============================================================
# 🎨 Créditos
# ============================================================
st.markdown("---")
st.caption("Desarrollado por Emmanuel Cerón | Proyecto ML - Predicción de adopción de mascotas 🧠🐾")
