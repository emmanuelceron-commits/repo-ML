# model_monitoring.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import joblib

# --- Función para PSI ---
def psi(expected, actual, bins=10):
    """Population Stability Index (PSI)"""
    expected_perc, bin_edges = np.histogram(expected, bins=bins)
    actual_perc, _ = np.histogram(actual, bins=bin_edges)
    expected_perc = expected_perc / len(expected)
    actual_perc = actual_perc / len(actual)
    psi_val = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-6) / (actual_perc + 1e-6)))
    return psi_val

print("📦 Cargando datos procesados...")
X_train, X_test, y_train, y_test, preprocessor = joblib.load("processed_data.pkl")

# Simular nuevos datos
X_new = X_test.sample(frac=0.6, random_state=42).copy()

# Convertir todos a DataFrame (por si vienen en arrays)
X_train = pd.DataFrame(X_train)
X_new = pd.DataFrame(X_new)

drift_results = []

# Analizar columnas numéricas y categóricas por separado
for col in X_train.columns:
    # Intentar identificar tipo de variable
    if np.issubdtype(X_train[col].dtype, np.number):
        stat, p_value = ks_2samp(X_train[col], X_new[col])
        drift_results.append({
            "Variable": col,
            "Tipo": "Numérica",
            "Métrica": "KS Test",
            "Valor": stat,
            "P-value": p_value,
            "Alerta": "⚠️ Drift" if p_value < 0.05 else "✅ Estable"
        })
    else:
        # Evitar columnas sin valores comunes
        comunes = set(X_train[col].dropna().unique()) & set(X_new[col].dropna().unique())

        # Evitar errores por columnas sin valores comunes o con pocos datos
        if len(comunes) <= 1:
            drift_results.append({
                "Variable": col,
                "Tipo": "Categórica",
                "Métrica": "Chi²",
                "Valor": None,
                "P-value": np.nan,
                "Alerta": "N/A (sin valores comunes o insuficientes)"
            })
            continue

        # Crear tabla de contingencia solo con valores comunes
        contingency = pd.crosstab(
            X_train[col][X_train[col].isin(comunes)],
            X_new[col][X_new[col].isin(comunes)]
        )

        # Saltar si la tabla queda vacía
        if contingency.empty:
            drift_results.append({
                "Variable": col,
                "Tipo": "Categórica",
                "Métrica": "Chi²",
                "Valor": None,
                "P-value": np.nan,
                "Alerta": "N/A (tabla vacía)"
            })
            continue

        # Test Chi²
        _, p_value, _, _ = chi2_contingency(contingency, correction=False)
        drift_results.append({
            "Variable": col,
            "Tipo": "Categórica",
            "Métrica": "Chi²",
            "Valor": None,
            "P-value": p_value,
            "Alerta": "⚠️ Drift" if p_value < 0.05 else "✅ Estable"
        })


# Convertir resultados a DataFrame
df_drift = pd.DataFrame(drift_results)
print("\n📊 Resultados de Drift:")
print(df_drift)

# Calcular PSI global (solo numéricas)
print("\n📈 Population Stability Index (PSI):")
for c in X_train.select_dtypes(include=[np.number]).columns:
    psi_val = psi(X_train[c], X_new[c])
    alerta = "⚠️ Alto Drift" if psi_val > 0.2 else "✅ Estable"
    print(f"{c}: {psi_val:.4f}  -> {alerta}")

df_drift.to_csv("drift_results.csv", index=False)
print("\n💾 Resultados guardados en 'drift_results.csv'")
