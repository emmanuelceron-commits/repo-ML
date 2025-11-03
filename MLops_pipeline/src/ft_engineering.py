# Feature engineering

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

def run_pipeline():
    # 1. Cargar datos
    df = pd.read_csv('pet_adoption_data.csv')

    # 2. Separar target y features
    y = df['AdoptionLikelihood']
    X = df.drop(columns=['AdoptionLikelihood', 'PetID'])

    # 3. Definir tipos de variables
    num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['category','object']).columns.tolist()

    # 4. Pipelines
    num_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    cat_pipe = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols)
        ]
    )

    # 5. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Guardar conjuntos
    joblib.dump((X_train, X_test, y_train, y_test, preprocessor), 'processed_data.pkl')
    print("✅ Datos procesados y guardados.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    run_pipeline()
