# Feature engineering

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

size_map = {'Small': 0, 'Medium': 1, 'Large': 2}              # Small < Medium < Large
color_map = {'Black': 0, 'Brown': 1, 'Gray': 2, 'Orange': 3, 'White': 4}  # oscuro -> claro


def run_pipeline():

    # Carga
    PATH = "../../Base_de_datos.csv"   
    df = pd.read_csv(PATH)

    # borrar id si existe
    if 'PetID' in df.columns:
        df = df.drop(columns=['PetID'])

    # target y features
    y = df['AdoptionLikelihood'].astype(int)
    X = df.drop(columns=['AdoptionLikelihood'])

    # columnas esperadas (se usan solo si existen)
    num_cols = [c for c in ['AgeMonths','WeightKg','TimeInShelterDays','AdoptionFee'] if c in X.columns]
    cat_cols = [c for c in ['PetType','Breed'] if c in X.columns]
    bin_cols = [c for c in ['Vaccinated','HealthCondition','PreviousOwner'] if c in X.columns]

    # transformar Size y Color a ordinal numérico (si existen)
    if 'Size' in X.columns:
        X['Size'] = X['Size'].fillna(X['Size'].mode()[0])
        X['Size'] = X['Size'].map(size_map)
        X['Size'] = X['Size'].fillna(int(X['Size'].median()))
        # meter Size entre num_cols
        num_cols = num_cols + ['Size']

    if 'Color' in X.columns:
        X['Color'] = X['Color'].fillna(X['Color'].mode()[0])
        X['Color'] = X['Color'].map(color_map)
        X['Color'] = X['Color'].fillna(int(X['Color'].median()))
        # meter Color entre num_cols
        num_cols = num_cols + ['Color']

    # pipelines simples
    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median'))])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    bin_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent'))])  # 0/1, por si hay nulos

    # ColumnTransformer
    transformers = []
    if num_cols: transformers.append(('num', num_pipe, num_cols))
    if cat_cols: transformers.append(('cat', cat_pipe, cat_cols))
    if bin_cols: transformers.append(('bin', bin_pipe, bin_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')

    # split estratificado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # guardar lo necesario
    joblib.dump((X_train, X_test, y_train, y_test, preprocessor), 'processed_data.pkl')
    print("Guardado processed_data.pkl")
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    run_pipeline()
