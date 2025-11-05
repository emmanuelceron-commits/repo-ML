# model_training_evaluation.py

from matplotlib import pyplot as plt
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# Cargar datos procesados
# X_train, X_test, y_train, y_test, preprocessor = joblib.load("processed_data.pkl")

def summarize_classification(y_true, y_pred, model_name):
    print(f"\n🔍 Resultados para {model_name}")
    print(classification_report(y_true, y_pred))
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "roc": roc_auc_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

def build_model(model, model_name):
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    metrics = summarize_classification(y_test, preds, model_name)
    joblib.dump(pipe, f"{model_name}_model.pkl")
    return metrics

if __name__ == "__main__":
    print("📦 Cargando datos procesados...")
    X_train, X_test, y_train, y_test, preprocessor = joblib.load("processed_data.pkl")

    # Entrenamiento de varios modelos
    results = []
    results.append(build_model(LogisticRegression(max_iter=5000, random_state=42, n_jobs=-1), "LogisticRegression"))
    results.append(build_model(RandomForestClassifier(n_estimators=200, random_state=42,n_jobs=-1, class_weight='balanced_subsample'), "RandomForest"))
    results.append(build_model(
        RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42), 
        "RandomForest_MaxDepth10"
    ))
    results.append(build_model(
        RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42), 
        "RandomForest_MaxDepthNone"
    ))
    results.append(build_model(DecisionTreeClassifier(random_state=42), "DecisionTree"))
    results.append(build_model(GradientBoostingClassifier(n_estimators=200, random_state=42), "GradientBoosting"))
    results.append(build_model(XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42,n_jobs=-1), "XGBoost"))
    # results.append(build_model(
    #     DecisionTreeClassifier(max_depth=20, random_state=42), 
    #     "DecisionTree_MaxDepth10"
    # ))
    
    # Comparar resultados
    df_results = pd.DataFrame(results)
    print("\n📊 Resultados comparativos:\n", df_results)

    best_model = df_results.sort_values(by="F1", ascending=False).iloc[0]["Model"]
    print(f"\n🏆 Mejor modelo: {best_model}")

    df_results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "roc"]].plot(kind='bar', figsize=(8,5))
    plt.title("Comparación de métricas entre modelos")
    plt.xticks(rotation=10)
    plt.grid(axis='y')
    plt.show()
    
    
