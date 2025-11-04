import joblib
pipe = joblib.load("RandomForest_model.pkl")
print(pipe.named_steps["preprocessor"].transformers)