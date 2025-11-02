import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# XGBoost (install if not already)
try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not found. Installing...")
    import os
    os.system("pip install xgboost")
    from xgboost import XGBClassifier

# 1. Load dataset
data = load_iris()
X = data.data
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Define algorithms to compare
models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
}

# 4. Set experiment name
mlflow.set_experiment("Algorithm_Comparison")

# 5. Loop through models
for model_name, model in models.items():
    with mlflow.start_run():
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        # Log parameters and metrics
        mlflow.log_param("algorithm", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name}: Accuracy={acc:.4f}, F1-score={f1:.4f}")
