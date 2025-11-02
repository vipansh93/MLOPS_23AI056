import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load dataset
data = load_iris()
X = data.data
y = data.target

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Parameter combinations
param_combinations = [
    {"n_estimators": 50, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 7},
]

# 4. Set experiment name
mlflow.set_experiment("Intro_to_MLflow")

# 5. Loop through parameter combinations
for params in param_combinations:
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", params["n_estimators"])
        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_metric("accuracy", accuracy)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        # ✅ Generate Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix (n={params["n_estimators"]}, depth={params["max_depth"]})')

        # Save plot to file
        plot_path = f"confusion_matrix_{params['n_estimators']}_{params['max_depth']}.png"
        plt.savefig(plot_path)
        plt.close()

        # ✅ Log the plot as an artifact
        mlflow.log_artifact(plot_path)

        # ✅ Log the Python script itself
        script_path = "mlflow_assignment.py"
        if os.path.exists(script_path):
            mlflow.log_artifact(script_path)

        print(f"Logged run with n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, accuracy={accuracy:.4f}")
