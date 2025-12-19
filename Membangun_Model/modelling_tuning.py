import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

# Konfigurasi MLflow
mlflow.set_experiment("Telco_Churn_Tuning")

def train_and_tune():

    # 1. Load Data
    try:
        df = pd.read_csv('telco_preprocessing/telco_churn_processed.csv')
    except FileNotFoundError:
        print("Error: Dataset tidak ditemukan.")
        return

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Setup Grid Search
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 8, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 4]
    }
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 3. Evaluasi Model
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Hasil 
    print(f"\nParameter Terbaik: {best_params}")
    print(f"Metrics -> Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # 4. MLflow Manual Logging 
    with mlflow.start_run(run_name="Best_Tuned_RF"):
        
        # Log Parameter & Metrik
        mlflow.log_params(best_params)
        mlflow.log_metrics({"accuracy": acc, "recall": rec, "f1_score": f1})
        
        # Log Model
        signature = mlflow.models.infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        # Artefak 1: Confusion Matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close() # Tutup plot agar hemat memori
        
        # Artefak 2: Feature Importance
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        plt.figure(figsize=(10,6))
        plt.title("Top 10 Feature Importances")
        plt.bar(range(10), importances[indices], align="center")
        plt.xticks(range(10), X.columns[indices], rotation=45)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()
        
        print("Logging MLflow selesai. Artefak tersimpan.")

if __name__ == "__main__":
    train_and_tune()