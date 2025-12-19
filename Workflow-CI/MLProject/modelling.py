import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Nama eksperimen untuk Production
mlflow.set_experiment("Telco_Churn_Production")

def train():
    print("=== Training Model Production (Best Config) ===")

    # 1. Load Data
    try:
        # Gunakan nama folder dataset Anda
        df = pd.read_csv('telco_preprocessing/telco_churn_processed.csv')
    except FileNotFoundError:
        print("Error: Dataset tidak ditemukan. Cek path 'telco_preprocessing'.")
        return

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Model Definition (Pakai Parameter JUARA Hasil Tuning)
    best_params = {
        'n_estimators': 100,
        'max_depth': 8,
        'min_samples_leaf': 4,
        'min_samples_split': 5,
        'class_weight': 'balanced',  # <--- Kunci Recall Tinggi
        'random_state': 42
    }
    
    print(f"Melatih model dengan parameter: {best_params}")
    
    with mlflow.start_run(run_name="Production_Release"):
        model = RandomForestClassifier(**best_params)
        model.fit(X_train, y_train)

        # 4. Evaluasi
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Metrics -> Accuracy: {acc:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

        # 5. Log ke MLflow
        mlflow.log_params(best_params)
        mlflow.log_metrics({"accuracy": acc, "recall": rec, "f1_score": f1})
        
        # Log Model dengan Signature (Wajib untuk Deployment/Serving)
        signature = mlflow.models.infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        print("Model production berhasil disimpan.")

if __name__ == "__main__":
    train()