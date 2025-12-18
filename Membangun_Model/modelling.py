import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Konfigurasi MLflow
mlflow.set_experiment("Telco_Churn_Basic")

def train_basic_model():

    # 1. Load Data
    try:
        df = pd.read_csv('telco_preprocessing/telco_churn_processed.csv')
    except FileNotFoundError:
        print("Error: File dataset tidak ditemukan. Pastikan 'telco_churn_processed.csv' ada.")
        return

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # 2. Split Data
    # Menggunakan stratify=y karena data imbalance (Churn vs No Churn)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Aktifkan Autologging
    # otomatis mencatat parameter default, metrik akurasi, dan model artifact
    mlflow.autolog()

    # 4. Training Model
    with mlflow.start_run(run_name="Basic_RandomForest"):
        # gunakan Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # 5. Evaluasi Sederhana 
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Training Selesai.")
        print(f"Accuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_basic_model()