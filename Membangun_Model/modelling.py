import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Setup Eksperimen
mlflow.set_experiment("Telco_Churn_Baseline")

def train_baseline():
    
    # 1. Load Data
    try:
        df = pd.read_csv('telco_preprocessing/telco_churn_processed.csv')
    except FileNotFoundError:
        print("Error: Dataset tidak ditemukan.")
        return

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # 2. Split Data (Stratify wajib)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Autologging
    mlflow.autolog()

    with mlflow.start_run(run_name="Baseline_RF_Default"):
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        
        model.fit(X_train, y_train)
        
        # Evaluasi
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n--- RESULT ---")
        print(f"Accuracy : {acc:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")

if __name__ == "__main__":
    train_baseline()