import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def automate_preprocessing(raw_path, output_path):
    """
    Fungsi preprocessing otomatis untuk dataset Telco Customer Churn.
    """
    print("🚀 Memulai Preprocessing...")
    
    # 1. Load Data
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File {raw_path} tidak ditemukan!")
    
    df = pd.read_csv(raw_path)
    
    # 2. Cleaning Data
    # Ubah TotalCharges jadi angka, handle error jika ada spasi kosong
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Hapus baris NaN (biasanya < 15 baris)
    df_clean = df.dropna().copy()
    
    # Buang CustomerID
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
    
    # 3. Target Encoding (Churn: Yes=1, No=0)
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # 4. Categorical Encoding (One-Hot Encoding)
    # Kita minta int, tapi kita tambahkan langkah paksa di bawahnya
    df_processed = pd.get_dummies(df_clean, drop_first=True, dtype=int)
    
    # --- [LANGKAH TAMBAHAN: PAKSA UBAH TRUE/FALSE JADI 1/0] ---
    # Cari semua kolom yang masih bertipe boolean
    bool_cols = df_processed.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        print(f"⚠️ Mengonversi {len(bool_cols)} kolom Boolean menjadi Angka (0/1)...")
        df_processed[bool_cols] = df_processed[bool_cols].astype(int)
    # -----------------------------------------------------------
    
    # 5. Scaling Fitur Numerik
    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Pastikan kolom numerik ada sebelum scaling
    for col in numeric_cols:
        if col in df_processed.columns:
            # Gunakan .loc untuk menghindari SettingWithCopyWarning
            df_processed.loc[:, col] = scaler.fit_transform(df_processed[[col]])
            
    # 6. Simpan Data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    
    print(f"Preprocessing Selesai!")
    print(f"Ukuran Data Awal: {df.shape}")
    print(f"Ukuran Data Akhir: {df_processed.shape}")
    print(f"File disimpan di: {output_path}")

if __name__ == "__main__":
    RAW_DATA = "../dataset_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv" 
    PROCESSED_DATA = "namadataset_preprocessing/telco_churn_processed.csv"
    
    automate_preprocessing(RAW_DATA, PROCESSED_DATA)