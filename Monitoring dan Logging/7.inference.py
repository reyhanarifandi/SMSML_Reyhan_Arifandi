import pandas as pd
import numpy as np

# Ini adalah simulasi file inference model

def predict(data):

    print("Memproses data:", data)
    prediction = np.random.choice([0, 1])
    return prediction

if __name__ == "__main__":
    sample_data = {"tenure": 12, "MonthlyCharges": 70.5}
    print("Hasil Prediksi:", predict(sample_data))