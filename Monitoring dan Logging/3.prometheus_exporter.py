import time
import random
import psutil
from prometheus_client import start_http_server, Gauge, Counter, Histogram

# --- 1. METRIKS SISTEM ---
RAM_USAGE = Gauge('server_ram_usage_percent', 'Persentase penggunaan RAM server')
CPU_USAGE = Gauge('server_cpu_usage_percent', 'Persentase penggunaan CPU server')

# --- 2. METRIKS MODE ---
CHURN_PREDICTIONS = Counter('model_churn_predictions_total', 'Total prediksi churn yang dilakukan')
MODEL_ACCURACY = Gauge('model_accuracy_score', 'Skor akurasi model real-time')
REQUEST_LATENCY = Histogram('model_request_latency_seconds', 'Waktu proses prediksi (detik)')

def generate_metrics():
    print("Prometheus Exporter Berjalan... Menunggu request dari Prometheus.")
    while True:
        # Update CPU & RAM
        RAM_USAGE.set(psutil.virtual_memory().percent)
        CPU_USAGE.set(psutil.cpu_percent())

        # Simulasi Prediksi Churn (Random trigger)
        if random.random() < 0.7: 
            CHURN_PREDICTIONS.inc()
            
            # Simulasi Latency
            with REQUEST_LATENCY.time():
                time.sleep(random.uniform(0.1, 0.5))

        # Simulasi Akurasi (Naik turun di angka 85%)
        acc = 0.85 + random.uniform(-0.05, 0.05)
        MODEL_ACCURACY.set(acc)

        time.sleep(1)

if __name__ == '__main__':
    start_http_server(8000)
    generate_metrics()