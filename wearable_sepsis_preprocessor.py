import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dummy_physionet_data(filename='physionet_dummy.csv'):
    """
    Menghasilkan dataset dummy yang meniru format PhysioNet Sepsis Challenge.
    Termasuk data vital sign dan data laboratorium untuk menunjukkan proses filtering.
    """
    np.random.seed(42)
    n_patients = 5
    hours_per_patient = 12
    data = []

    for p_id in range(1, n_patients + 1):
        # SepsisLabel biasanya tetap atau berubah jadi 1 di jam tertentu
        sepsis_status = np.random.choice([0, 1])

        for h in range(hours_per_patient):
            data.append({
                'Patient_ID': f'USER_{p_id:03d}',
                'Hour': h,
                # Vital Signs (Bisa didapat dari Smartwatch)
                'HR': np.random.choice([np.nan, np.random.normal(80, 10)], p=[0.1, 0.9]),
                'O2Sat': np.random.choice([np.nan, np.random.normal(98, 1)], p=[0.1, 0.9]),
                'Temp': np.random.choice([np.nan, np.random.normal(36.5, 0.5)], p=[0.2, 0.8]),
                'Resp': np.random.choice([np.nan, np.random.normal(16, 2)], p=[0.1, 0.9]),
                # Lab Data (TIDAK bisa didapat dari Smartwatch)
                'WBC': np.random.normal(10, 2),
                'Bilirubin': np.random.normal(1, 0.2),
                'Creatinine': np.random.normal(1, 0.1),
                'SepsisLabel': sepsis_status if h > 6 else 0
            })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Dataset dummy PhysioNet dibuat: {filename}")
    return filename

def generate_epro(hr_mean, temp_mean, o2sat_mean):
    """
    Menghasilkan narasi keluhan pasien (ePRO) berdasarkan ambang batas klinis
    yang ditentukan untuk simulasi data teks dari pengguna.
    """
    if hr_mean > 100 and temp_mean > 37.5:
        return "Saya merasa jantung berdebar dan badan panas menggigil."
    elif o2sat_mean < 94:
        return "Napas saya terasa agak berat."
    else:
        return "Saya merasa baik-baik saja."

def calculate_slope(series):
    """
    Menghitung tren (slope) menggunakan regresi linear sederhana.
    """
    if len(series.dropna()) < 2:
        return 0
    y = series.dropna().values
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope

def preprocess_wearable_data(filepath):
    print("\n--- Memulai Preprocessing Data Wearable ---")
    df = pd.read_csv(filepath)

    # 1. Filter Fitur Smartwatch
    # Penjelasan: Data laboratorium (WBC, Bilirubin, dll) dibuang karena smartwatch
    # hanya mendukung sensor non-invasif. Kita hanya fokus pada data yang bisa
    # didapat secara real-time dari wearable.
    wearable_features = ['Patient_ID', 'Hour', 'HR', 'O2Sat', 'Temp', 'Resp', 'SepsisLabel']
    df = df[wearable_features]
    print(f"Fitur difilter. Sisa kolom: {list(df.columns)}")

    # 2. Handling Missing Values (Forward Fill per Patient_ID)
    # Smartwatch sering kehilangan sinyal (intermittent loss), ffill digunakan untuk
    # menjaga kontinuitas data medis.
    df[['HR', 'O2Sat', 'Temp', 'Resp']] = df.groupby('Patient_ID')[['HR', 'O2Sat', 'Temp', 'Resp']].ffill()

    # 3. Windowing (6 Jam terakhir per pasien)
    # Kita mengambil jendela 6 jam terakhir untuk setiap pasien untuk flattening
    latest_windows = df.groupby('Patient_ID').tail(6)

    # 4. Aggregation & Flattening
    # Menghitung Mean dan Trend (Slope)
    final_data = []
    for p_id, group in latest_windows.groupby('Patient_ID'):
        # Hitung Mean
        hr_mean = group['HR'].mean()
        o2_mean = group['O2Sat'].mean()
        temp_mean = group['Temp'].mean()
        resp_mean = group['Resp'].mean()

        # Hitung Trend
        hr_trend = calculate_slope(group['HR'])
        o2_trend = calculate_slope(group['O2Sat'])
        temp_trend = calculate_slope(group['Temp'])

        # Simulasi ePRO
        epro_text = generate_epro(hr_mean, temp_mean, o2_mean)

        # Ambil Label Sepsis Terakhir
        label = group['SepsisLabel'].iloc[-1]

        final_data.append({
            'Patient_ID': p_id,
            'HR_mean': round(hr_mean, 2),
            'HR_trend': round(hr_trend, 2),
            'O2Sat_mean': round(o2_mean, 2),
            'O2Sat_trend': round(o2_trend, 2),
            'Temp_mean': round(temp_mean, 2),
            'Temp_trend': round(temp_trend, 2),
            'Resp_mean': round(resp_mean, 2),
            'epro_text': epro_text,
            'SepsisLabel': label
        })

    return pd.DataFrame(final_data)

if __name__ == "__main__":
    # Setup
    csv_file = generate_dummy_physionet_data()

    # Process
    processed_df = preprocess_wearable_data(csv_file)

    # Output
    print("\n--- DATA FINAL PER PASIEN (Siap untuk Gemini/LLM) ---")
    print(processed_df.to_string(index=False))

    # Penjelasan Tambahan untuk Skripsi
    print("\n[DOKUMENTASI RISET]")
    print("Mengapa data Lab dibuang?")
    print("- Simulasi Keterbatasan Wearable: Smartwatch tidak memiliki sensor invasif untuk mengecek darah.")
    print("- Real-time Monitoring: Fokus pada deteksi dini yang bersifat mobile dan kontinu.")
    print("- Efisiensi LLM: Mengurangi noise dari fitur yang tidak relevan dengan konteks sensor wearable.")
