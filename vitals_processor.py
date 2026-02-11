import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dummy_data(filename='dummy_sepsis_data.csv'):
    """
    Menghasilkan dataset sepsis dummy dengan awalan 'dummy_' untuk simulasi.
    Dataset ini mencakup tanda-tanda vital yang biasa didapat dari sensor wearable.
    """
    np.random.seed(42)
    n_patients = 3
    hours = 24
    data = []

    for p_id in range(1, n_patients + 1):
        for h in range(hours):
            data.append({
                'Patient_ID': f'P{p_id:03d}',
                'Hour': h,
                'HR': np.random.choice([np.nan, np.random.normal(90, 15)], p=[0.1, 0.9]),
                'O2Sat': np.random.choice([np.nan, np.random.normal(96, 2)], p=[0.1, 0.9]),
                'Temp': np.random.choice([np.nan, np.random.normal(37, 1)], p=[0.2, 0.8]),
                'Resp': np.random.choice([np.nan, np.random.normal(20, 4)], p=[0.1, 0.9]),
                'SepsisLabel': 1 if (p_id == 1 and h > 15) else 0 # Simulasi ground truth
            })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Dataset dummy berhasil dibuat: {filename}")
    return filename

def inject_epro_feedback(row):
    """
    Menghasilkan narasi keluhan pasien berdasarkan kondisi klinis dalam jendela waktu.
    """
    complaints = []
    if row['HR_mean'] > 100:
        complaints.append("Jantung berdebar")
    if row['Temp_max'] > 38:
        complaints.append("Merasa menggigil")
    if row['O2Sat_mean'] < 94:
        complaints.append("Sesak napas")

    return ". ".join(complaints) + "." if complaints else "Saya merasa baik-baik saja."

def calculate_slope(series):
    """
    Menghitung tren (slope) menggunakan regresi linear.
    """
    y = series.dropna().values
    if len(y) < 2: return 0
    x = np.arange(len(y))
    slope, _ = np.polyfit(x, y, 1)
    return slope

def process_vitals_to_window(filepath):
    """
    Melakukan preprocessing: cleaning, windowing 6 jam, dan agregasi statistik.
    """
    df = pd.read_csv(filepath)
    vitals = ['HR', 'O2Sat', 'Temp', 'Resp']

    # 1. Cleaning: Forward Fill per Patient_ID
    df[vitals] = df.groupby('Patient_ID')[vitals].ffill()

    # 2. Windowing: Blok 6 Jam
    df['Window_ID'] = df['Hour'] // 6

    # 3. Aggregation
    grouped = df.groupby(['Patient_ID', 'Window_ID'])

    # Hitung Mean, Trend, dan Max
    df_mean = grouped[vitals].mean().add_suffix('_mean')
    # Gunakan transform/agg untuk memastikan setiap kolom memiliki trennya sendiri
    df_trend = grouped[vitals].agg(calculate_slope).add_suffix('_trend')
    df_max = grouped['Temp'].max().rename('Temp_max')

    # Ambil SepsisLabel (Ground Truth) terakhir dalam jendela tersebut
    df_label = grouped['SepsisLabel'].last()

    # Gabungkan hasil
    processed_df = pd.concat([df_mean, df_trend, df_max, df_label], axis=1).reset_index()

    # 4. Multimodal: Inject ePRO Feedback
    processed_df['epro_feedback'] = processed_df.apply(inject_epro_feedback, axis=1)

    return processed_df

if __name__ == "__main__":
    print("="*60)
    print("CORE PRE-PROCESSOR: VITAL DATA TO WINDOW")
    print("="*60)

    # Jalankan simulasi
    raw_csv = generate_dummy_data()
    final_df = process_vitals_to_window(raw_csv)

    # Simpan output dengan awalan 'dummy_'
    output_filename = 'dummy_processed_vitals.csv'
    final_df.to_csv(output_filename, index=False)

    print(f"\nHasil preprocessing (3 baris pertama):\n")
    print(final_df[['Patient_ID', 'Window_ID', 'HR_mean', 'epro_feedback']].head(3).to_string(index=False))
    print(f"\nFile output disimpan ke: {output_filename}")
