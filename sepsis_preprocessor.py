import pandas as pd
import numpy as np

def generate_dummy_sepsis_csv(filename='sepsis_data.csv'):
    """
    Menghasilkan dataset sepsis dummy dalam format CSV untuk simulasi.
    Dataset ini meniru struktur umum dataset PhysioNet Challenge.
    """
    np.random.seed(42)
    # Simulasi 5 pasien dengan 24 jam observasi masing-masing
    n_patients = 5
    hours = 24
    data = []

    for p_id in range(1, n_patients + 1):
        for h in range(hours):
            # Simulasi nilai dengan beberapa NaN (Missing Values)
            data.append({
                'Patient_ID': f'P{p_id:03d}',
                'Hour': h,
                'HR': np.random.choice([np.nan, np.random.normal(85, 15)], p=[0.2, 0.8]),
                'O2Sat': np.random.choice([np.nan, np.random.normal(97, 2)], p=[0.2, 0.8]),
                'Temp': np.random.choice([np.nan, np.random.normal(37, 1)], p=[0.3, 0.7]),
                'SBP': np.random.choice([np.nan, np.random.normal(110, 20)], p=[0.2, 0.8]),
                'MAP': np.random.choice([np.nan, np.random.normal(80, 15)], p=[0.2, 0.8]),
                'Resp': np.random.choice([np.nan, np.random.normal(18, 4)], p=[0.2, 0.8])
            })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Dataset dummy berhasil dibuat: {filename}")

def inject_epro(row):
    """
    Fungsi untuk menyuntikkan narasi keluhan pasien (Synthetic ePRO)
    berdasarkan kondisi klinis dalam jendela waktu tertentu.
    """
    complaints = []
    # Logika aturan klinis untuk menghasilkan teks
    if row['HR_mean'] > 100:
        complaints.append("Jantung saya terasa berdebar-debar")
    if row['Temp_max'] > 38:
        complaints.append("Saya merasa menggigil dan panas")
    if row['O2Sat_mean'] < 94:
        complaints.append("Napas saya terasa sesak")

    # Jika tidak ada gejala spesifik, berikan status umum
    if not complaints:
        return "Saya merasa baik-baik saja, namun sedikit lemas."

    return ". ".join(complaints) + "."

def preprocess_sepsis_data(filepath):
    # 1. Memuat Dataset
    df = pd.read_csv(filepath)

    # List kolom tanda vital yang akan diproses
    vitals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp']

    # 2. Data Cleaning: Forward Fill (ffill) per Patient_ID
    # Sepsis dataset sering memiliki nilai kosong karena observasi yang tidak kontinu.
    # Kita mengisi nilai kosong dengan nilai terakhir yang diketahui dari pasien tersebut.
    df[vitals] = df.groupby('Patient_ID')[vitals].ffill()

    # 3. Windowing: Buat Jendela Waktu 6 Jam
    # Kita membagi kolom 'Hour' menjadi blok 6 jam (0-5, 6-11, dst.)
    df['Window_ID'] = df['Hour'] // 6

    # 4. Aggregation: Menghitung Statistik per Jendela
    # Kita mengelompokkan berdasarkan Patient_ID dan Window_ID
    grouped = df.groupby(['Patient_ID', 'Window_ID'])

    # Menghitung Rata-rata (Mean) untuk semua tanda vital
    df_mean = grouped[vitals].mean().add_suffix('_mean')

    # Menghitung Max untuk Suhu (penting untuk mendeteksi demam)
    df_max_temp = grouped['Temp'].max().rename('Temp_max')

    # Menghitung Tren (Nilai Akhir - Nilai Awal dalam jendela 6 jam)
    # Ini membantu model memahami apakah kondisi membaik atau memburuk
    df_trend = grouped[vitals].apply(lambda x: x.iloc[-1] - x.iloc[0]).add_suffix('_trend')

    # Menggabungkan hasil agregasi menjadi satu DataFrame
    processed_df = pd.concat([df_mean, df_max_temp, df_trend], axis=1).reset_index()

    # 5. Synthetic ePRO Injection
    # Mengonversi data numerik menjadi narasi teks untuk input multimodal Gemini
    processed_df['epro_feedback'] = processed_df.apply(inject_epro, axis=1)

    return processed_df

if __name__ == "__main__":
    # Langkah 1: Buat data simulasi
    csv_file = 'sepsis_data.csv'
    generate_dummy_sepsis_csv(csv_file)

    # Langkah 2: Lakukan Preprocessing
    print("Memulai proses preprocessing...")
    final_df = preprocess_sepsis_data(csv_file)

    # Langkah 3: Tampilkan Hasil
    print("\n--- HASIL PREPROCESSING DATA MULTIMODAL (6 Jam per Baris) ---")
    # Tampilkan 10 baris pertama secara ringkas
    cols_to_show = ['Patient_ID', 'Window_ID', 'HR_mean', 'Temp_max', 'O2Sat_mean', 'epro_feedback']
    print(final_df[cols_to_show].head(10).to_string(index=False))

    # Simpan hasil untuk review
    final_df.to_csv('processed_multimodal_sepsis.csv', index=False)
    print("\nFile hasil preprocessing disimpan ke: processed_multimodal_sepsis.csv")
