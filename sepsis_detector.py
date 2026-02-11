import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Memuat variabel lingkungan dari file .env
load_dotenv()

# Konfigurasi Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Dummy Dataset untuk Pasien Onkologi
# Pasien Onkologi seringkali memiliki risiko sepsis yang lebih tinggi karena imunosupresi
dummy_data = [
    {
        "patient_id": "P001",
        "hr": 110,      # Takikardia
        "spo2": 92,     # Hipoksia ringan
        "temp": 38.5,   # Demam
        "rr": 24,       # Takipnea
        "epro": "Saya merasa sangat menggigil, pusing, dan badan saya terasa lemas sekali sejak tadi pagi."
    },
    {
        "patient_id": "P002",
        "hr": 75,
        "spo2": 98,
        "temp": 36.6,
        "rr": 16,
        "epro": "Saya merasa baik hari ini, hanya sedikit nafsu makan berkurang."
    }
]

def preprocess_data(vitals_list):
    """
    Fungsi untuk melakukan pra-pemrosesan data vital pasien.
    Mengonversi list of dictionary menjadi DataFrame dan menormalisasi input angka.
    """
    df = pd.DataFrame(vitals_list)

    # Kolom numerik yang akan dinormalisasi
    numeric_cols = ['hr', 'spo2', 'temp', 'rr']

    # Normalisasi sederhana (Min-Max Scaling) untuk membantu konsistensi data
    # Dalam skenario klinis nyata, normalisasi bisa menggunakan referensi nilai normal (Z-score)
    for col in numeric_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val != min_val:
            df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[f'{col}_normalized'] = 0.5 # Default jika hanya ada satu nilai atau nilai sama

    return df

def generate_clinical_prompt(vitals_row, epro_text):
    """
    Fungsi untuk menggabungkan data vital dan narasi pasien menjadi prompt klinis.
    Menyertakan kriteria SIRS dan qSOFA sebagai referensi medis untuk Gemini.
    """
    prompt = f"""
    Anda adalah seorang AI Researcher di bidang Healthcare yang ahli dalam deteksi dini sepsis.
    Analisis data pasien onkologi berikut untuk mendeteksi risiko sepsis.

    DATA VITAL PASIEN:
    - Heart Rate (HR): {vitals_row['hr']} bpm (Normalized: {vitals_row['hr_normalized']:.2f})
    - SpO2: {vitals_row['spo2']}% (Normalized: {vitals_row['spo2_normalized']:.2f})
    - Temperature: {vitals_row['temp']}°C (Normalized: {vitals_row['temp_normalized']:.2f})
    - Respiratory Rate (RR): {vitals_row['rr']} bpm (Normalized: {vitals_row['rr_normalized']:.2f})

    DATA ePRO (Keluhan Pasien):
    "{epro_text}"

    REFERENSI KRITERIA KLINIS:
    1. SIRS (Systemic Inflammatory Response Syndrome):
       - HR > 90 bpm
       - Temp > 38°C atau < 36°C
       - RR > 20 bpm
    2. qSOFA (Quick SOFA):
       - RR >= 22 bpm
       - Perubahan status mental (analisis dari data ePRO)
       - Tekanan darah sistolik <= 100 mmHg (jika tidak ada data, asumsikan berdasarkan keluhan lain)

    TUGAS ANDA:
    Lakukan analisis mendalam terhadap data tersebut. Pasien onkologi memiliki risiko tinggi karena kondisi imunokompromais.
    Berikan output dalam format JSON yang valid dengan kunci sebagai berikut:
    - status_sepsis: boolean (True jika risiko tinggi/ada indikasi sepsis, False jika tidak)
    - skor_risiko: integer (skala 0-100)
    - rasionalisasi_medis: string (penjelasan singkat mengapa Anda mengambil keputusan tersebut berdasarkan kriteria klinis)

    PENTING: Hanya berikan output dalam format JSON.
    """
    return prompt

def detect_sepsis(prompt):
    """
    Fungsi untuk memanggil Gemini API dan mendapatkan hasil deteksi sepsis.
    Menggunakan model gemini-1.5-flash untuk efisiensi.
    """
    try:
        # Inisialisasi model
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Memanggil API Gemini
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        # Mengambil teks respon dan membersihkannya jika perlu (biasanya sudah bersih dengan response_mime_type)
        result_text = response.text.strip()

        # Parsing JSON
        result_json = json.loads(result_text)
        return result_json

    except Exception as e:
        return {
            "error": str(e),
            "status_sepsis": None,
            "skor_risiko": 0,
            "rasionalisasi_medis": "Gagal mendapatkan respons dari API."
        }

if __name__ == "__main__":
    print("="*60)
    print("SISTEM DETEKSI DINI SEPSIS - PASIEN ONKOLOGI (Gemini AI)")
    print("="*60)

    # 1. Pra-pemrosesan Data
    df_vitals = preprocess_data(dummy_data)

    # 2. Iterasi setiap pasien untuk deteksi
    for index, row in df_vitals.iterrows():
        print(f"\nMenganalisis Pasien ID: {row['patient_id']}...")

        # Generasi Prompt
        prompt = generate_clinical_prompt(row, row['epro'])

        # Deteksi Sepsis menggunakan Gemini
        result = detect_sepsis(prompt)

        # 3. Menampilkan Output secara Rapi
        print("-" * 40)
        if "error" in result and result["status_sepsis"] is None:
            print(f"STATUS: ERROR")
            print(f"Pesan: {result['error']}")
        else:
            status_str = "RISIKO SEPSIS TERDETEKSI" if result.get('status_sepsis') else "RISIKO RENDAH"
            print(f"STATUS SEPSIS     : {status_str}")
            print(f"SKOR RISIKO (0-100): {result.get('skor_risiko')}")
            print(f"RASIONALISASI MEDIS: {result.get('rasionalisasi_medis')}")
        print("-" * 40)

    print("\nAnalisis selesai.")
