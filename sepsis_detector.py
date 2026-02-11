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
    Fungsi untuk menggabungkan data vital dan narasi pasien menjadi prompt klinis
    sesuai dengan template yang telah ditentukan.
    """
    # Menentukan trend sederhana berdasarkan data dummy (bisa dikembangkan lebih lanjut)
    hr_trend = "Increasing" if vitals_row['hr'] > 100 else "Stable"
    spo2_trend = "Decreasing" if vitals_row['spo2'] < 95 else "Stable"

    prompt = f"""
    Sepsis Risk Analysis Request:
    -----------------------------
    [INPUT DATA]
    - Time-Series Trends (Last 6 Hours):
      * Heart Rate: {hr_trend} (Avg: {vitals_row['hr']} bpm)
      * SpO2: {spo2_trend} (Min: {vitals_row['spo2']}%)
      * Temperature: {vitals_row['temp']}°C
      * Respiratory Rate: {vitals_row['rr']} breaths/min

    - Patient-Reported Outcomes (ePRO):
      * Subjective Complaints: "{epro_text}"
      * Reported At: 10:00 AM (Fixed Dummy Time)

    [TASK]
    1. Correlate the vital sign trends with the ePRO narrative.
    2. Evaluate if the subjective "shivering" or "confusion" matches the physiological data.
    3. Provide a risk score (0-100).

    [REQUIRED JSON STRUCTURE]
    {{
      "risk_score": float,
      "risk_level": "Low/Medium/High",
      "clinical_indicators": ["list findings"],
      "reasoning": "Detailed explanation using medical terminology",
      "is_emergency": boolean
    }}
    """
    return prompt

def detect_sepsis(prompt):
    """
    Fungsi untuk memanggil Gemini API dan mendapatkan hasil deteksi sepsis.
    Menggunakan model gemini-2.0-flash untuk kapabilitas terbaru.
    """
    try:
        # Instruksi Sistem (System Instruction) untuk Gemini
        system_instruction = """
        ROLE:
        Senior Clinical AI Researcher specializing in Oncology and Sepsis Early Warning Systems.

        OBJECTIVE:
        Analyze multimodal data (Numerical Vital Signs & Patient-Reported Outcomes) to predict the risk of sepsis. Your goal is to provide a "Risk Score" and clinical reasoning that bridges the gap between raw sensor data and subjective patient feelings.

        KNOWLEDGE BASE:
        1. SIRS Criteria (Temp >38C or <36C, HR >90, RR >20).
        2. qSOFA Criteria (Altered mental status, Systolic BP <=100, RR >=22).
        3. Oncology Context: Distinguish between neutropenic fever and septic shock.

        OUTPUT SPECIFICATION:
        You must ALWAYS respond in valid JSON format so the Python application can parse your analysis.
        """

        # Inisialisasi model dengan instruksi sistem
        # Menggunakan gemini-2.0-flash sesuai instruksi user terbaru
        model = genai.GenerativeModel(
            'gemini-2.0-flash',
            system_instruction=system_instruction
        )

        # Memanggil API Gemini
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        # Mengambil teks respon
        result_text = response.text.strip()

        # Parsing JSON
        result_json = json.loads(result_text)
        return result_json

    except Exception as e:
        # Penanganan error (Quota, API Failure, dll)
        error_msg = str(e)
        if "quota" in error_msg.lower():
            error_msg = "API Quota exceeded. Please try again later."
        elif "api key" in error_msg.lower():
            error_msg = "Invalid API Key."

        return {
            "error": error_msg,
            "risk_score": 0,
            "risk_level": "Unknown",
            "clinical_indicators": [],
            "reasoning": f"Gagal mendapatkan respons dari API: {error_msg}",
            "is_emergency": False
        }

def analyze_sepsis(vital_data, epro_text):
    """
    Fungsi multimodal untuk menyusun prompt dan memanggil deteksi.
    """
    # Menentukan trend sederhana
    hr_avg = vital_data.get('hr_avg', 0)
    hr_trend = vital_data.get('hr_trend', 'stable')
    temp = vital_data.get('temp', 0)

    # Membuat prompt (menggunakan template yang diminta)
    prompt = f"""
    Sepsis Risk Analysis Request:
    -----------------------------
    [INPUT DATA]
    - Vital Data:
      * Heart Rate Avg: {hr_avg} bpm (Trend: {hr_trend})
      * Temperature: {temp}°C

    - Patient-Reported Outcomes (ePRO):
      * Subjective Complaints: "{epro_text}"

    [TASK]
    Analyze the risk of sepsis based on the provided multimodal data.

    [REQUIRED JSON STRUCTURE]
    {{
      "risk_score": float,
      "risk_level": "Low/Medium/High",
      "clinical_indicators": ["list findings"],
      "reasoning": "Detailed explanation using medical terminology",
      "is_emergency": boolean
    }}
    """
    return detect_sepsis(prompt)

if __name__ == "__main__":
    print("="*60)
    print("SISTEM DETEKSI DINI SEPSIS - GEMINI 2.0 FLASH")
    print("="*60)

    # Contoh penggunaan fungsi analyze_sepsis sesuai instruksi terbaru
    sample_vitals = {'hr_avg': 105, 'hr_trend': 'upward', 'temp': 38.5}
    sample_epro = "Saya merasa sangat menggigil"

    print("\nMenganalisis data pasien...")
    result = analyze_sepsis(sample_vitals, sample_epro)

    # Menampilkan hasil
    print("-" * 40)
    if "error" in result and result["risk_level"] == "Unknown":
        print(f"STATUS: ERROR")
        print(f"Pesan: {result['error']}")
    else:
        print(f"LEVEL RISIKO   : {result.get('risk_level')}")
        print(f"SKOR RISIKO    : {result.get('risk_score')}")
        print(f"EMERGENCY      : {result.get('is_emergency')}")
        print(f"INDIKATOR      : {', '.join(result.get('clinical_indicators', []))}")
        print(f"REASONING      : {result.get('reasoning')}")
    print("-" * 40)

    print("\nAnalisis selesai.")
