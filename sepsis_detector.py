import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Memuat API Key dari .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_sepsis_model():
    """
    Inisialisasi model Gemini 2.5 Flash dengan instruksi sistem medis ICU.
    """
    # Menggunakan gemini-2.5-flash sesuai instruksi user terbaru
    return genai.GenerativeModel('gemini-2.5-flash')

def analyze_sepsis_risk(row):
    """
    Mengirim prompt multimodal ke Gemini API untuk deteksi sepsis berdasarkan baris data pasien.
    Menggunakan konteks ICU dan Lama Rawat (LOS).
    """
    model = get_sepsis_model()

    prompt = f"""
    ROLE: Anda adalah Dokter Spesialis ICU (Intensive Care Unit).

    KONTEKS WAKTU PASIEN:
    - Total Lama Perawatan (LOS): {row['Hour_Last']} Jam.
    - Status: Ini adalah perawatan HARI KE-{row['Day_Of_Stay']}.

    DATA OBSERVASI (Jendela 6 Jam Terakhir):
    1. Tren Heart Rate: {row['HR_Delta']:+.1f} bpm (Dari {row['HR_First']} menjadi {row['HR_Last']}).
    2. Suhu Tubuh Saat Ini: {row['Temp_Last']}°C.
    3. Keluhan Pasien (ePRO): "{row['ePRO_Text']}"

    TUGAS ANALISIS:
    Apakah pasien ini menunjukkan tanda-tanda awal Sepsis?
    Perhatikan bahwa pasien yang dirawat lebih dari 48 jam (Hari ke-3+) memiliki risiko tinggi Infeksi Nosokomial jika terjadi perubahan tanda vital mendadak.

    OUTPUT (JSON):
    {{
      "risk_level": "Low/Medium/High",
      "reasoning": "Jelaskan alasan medis dengan mengaitkan lama perawatan dan gejala terkini.",
      "recommendation": "Tindakan medis singkat."
    }}
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"Error pada Pasien {row.get('Patient_ID', 'Unknown')}: {e}")
        return {
            "risk_level": "Error",
            "reasoning": f"Gagal memproses API: {str(e)}",
            "recommendation": "Periksa koneksi sistem"
        }

if __name__ == "__main__":
    print("="*60)
    print("ICU SPECIALIST - SEPSIS DETECTION PROMPTER")
    print("="*60)

    # Contoh data input (simulasi satu baris data pasien Hour 50)
    sample_row = {
        'Patient_ID': 'P001',
        'Hour_Last': 50,
        'Day_Of_Stay': 3,
        'HR_First': 85,
        'HR_Last': 110,
        'HR_Delta': 25.0,
        'Temp_Last': 38.9,
        'ePRO_Text': 'Saya merasa menggigil hebat.'
    }

    print("\nMeminta analisis dari Gemini...")
    result = analyze_sepsis_risk(sample_row)

    print("-" * 40)
    if result.get('risk_level') == "Error":
        print(f"STATUS          : ERROR")
        print(f"REASONING       : {result.get('reasoning')}")
    else:
        print(f"RISK LEVEL      : {result.get('risk_level')}")
        print(f"REASONING       : {result.get('reasoning')}")
        print(f"RECOMMENDATION  : {result.get('recommendation')}")
    print("-" * 40)
