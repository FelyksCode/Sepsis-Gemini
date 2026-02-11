import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Memuat API Key dari .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Definisi System Prompt sesuai spesifikasi terbaru
SYSTEM_PROMPT = """
Anda adalah Asisten AI Medis Spesialis Sepsis (CDSS).
Tugas Anda:
1. Analisis risiko sepsis dari data vital (tren 6 jam) dan keluhan ePRO.
2. Berikan PREDIKSI risiko (0-100%).
3. Berikan ALASAN (Reasoning) berbasis medis (hubungkan data vital & teks).
4. Berikan SOLUSI KLINIS (Recommendations) berdasarkan tingkat risiko:
   - Jika Risiko TINGGI (>80%): Sarankan protokol 'Sepsis Hour-1 Bundle' (e.g., Ambil kultur darah, Pasang IV line, Konsul Dokter segera).
   - Jika Risiko SEDANG (50-80%): Sarankan observasi ketat, cek ulang laktat/vital dalam 1 jam.
   - Jika Risiko RENDAH (<50%): Sarankan monitoring rutin.

Output HARUS dalam format JSON yang valid.
"""

def get_sepsis_model():
    """
    Inisialisasi model Gemini 2.5 Flash dengan instruksi sistem medis.
    """
    # Menggunakan gemini-2.5-flash sesuai instruksi user terbaru
    return genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction=SYSTEM_PROMPT
    )

def analyze_sepsis_risk(row):
    """
    Mengirim prompt multimodal ke Gemini API untuk deteksi sepsis berdasarkan baris data pasien.
    """
    model = get_sepsis_model()

    prompt = f"""
    DATA PASIEN (Window 6 Jam):
    - Heart Rate: Rata-rata {row['HR_Mean']} bpm (Tren: {row['HR_Trend']})
    - Suhu Maksimal: {row['Temp_Max']} C
    - O2 Saturation Min: {row['O2Sat_Min']} %
    - Respirasi Rata-rata: {row['Resp_Mean']} bpm

    LAPORAN PASIEN (ePRO):
    "{row['ePRO_Text']}"

    TUGAS:
    Analisis pasien ini.

    FORMAT JSON OUTPUT:
    {{
        "prediction_score": (integer 0-100),
        "is_sepsis": (true/false),
        "risk_level": "(Low/Medium/High)",
        "reasoning": "(Penjelasan medis singkat max 2 kalimat)",
        "recommendations": [
            "Langkah 1",
            "Langkah 2",
            "Langkah 3"
        ]
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
            "prediction_score": 0,
            "is_sepsis": False,
            "risk_level": "Error",
            "reasoning": f"Gagal memproses API: {str(e)}",
            "recommendations": ["Periksa koneksi sistem"]
        }

if __name__ == "__main__":
    print("="*60)
    print("GEMINI 2.5 FLASH - SEPSIS DETECTION PROMPTER (CDSS)")
    print("="*60)

    # Contoh data input (simulasi satu baris data pasien)
    sample_row = {
        'Patient_ID': 'P001',
        'HR_Mean': 105,
        'HR_Trend': 'Meningkat',
        'Temp_Max': 38.5,
        'O2Sat_Min': 93,
        'Resp_Mean': 22,
        'ePRO_Text': 'Saya merasa sangat menggigil dan jantung berdebar.'
    }

    print("\nMeminta analisis dari Gemini...")
    result = analyze_sepsis_risk(sample_row)

    print("-" * 40)
    if result.get('risk_level') == "Error":
        print(f"STATUS      : ERROR")
        print(f"REASONING   : {result.get('reasoning')}")
    else:
        print(f"RISK LEVEL  : {result.get('risk_level')}")
        print(f"SCORE       : {result.get('prediction_score')}%")
        print(f"IS SEPSIS   : {result.get('is_sepsis')}")
        print(f"REASONING   : {result.get('reasoning')}")
        print(f"RECOMMENDATIONS:")
        for rec in result.get('recommendations', []):
            print(f"  - {rec}")
    print("-" * 40)
