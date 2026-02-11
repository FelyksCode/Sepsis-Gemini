import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Memuat API Key dari .env
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_sepsis_model():
    """
    Inisialisasi model Gemini 2.0 Flash dengan instruksi sistem medis.
    """
    system_instruction = """
    ROLE:
    Senior Clinical AI Researcher specializing in Oncology and Sepsis Early Warning Systems.

    OBJECTIVE:
    Analyze multimodal data (Numerical Vital Signs & Patient-Reported Outcomes) to predict the risk of sepsis.
    Provide a "Risk Score" and clinical reasoning.

    KNOWLEDGE BASE:
    1. SIRS Criteria (Temp >38C or <36C, HR >90, RR >20).
    2. qSOFA Criteria (Altered mental status, Systolic BP <=100, RR >=22).

    OUTPUT SPECIFICATION:
    ALWAYS respond in valid JSON format.
    """

    return genai.GenerativeModel(
        'gemini-2.0-flash',
        system_instruction=system_instruction
    )

def analyze_sepsis_risk(vital_summary, epro_text):
    """
    Mengirim prompt multimodal ke Gemini API untuk deteksi sepsis.
    """
    model = get_sepsis_model()

    prompt = f"""
    Sepsis Risk Analysis Request:
    -----------------------------
    [INPUT DATA]
    - Vital Summary: {vital_summary}
    - Patient ePRO: "{epro_text}"

    [REQUIRED JSON STRUCTURE]
    {{
      "risk_score": float,
      "risk_level": "Low/Medium/High",
      "clinical_indicators": ["list findings"],
      "reasoning": "Detailed explanation using medical terminology",
      "is_emergency": boolean
    }}
    """

    try:
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text.strip())
    except Exception as e:
        return {"error": str(e), "risk_level": "Unknown", "is_emergency": False}

if __name__ == "__main__":
    print("="*60)
    print("GEMINI 2.0 FLASH - SEPSIS DETECTION PROMPTER")
    print("="*60)

    # Contoh data input
    sample_vitals = "HR Mean: 105, Temp Max: 38.5, O2Sat Mean: 93%"
    sample_epro = "Saya merasa sangat menggigil dan jantung berdebar."

    print("\nMeminta analisis dari Gemini...")
    result = analyze_sepsis_risk(sample_vitals, sample_epro)

    print("-" * 40)
    if "error" in result:
        print(f"ERROR: {result['error']}")
    else:
        print(f"RISK LEVEL  : {result.get('risk_level')}")
        print(f"SCORE       : {result.get('risk_score')}")
        print(f"EMERGENCY   : {result.get('is_emergency')}")
        print(f"REASONING   : {result.get('reasoning')}")
    print("-" * 40)
