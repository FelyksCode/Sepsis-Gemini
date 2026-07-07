from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "f1_model.pkl"
DEFAULT_THRESHOLD_PATH = ROOT_DIR / "best_threshold.json"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "MedAIBase/MedGemma1.5:4b"


FEATURE_GROUPS = [
    (
        "Profil Pasien",
        ["Ageyears", "Sex", "Active_cancer"],
    ),
    (
        "Tanda Vital Awal",
        [
            "SIRS_Upon_presentation_ED",
            "Heart_Rate_Upon_presentation_ED",
            "SBP_Upon_presentation_ED",
            "DBP_Upon_presentation_ED",
            "Respiratory_Rate_Upon_presentation_ED",
            "O2_saturation_Upon_presentation_ED",
            "Temperature_Upon_presentation_ED",
        ],
    ),
    (
        "Laboratorium",
        [
            "Phosphate",
            "WBC",
            "Creatinine",
            "Magnesium",
            "Hemoglobin",
            "Potassium",
            "Calcium",
            "Chloride",
            "BodyUreaNitrogen",
            "Bicarbonate",
            "Glucose",
            "Sodium",
        ],
    ),
    (
        "Riwayat dan Terapi",
        [
            "SystolicCHF_EF_less_than40",
            "Non_SystolicCHF_EF≥40",
            "CerebrovascularaccidentsCVA",
            "COPD_Emphysema",
            "DM",
            "HTN",
            "CAD",
            "IVfluids_requirement_first_24hrs",
            "IVfluids_requirement_first_6hrs",
            "Steroids_Use",
            "Patient_receive_vasopressors_inotropes_withinfirst_24h",
            "Time_to_Antibiotic_treatment_initiation_hours",
        ],
    ),
    (
        "Monitoring 6 Jam",
        [
            "HeartRate_6hours",
            "RespiratoryRate_6hours",
            "DBP_6hours",
            "MAP_6hours",
            "O2saturation_6hours",
            "TemperatureT_6hours",
        ],
    ),
    (
        "Pola Kuman",
        ["Pseudomonas", "E.Coli"],
    ),
]


DEFAULT_VALUES = {
    "SIRS_Upon_presentation_ED": 0,
    "Phosphate": 3.5,
    "WBC": 8000,
    "SystolicCHF_EF_less_than40": 0,
    "Creatinine": 1.0,
    "CerebrovascularaccidentsCVA": 0,
    "IVfluids_requirement_first_24hrs": 0,
    "Steroids_Use": 0,
    "Heart_Rate_Upon_presentation_ED": 84,
    "Magnesium": 2.0,
    "Hemoglobin": 13.0,
    "Non_SystolicCHF_EF≥40": 0,
    "Patient_receive_vasopressors_inotropes_withinfirst_24h": 0,
    "HeartRate_6hours": 82,
    "Pseudomonas": 0,
    "SBP_Upon_presentation_ED": 118,
    "Potassium": 4.0,
    "Calcium": 9.0,
    "Chloride": 103.0,
    "TemperatureT_6hours": 36.8,
    "COPD_Emphysema": 0,
    "BodyUreaNitrogen": 15.0,
    "Respiratory_Rate_Upon_presentation_ED": 18,
    "Active_cancer": 0,
    "O2_saturation_Upon_presentation_ED": 97,
    "Bicarbonate": 24.0,
    "Temperature_Upon_presentation_ED": 36.8,
    "MAP_6hours": 85.0,
    "Glucose": 110.0,
    "DM": 0,
    "Time_to_Antibiotic_treatment_initiation_hours": 2.0,
    "Ageyears": 55.0,
    "O2saturation_6hours": 96,
    "IVfluids_requirement_first_6hrs": 0,
    "HTN": 0,
    "CAD": 0,
    "RespiratoryRate_6hours": 18,
    "DBP_6hours": 75,
    "DBP_Upon_presentation_ED": 76,
    "Sex": 0,
    "Sodium": 140.0,
    "E.Coli": 0,
}


def infer_step(feature_name: str) -> float:
    if feature_name in {
        "Ageyears",
        "Heart_Rate_Upon_presentation_ED",
        "HeartRate_6hours",
        "SBP_Upon_presentation_ED",
        "DBP_Upon_presentation_ED",
        "DBP_6hours",
        "MAP_6hours",
        "Respiratory_Rate_Upon_presentation_ED",
        "RespiratoryRate_6hours",
        "O2_saturation_Upon_presentation_ED",
        "O2saturation_6hours",
    }:
        return 1.0
    if feature_name in {"WBC"}:
        return 100.0
    if feature_name in {"Time_to_Antibiotic_treatment_initiation_hours"}:
        return 0.5
    if feature_name in {
        "Phosphate",
        "Creatinine",
        "Magnesium",
        "Hemoglobin",
        "Potassium",
        "Calcium",
        "Chloride",
        "BodyUreaNitrogen",
        "Bicarbonate",
        "Glucose",
        "TemperatureT_6hours",
        "Temperature_Upon_presentation_ED",
        "Sodium",
    }:
        return 0.1
    return 1.0


def binary_features() -> set[str]:
    return {
        "SIRS_Upon_presentation_ED",
        "SystolicCHF_EF_less_than40",
        "CerebrovascularaccidentsCVA",
        "IVfluids_requirement_first_24hrs",
        "Steroids_Use",
        "Non_SystolicCHF_EF≥40",
        "Patient_receive_vasopressors_inotropes_withinfirst_24h",
        "Pseudomonas",
        "COPD_Emphysema",
        "Active_cancer",
        "Sex",
        "DM",
        "IVfluids_requirement_first_6hrs",
        "HTN",
        "CAD",
        "E.Coli",
    }


def bin_model_feature(feature: str, values: pd.Series) -> pd.Series:
    if feature == "WBC":
        bins = [-np.inf, 4000, 11000, np.inf]
        labels = ["Low", "Normal", "High"]
    elif feature == "Creatinine":
        bins = [-np.inf, 1.2, 2.4, np.inf]
        labels = ["Normal", "Mild", "Severe"]
    elif feature == "Heart_Rate_Upon_presentation_ED":
        bins = [0, 60, 100, np.inf]
        labels = ["Low", "Normal", "High"]
    elif feature == "Hemoglobin":
        bins = [0, 10.0, 13.0, np.inf]
        labels = ["Severe_Anemia", "Mild_Anemia", "Normal"]
    elif feature == "HeartRate_6hours":
        bins = [0, 60, 100, np.inf]
        labels = ["Low", "Normal", "High"]
    elif feature == "BodyUreaNitrogen":
        bins = [0, 20.0, 50.0, np.inf]
        labels = ["Normal", "Moderate", "Severe"]
    elif feature == "Bicarbonate":
        bins = [0, 22.0, 26.0, np.inf]
        labels = ["Low", "Normal", "High"]
    elif feature == "Glucose":
        bins = [-np.inf, 70, 140, np.inf]
        labels = ["Low", "Normal", "High"]
    elif feature == "Time_to_Antibiotic_treatment_initiation_hours":
        bins = [-np.inf, 1.0, 3.0, np.inf]
        labels = ["Immediate_Within_1h", "Delayed_1_3h", "Late_Over_3h"]
    elif feature == "Ageyears":
        bins = [0.0, 40.0, 60.0, np.inf]
        labels = ["Young", "Middle_Aged", "Elderly"]
    else:
        return values

    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True)


def parse_csv_numeric_value(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return float(default)

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip().replace(",", "")
    if not text:
        return float(default)

    if "-" in text:
        parts = [part.strip() for part in text.split("-") if part.strip()]
        if len(parts) == 2:
            low = pd.to_numeric(parts[0], errors="coerce")
            high = pd.to_numeric(parts[1], errors="coerce")
            if pd.notna(low) and pd.notna(high):
                return float((low + high) / 2)

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric):
        return float(numeric)

    return float(default)


def prepare_model_input(sample_df: pd.DataFrame, model_feature_types: list[str], feature_names: list[str]) -> pd.DataFrame:
    prepared = sample_df.copy()
    feature_type_map = dict(zip(feature_names, model_feature_types))

    for feature, feature_type in feature_type_map.items():
        if feature_type == "c":
            prepared[feature] = bin_model_feature(feature, prepared[feature])
        elif feature_type == "int":
            prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce").fillna(0).astype(int)
        else:
            prepared[feature] = pd.to_numeric(prepared[feature], errors="coerce").astype(float)

    return prepared[feature_names]


def build_csv_template(feature_names: list[str]) -> pd.DataFrame:
    template_values = {feature: DEFAULT_VALUES.get(feature, 0.0) for feature in feature_names}
    return pd.DataFrame([template_values], columns=feature_names)


def translate_csv_row_to_form(csv_row: pd.Series, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_names:
        raw_value = csv_row.get(feature, DEFAULT_VALUES.get(feature, 0.0))
        if feature in binary_features():
            translated_value = int(parse_csv_numeric_value(raw_value, DEFAULT_VALUES.get(feature, 0.0)))
        else:
            translated_value = parse_csv_numeric_value(raw_value, DEFAULT_VALUES.get(feature, 0.0))

        rows.append(
            {
                "feature": feature,
                "raw_value": raw_value,
                "form_value": translated_value,
            }
        )

    return pd.DataFrame(rows)


def parse_patient_csv(uploaded_file, feature_names: list[str]) -> tuple[pd.DataFrame | None, str | None]:
    if uploaded_file is None:
        return None, None

    try:
        csv_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, f"Gagal membaca CSV: {exc}"

    if csv_df.empty:
        return None, "CSV kosong."

    missing_columns = [feature for feature in feature_names if feature not in csv_df.columns]
    if missing_columns:
        return None, f"CSV belum memuat kolom berikut: {', '.join(missing_columns)}"

    return csv_df[feature_names].copy(), None


def normalize_csv_row(csv_row: pd.Series, feature_names: list[str]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for feature in feature_names:
        value = csv_row.get(feature, DEFAULT_VALUES.get(feature, 0.0))
        if feature in binary_features():
            normalized[feature] = int(parse_csv_numeric_value(value, DEFAULT_VALUES.get(feature, 0.0)))
        else:
            normalized[feature] = parse_csv_numeric_value(value, DEFAULT_VALUES.get(feature, 0.0))
    return normalized


def apply_form_prefill(prefill_values: dict[str, float], feature_names: list[str]) -> None:
    for feature in feature_names:
        if feature in prefill_values:
            st.session_state[f"field_{feature}"] = prefill_values[feature]


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data
def load_threshold(threshold_path: str) -> float:
    with open(threshold_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload.get("best_threshold", 0.5))


@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


def feature_defaults(feature_names: list[str]) -> dict[str, float]:
    defaults = {}
    for feature in feature_names:
        defaults[feature] = DEFAULT_VALUES.get(feature, 0.0)
    return defaults


def render_numeric_input(feature: str, default_value: float) -> float:
    return st.number_input(
        feature,
        value=float(default_value),
        step=infer_step(feature),
        format="%.3f",
        key=f"field_{feature}",
    )


def render_binary_input(feature: str, default_value: float) -> int:
    options = [0, 1]
    index = 1 if int(default_value) == 1 else 0

    if feature == "Sex":
        return int(
            st.selectbox(
                feature,
                options=options,
                index=index,
                format_func=lambda value: "0 - Female" if value == 0 else "1 - Male",
                key=f"field_{feature}",
            )
        )

    return int(
        st.selectbox(
            feature,
            options=options,
            index=index,
            format_func=lambda value: f"{value}",
            key=f"field_{feature}",
        )
    )


def collect_patient_input(feature_names: list[str], prefill_values: dict[str, float] | None = None) -> dict[str, float]:
    defaults = feature_defaults(feature_names)
    if prefill_values:
        defaults.update(prefill_values)
    values: dict[str, float] = {}

    for group_name, group_features in FEATURE_GROUPS:
        with st.expander(group_name, expanded=group_name in {"Profil Pasien", "Tanda Vital Awal"}):
            cols = st.columns(2)
            for index, feature in enumerate(group_features):
                with cols[index % 2]:
                    if feature in binary_features():
                        values[feature] = render_binary_input(feature, defaults[feature])
                    else:
                        values[feature] = render_numeric_input(feature, defaults[feature])

    ordered_values = {feature: values[feature] for feature in feature_names}
    return ordered_values


def build_explanation_payload(sample_row: pd.Series, probability: float, threshold: float, prediction: int, shap_values: pd.DataFrame) -> dict:
    top_positive = (
        shap_values[shap_values["shap"] > 0]
        .sort_values("shap", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )
    top_negative = (
        shap_values[shap_values["shap"] < 0]
        .sort_values("shap", ascending=True)
        .head(5)
        .to_dict(orient="records")
    )

    return {
        "patient_data": sample_row.to_dict(),
        "prediction": {
            "probability_mortality": round(float(probability), 4),
            "threshold": round(float(threshold), 4),
            "predicted_label": int(prediction),
        },
        "shap_explanation": {
            "top_positive": top_positive,
            "top_negative": top_negative,
        },
    }


def make_shap_table(sample_row: pd.Series, shap_vector: np.ndarray) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "feature": sample_row.index,
            "value": sample_row.values,
            "shap": shap_vector,
        }
    )
    table["abs_shap"] = table["shap"].abs()
    return table.sort_values("abs_shap", ascending=False)


def render_shap_chart(shap_table: pd.DataFrame) -> None:
    chart_df = shap_table.head(12).sort_values("shap")
    colors = ["#2563eb" if value < 0 else "#ef4444" for value in chart_df["shap"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(chart_df["feature"], chart_df["shap"], color=colors)
    ax.axvline(0, color="#1f2937", linewidth=1)
    ax.set_xlabel("SHAP value")
    ax.set_ylabel("Fitur")
    ax.set_title("Kontribusi fitur terhadap prediksi")
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def ollama_generate(
    base_url: str,
    model_name: str,
    prompt: str,
    progress_callback=None,
) -> str:
    start_time = time.perf_counter()

    response = requests.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0,
                "num_ctx": 2048,
            },
        },
        stream=True,
        timeout=180,
    )
    response.raise_for_status()

    chunks: list[str] = []
    chunk_count = 0

    if progress_callback is not None:
        progress_callback(5, "Memulai request ke Ollama", 0.0)

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue

        payload = json.loads(line)
        if payload.get("response"):
            chunks.append(str(payload.get("response", "")))
            chunk_count += 1

            if progress_callback is not None:
                progress_value = min(95, 5 + chunk_count)
                progress_callback(
                    progress_value,
                    f"Menerima chunk respons ke-{chunk_count}",
                    time.perf_counter() - start_time,
                )

        if payload.get("done"):
            break

    if progress_callback is not None:
        progress_callback(100, "Selesai", time.perf_counter() - start_time)

    return "".join(chunks).strip()


def xai_translation_prompt(payload: dict) -> str:
    return f"""
Anda adalah penerjemah output Explainable AI (XAI).

Tugas:
Ubah JSON berikut menjadi ringkasan deskriptif yang netral.

DATA:
{json.dumps(payload, indent=2, ensure_ascii=False)}

ATURAN:
1. Gunakan Bahasa Indonesia formal.
2. Sebutkan probabilitas prediksi model.
3. Sebutkan fitur positif sebagai faktor yang meningkatkan skor prediksi model.
4. Sebutkan fitur negatif sebagai faktor yang menurunkan skor prediksi model.
5. Gunakan HANYA informasi yang tersedia pada JSON.
6. Jangan menambahkan interpretasi klinis.
7. Jangan membuat diagnosis.
8. Jangan membuat hipotesis medis.
9. Jangan menjelaskan hubungan sebab-akibat.
10. Maksimal 100 kata.

FORMAT WAJIB:
Probabilitas Prediksi:
<isi>

Fitur Pendorong (Positif):
- <fitur>: nilai=<value>, kontribusi SHAP=<shap>

Fitur Penurun (Negatif):
- <fitur>: nilai=<value>, kontribusi SHAP=<shap>
""".strip()


def clinical_reasoning_prompt(xai_text: str) -> str:
    return f"""
Anda adalah AI spesialis Informatika Medis.

Berikut adalah hasil translasi Explainable AI (XAI):

{xai_text}

TUGAS:
Analisis kemungkinan alasan model machine learning menghasilkan prediksi berdasarkan pola fitur yang diberikan.

ATURAN:
1. Gunakan bahasa medis formal.
2. Fokus pada pola fitur yang muncul pada penjelasan SHAP.
3. Gunakan hanya informasi yang tersedia.
4. Jangan membuat diagnosis baru.
5. Jangan mengasumsikan informasi klinis yang tidak diberikan.
6. Maksimal 150 kata.

FORMAT:
Analisis Pola Model:
<isi>

Interpretasi Klinis:
<isi>

Disclaimer:
Analisis ini merupakan interpretasi berbasis model AI dan tidak dapat digunakan sebagai diagnosis klinis.
""".strip()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 58%, #0ea5e9 100%);
            color: white;
            padding: 1.4rem 1.6rem;
            border-radius: 1.25rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.1;
        }
        .hero p {
            margin: 0.5rem 0 0;
            opacity: 0.95;
            font-size: 0.98rem;
        }
        .section-card {
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 1rem;
            padding: 1rem 1rem 0.5rem;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
        }
        .small-note {
            color: #475569;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="XAI + Ollama Demo", page_icon="🩺", layout="wide")
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>Demo Machine Learning XAI + Ollama</h1>
            <p>Website sederhana untuk prediksi mortalitas 28 hari dengan XGBoost, SHAP, dan ringkasan bahasa alami via Ollama.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Konfigurasi")
        model_path = st.text_input("Path model", value=str(DEFAULT_MODEL_PATH))
        threshold_path = st.text_input("Path threshold", value=str(DEFAULT_THRESHOLD_PATH))
        ollama_base_url = st.text_input("Ollama base URL", value=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_URL))
        ollama_model = st.text_input("Model Ollama", value=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
        st.caption("Ollama harus aktif di localhost:11434 atau sesuai URL yang Anda isi.")

        st.divider()
        st.subheader("CSV Pasien")
        csv_template = build_csv_template([])
        uploaded_csv = st.file_uploader("Upload CSV pasien", type=["csv"])

        if uploaded_csv is not None:
            st.caption("CSV harus memiliki kolom yang sama dengan fitur model.")

    csv_data = None
    csv_error = None

    model_file = Path(model_path)
    threshold_file = Path(threshold_path)

    if not model_file.exists():
        st.error(f"Model tidak ditemukan: {model_file}")
        st.stop()

    if not threshold_file.exists():
        st.error(f"File threshold tidak ditemukan: {threshold_file}")
        st.stop()

    model = load_model(str(model_file))
    threshold = load_threshold(str(threshold_file))
    explainer = load_explainer(model)

    feature_names = list(model.feature_names_in_)
    model_feature_types = list(model.get_booster().feature_types)

    if uploaded_csv is not None:
        csv_data, csv_error = parse_patient_csv(uploaded_csv, feature_names)

    if csv_data is not None:
        csv_row_options = list(csv_data.index)
        selected_csv_row = st.sidebar.selectbox(
            "Pilih baris CSV",
            options=csv_row_options,
            format_func=lambda idx: f"Baris {idx + 1}",
        )

        preview_row = csv_data.loc[[selected_csv_row]]
        st.sidebar.dataframe(preview_row, use_container_width=True, height=240)

        translated_preview = translate_csv_row_to_form(csv_data.loc[selected_csv_row], feature_names)
        st.sidebar.caption("Translasi ke form input")
        st.sidebar.dataframe(translated_preview, use_container_width=True, height=280)

        if st.sidebar.button("Isi form dari CSV", use_container_width=True):
            st.session_state["csv_prefill_values"] = normalize_csv_row(csv_data.loc[selected_csv_row], feature_names)

        if st.sidebar.button("Kosongkan CSV", use_container_width=True):
            st.session_state.pop("csv_prefill_values", None)

        st.sidebar.download_button(
            "Unduh template CSV",
            data=build_csv_template(feature_names).to_csv(index=False).encode("utf-8"),
            file_name="template_pasien.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if csv_error:
        st.sidebar.error(csv_error)

    csv_prefill_values = st.session_state.get("csv_prefill_values")
    if csv_prefill_values:
        apply_form_prefill(csv_prefill_values, feature_names)

    st.markdown(
        """
        <div class="section-card">
        <p class="small-note">Isi nilai pasien di bawah ini lalu jalankan analisis. Nilai default sudah disiapkan agar demo bisa langsung dipakai.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("patient_form"):
        patient_values = collect_patient_input(feature_names, csv_prefill_values)
        submitted = st.form_submit_button("Jalankan Analisis")

    if submitted:
        sample_df = pd.DataFrame([patient_values], columns=feature_names)
        sample_df_model = prepare_model_input(sample_df, model_feature_types, feature_names)

        probability = float(model.predict_proba(sample_df_model)[0, 1])
        predicted_label = int(probability >= threshold)

        shap_values = explainer(sample_df_model)
        shap_vector = np.asarray(shap_values.values[0], dtype=float)
        shap_table = make_shap_table(sample_df_model.iloc[0], shap_vector)
        explanation_payload = build_explanation_payload(
            sample_df_model.iloc[0], probability, threshold, predicted_label, shap_table
        )

        st.session_state["analysis"] = {
            "sample_df": sample_df,
            "sample_df_model": sample_df_model,
            "probability": probability,
            "predicted_label": predicted_label,
            "shap_table": shap_table,
            "explanation_payload": explanation_payload,
        }
        st.session_state.pop("ollama_result", None)

    analysis = st.session_state.get("analysis")

    if analysis:
        probability = analysis["probability"]
        predicted_label = analysis["predicted_label"]
        shap_table = analysis["shap_table"]
        explanation_payload = analysis["explanation_payload"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Probabilitas Mortalitas", f"{probability:.3f}")
        with col2:
            st.metric("Threshold", f"{threshold:.3f}")
        with col3:
            st.metric("Prediksi", "Mortalitas" if predicted_label == 1 else "Survive")

        if predicted_label == 1:
            st.warning("Model memprediksi mortalitas 28 hari.")
        else:
            st.success("Model memprediksi survival 28 hari.")

        st.progress(min(max(probability, 0.0), 1.0))

        tab1, tab2, tab3 = st.tabs(["Ringkasan", "SHAP", "Ollama"])

        with tab1:
            left, right = st.columns([1.25, 0.75])
            with left:
                st.subheader("Data Pasien")
                display_df = analysis["sample_df"].T.reset_index()
                display_df.columns = ["feature", "value"]
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            with right:
                st.subheader("Payload JSON")
                st.json(explanation_payload)
                st.download_button(
                    "Unduh JSON",
                    data=json.dumps(explanation_payload, ensure_ascii=False, indent=2),
                    file_name="xai_payload.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with tab2:
            st.subheader("Kontribusi SHAP")
            render_shap_chart(shap_table)

            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.write("Fitur yang menaikkan skor prediksi")
                st.dataframe(
                    shap_table[shap_table["shap"] > 0][["feature", "value", "shap"]].head(8),
                    use_container_width=True,
                    hide_index=True,
                )
            with col_neg:
                st.write("Fitur yang menurunkan skor prediksi")
                st.dataframe(
                    shap_table[shap_table["shap"] < 0][["feature", "value", "shap"]].head(8),
                    use_container_width=True,
                    hide_index=True,
                )

        with tab3:
            st.subheader("Generasi ringkasan Ollama")
            st.caption("Gunakan model Ollama lokal untuk menghasilkan translasi XAI dan reasoning klinis.")

            if st.button("Generate Ollama Explanation"):
                try:
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    stopwatch_text = st.empty()

                    def format_elapsed(seconds: float) -> str:
                        minutes = int(seconds // 60)
                        remaining_seconds = seconds - (minutes * 60)
                        if minutes > 0:
                            return f"{minutes:02d}:{remaining_seconds:04.1f}"
                        return f"{remaining_seconds:04.1f}s"

                    def update_progress(percent: int, message: str, elapsed_seconds: float) -> None:
                        progress_bar.progress(max(0, min(100, percent)))
                        progress_text.caption(f"Progress Ollama: {max(0, min(100, percent))}% - {message}")
                        stopwatch_text.caption(f"Stopwatch Ollama: {format_elapsed(elapsed_seconds)}")

                    xai_text = ollama_generate(
                        ollama_base_url,
                        ollama_model,
                        xai_translation_prompt(explanation_payload),
                        progress_callback=update_progress,
                    )
                    clinical_text = ollama_generate(
                        ollama_base_url,
                        ollama_model,
                        clinical_reasoning_prompt(xai_text),
                        progress_callback=update_progress,
                    )

                    st.session_state["ollama_result"] = {
                        "xai_text": xai_text,
                        "clinical_text": clinical_text,
                    }
                except Exception as exc:
                    if "progress_bar" in locals():
                        progress_bar.progress(0)
                    if "progress_text" in locals():
                        progress_text.caption("Progress Ollama: gagal")
                    if "stopwatch_text" in locals():
                        stopwatch_text.caption("Stopwatch Ollama: gagal")
                    st.session_state["ollama_result"] = {
                        "xai_text": f"Gagal memanggil Ollama: {exc}",
                        "clinical_text": "",
                    }

            ollama_result = st.session_state.get("ollama_result")
            if ollama_result:
                st.markdown("### Translasi XAI")
                st.write(ollama_result.get("xai_text", ""))
                st.markdown("### Clinical Reasoning")
                st.write(ollama_result.get("clinical_text", ""))

    else:
        st.info("Klik 'Jalankan Analisis' untuk melihat prediksi, SHAP, dan opsi ringkasan Ollama.")


if __name__ == "__main__":
    main()