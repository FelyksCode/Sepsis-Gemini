# XAI + Ollama Demo

Website sederhana berbasis Streamlit untuk mendemokan model XGBoost dari notebook `death_prediction.ipynb`, lengkap dengan SHAP dan ringkasan teks via Ollama.

## Isi

- Prediksi mortalitas 28 hari dengan threshold dari `best_threshold.json`
- Visualisasi kontribusi fitur dengan SHAP
- Translasi XAI dan clinical reasoning dari model Ollama lokal

## Cara Menjalankan

1. Install dependensi:

```bash
pip install -r website_xai_ollama_demo/requirements.txt
```

2. Pastikan file model ada di root workspace:

- `f1_model.pkl`
- `best_threshold.json`

3. Jalankan Streamlit:

```bash
streamlit run website_xai_ollama_demo/app.py
```

## Ollama

App ini memakai endpoint lokal Ollama di `http://localhost:11434` secara default.

Jika ingin mengubahnya, isi sidebar:

- `Ollama base URL`
- `Model Ollama`

Model default yang disiapkan:

- `MedAIBase/MedGemma1.5:4b`

## Catatan

- Input form mengikuti urutan fitur yang dipakai model pickle.
- Jika Anda memindahkan file model, sesuaikan path di sidebar.