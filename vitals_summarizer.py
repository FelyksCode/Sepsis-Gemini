import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dummy_vitals(n_points=100):
    """
    Menghasilkan data dummy tanda-tanda vital untuk simulasi.
    Data mencakup timestamp, heart_rate, spo2, dan temperature.
    """
    now = datetime.now()
    timestamps = [now - timedelta(minutes=5*i) for i in range(n_points)]

    # Membuat tren meningkat untuk HR dan Suhu (simulasi sepsis)
    # HR: rata-rata 80, tren naik + noise
    hr = [80 + 0.5*i + np.random.normal(0, 2) for i in range(n_points)]
    # SpO2: rata-rata 98, tren turun sedikit + noise
    spo2 = [98 - 0.05*i + np.random.normal(0, 0.5) for i in range(n_points)]
    # Temp: rata-rata 36.5, tren naik + noise
    temp = [36.5 + 0.02*i + np.random.normal(0, 0.1) for i in range(n_points)]

    df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': hr,
        'spo2': spo2,
        'temperature': temp
    })

    # Urutkan berdasarkan waktu
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def calculate_trend(series):
    """
    Menghitung tren menggunakan regresi linear sederhana (Slope).
    Logika Matematis:
    y = mx + c
    di mana m adalah slope. Jika m > ambang batas positif (0.01), tren 'Meningkat'.
    Jika m < ambang batas negatif (-0.01), tren 'Menurun'.
    Selain itu, dianggap 'Stabil'.
    """
    if len(series) < 2:
        return "Stabil"

    x = np.arange(len(series))
    y = series.values

    # Fit linear regression (polyfit degree 1)
    slope, _ = np.polyfit(x, y, 1)

    threshold = 0.01 # Ambang batas perubahan

    if slope > threshold:
        return "Meningkat"
    elif slope < -threshold:
        return "Menurun"
    else:
        return "Stabil"

def summarize_vitals(df, n_hours=6):
    """
    Meringkas data vital sign dalam jendela waktu n_hours terakhir.
    """
    # Filter data berdasarkan jendela waktu
    cutoff_time = datetime.now() - timedelta(hours=n_hours)
    window_df = df[df['timestamp'] >= cutoff_time].copy()

    if window_df.empty:
        return "Data tidak tersedia dalam jendela waktu tersebut.", {}

    summary_dict = {}
    narrative_parts = []

    parameters = {
        'heart_rate': 'Detak Jantung',
        'spo2': 'SpO2',
        'temperature': 'Suhu'
    }

    for col, label in parameters.items():
        mean_val = window_df[col].mean()
        min_val = window_df[col].min()
        max_val = window_df[col].max()
        trend = calculate_trend(window_df[col])

        summary_dict[col] = {
            'mean': round(mean_val, 2),
            'min': round(min_val, 2),
            'max': round(max_val, 2),
            'trend': trend
        }

        unit = "bpm" if col == 'heart_rate' else "%" if col == 'spo2' else "°C"
        narrative_parts.append(f"{label} rata-rata {round(mean_val, 2)}{unit} dengan tren {trend} (Min: {round(min_val, 2)}, Max: {round(max_val, 2)})")

    narrative_string = ", ".join(narrative_parts) + "."

    return narrative_string, summary_dict

if __name__ == "__main__":
    print("="*60)
    print("VITALS SUMMARIZER - PRE-PROCESSING UNTUK LLM")
    print("="*60)

    # 1. Generate Dummy Data (simulasi 8 jam data)
    data = generate_dummy_vitals(n_points=96) # 96 points * 5 min = 480 min = 8 jam

    # 2. Ringkas 6 jam terakhir
    hours = 6
    print(f"\nMengambil ringkasan data {hours} jam terakhir...")
    narrative, summary = summarize_vitals(data, n_hours=hours)

    # 3. Output
    print("\n--- Summary Dictionary ---")
    import pprint
    pprint.pprint(summary)

    print("\n--- Narrative Output ---")
    print(narrative)

    print("\n" + "="*60)
