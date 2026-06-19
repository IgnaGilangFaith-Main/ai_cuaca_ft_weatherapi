# AI Prakiraan Cuaca — ML Backend Service

Proyek ini adalah **ML backend service** untuk prediksi cuaca menggunakan **Flask**, **scikit-learn**, dan **XGBoost**. Dikonsumsi oleh frontend Laravel via REST API.

## ✨ Fitur

- Prediksi cuaca untuk 4 waktu sehari (06:00, 12:00, 18:00, 21:00)
- Otomatis pilih model terbaik antara **RandomForest** vs **XGBoost**
- **GridSearchCV** hyperparameter tuning
- **TimeSeriesSplit** — tidak ada data leakage
- **21 fitur** — termasuk encoding musiman, wind direction, feels-like, visibility
- **Cache** WeatherAPI (30 menit TTL)
- **Model versioning** — setiap training simpan versi timestamp
- Health check endpoint (`GET /health`)
- Input validation + prediction logging

## Tech Stack

- **Python** 3.x
- **Flask** — REST API
- **scikit-learn** — RandomForest + GridSearch
- **XGBoost** — classifier alternatif
- **Pandas / NumPy** — feature engineering
- **WeatherAPI** — data historis & forecast
- **Joblib** — model serialization

## Cara Penggunaan

### 1. Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Konfigurasi API Key

Buat file `.env`, isi:

```
API_KEY=your_weatherapi_key_here
```

Dapatkan API key dari [WeatherAPI](https://www.weatherapi.com/).

### 3. (Opsional) Download Data Historis

Untuk training model, download data historis dari WeatherAPI:

```bash
python get_weather_data.py
```

> **Catatan:** Untuk data dalam jumlah besar, API key harus bertipe **premium**.

### 4. Training Model

```bash
python train_model.py
```

Proses:
- Feature engineering (21 fitur)
- TimeSeriesSplit (80/20 cutoff)
- GridSearchCV untuk RandomForest (81 kombinasi)
- Feature importance plot → `model/feature_importance.png`
- XGBoost comparison (16 kombinasi)
- Otomatis pilih model dengan akurasi terbaik
- Confusion matrix, loss curve, accuracy curve
- Simpan model versi timestamp + default

### 5. Jalankan API Server

```bash
python app.py
```

Server jalan di `http://localhost:5000`

### 6. Retrain dengan Data Baru

```bash
python retrain.py                              # retrain semua data
python retrain.py --csv data_baru.csv          # tambah data baru
python retrain.py --full                       # force full retrain
```

## API Endpoints

### `POST /predict-cuaca`

Request body:

```json
{
  "lokasi": "Jakarta"
}
```

Response:

```json
{
  "lokasi": "Jakarta",
  "hasil": [
    {
      "tanggal": "2026-06-18",
      "prediksi_cuaca": "Cerah",
      "detail": {
        "suhu": 31.9,
        "kelembapan": 52,
        "kecepatan_angin": 9.4,
        "kemungkinan_hujan": 3,
        "visibilitas_km": 10.0
      },
      "prediksi": [
        { "jam": 6, "waktu": "06:00", "prediksi_cuaca": "Cerah", "detail": { ... } },
        { "jam": 12, "waktu": "12:00", "prediksi_cuaca": "Cerah", "detail": { ... } },
        { "jam": 18, "waktu": "18:00", "prediksi_cuaca": "Cerah", "detail": { ... } },
        { "jam": 21, "waktu": "21:00", "prediksi_cuaca": "Kemungkinan Hujan Ringan", "detail": { ... } }
      ]
    }
  ],
  "akurasi_model": "98.18%",
  "terakhir_model_diperbarui": "2026-06-18 15:31:30"
}
```

> **Backward compatible** — field `prediksi_cuaca` dan `detail` tetap ada di level `hasil[]`.

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "encoder_loaded": true,
  "accuracy": "98.18%",
  "last_updated": "2026-06-18 15:31:30",
  "api_key_configured": true,
  "timestamp": "2026-06-18 15:32:00"
}
```

## Struktur Project

```
├── app.py                    # Flask API server
├── train_model.py            # Training pipeline (RF + XGBoost)
├── get_weather_data.py       # Data fetcher from WeatherAPI
├── retrain.py                # Retrain script
├── utils.py                  # Shared utilities & feature engineering
├── data_cuaca_histori.csv    # Historical weather data
├── data_prediksi_log.csv     # Prediction log
├── requirements.txt          # Dependencies
├── .env.example              # Environment template
└── model/                    # Model artifacts
    ├── model_cuaca.pkl           # Model aktif
    ├── model_cuaca_v*.pkl        # Model versi timestamp
    ├── label_encoder.pkl
    ├── feature_order.pkl
    ├── accuracy.txt
    ├── last_updated.txt
    ├── feature_importance.png
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── accuracy_curve.png
    └── loss_curve.png
```

## Performance

| Model | Akurasi | Note |
|-------|---------|------|
| RandomForest | 97.88% | GridSearch optimized |
| **XGBoost** | **98.18%** | ✅ Model terpilih |

Feature importance (top 5): `precip_mm`, `cloud`, `doy_cos`, `dewpoint_c`, `doy_sin`

---

**Catatan:** Project ini adalah ML backend service — frontend terpisah (Laravel).
