from flask import Flask, request, jsonify
import pandas as pd
import joblib
import requests
import logging
import os
import re
import csv
from datetime import datetime
import math
from dotenv import load_dotenv

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load .env
load_dotenv()
API_KEY = os.getenv('API_KEY')  # WeatherAPI key
if not API_KEY:
    logger.critical("Variabel environment API_KEY tidak diatur.")

# Konstanta
FORECAST_DAYS = 3
MODEL_PATH = 'model/model_cuaca.pkl'
ENCODER_PATH = 'model/label_encoder.pkl'
AKURASI_PATH = 'model/accuracy.txt'
FEATURE_ORDER_PATH = 'model/feature_order.pkl'
LAST_UPDATED_PATH = 'model/last_updated.txt'
PREDICTION_LOG = 'data_prediksi_log.csv'

# Cache TTL (seconds)
CACHE_TTL = 1800  # 30 menit

# Flask App
app = Flask(__name__)

# Terjemahan label sesuai mapping akhir
TERJEMAHAN_CUACA = {
    'Clear': 'Cerah',
    'Cloudy': 'Berawan',
    'Overcast': 'Mendung',
    'Patchy Rain Possible': 'Kemungkinan Hujan Ringan',
    'Rain': 'Hujan',
    'Heavy Rain': 'Hujan Lebat',
    'Thunderstorm': 'Hujan Petir',
    'Fog': 'Berkabut',
    'Snow': 'Salju',
    'Other': 'Lainnya'
}

# Waktu prediksi — beberapa jam dalam sehari
# Laravel bisa request spesifik jam, default prediksi 3 waktu
PREDICTION_HOURS = [6, 12, 18, 21]  # pagi, siang, sore, malam


# Load model
def load_ml_components():
    try:
        model = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        logger.info("Model dan Label Encoder dimuat.")
        return model, le
    except Exception as e:
        logger.critical("Gagal memuat model/encoder: %s", e)
        return None, None


def load_accuracy(path):
    try:
        with open(path, 'r') as f:
            return float(f.read().strip())
    except Exception as e:
        logger.warning("Gagal memuat akurasi: %s", e)
        return None


def load_feature_order(path):
    try:
        return joblib.load(path)
    except Exception as e:
        logger.critical("Gagal memuat urutan fitur: %s", e)
        return None


def load_last_updated(path):
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        logger.warning("Gagal memuat timestamp terakhir model diperbarui: %s", e)
        return None


model, le = load_ml_components()
AKURASI_MODEL = load_accuracy(AKURASI_PATH)
FEATURE_ORDER = load_feature_order(FEATURE_ORDER_PATH)
LAST_UPDATED = load_last_updated(LAST_UPDATED_PATH)


# ===================== CACHE =====================
_cache_store = {}
_cache_timestamps = {}


def get_cached(key):
    """Get from cache if within TTL."""
    if key in _cache_store and key in _cache_timestamps:
        age = (datetime.now() - _cache_timestamps[key]).total_seconds()
        if age < CACHE_TTL:
            logger.info("Cache HIT for %s (age: %ds)", key, int(age))
            return _cache_store[key]
    return None


def set_cache(key, value):
    """Set cache entry."""
    _cache_store[key] = value
    _cache_timestamps[key] = datetime.now()


# Ambil data cuaca dari API (with cache)
def get_weather_data(lokasi):
    if not API_KEY:
        raise ValueError("API key tidak tersedia.")

    # Check cache
    cache_key = f"weather_{lokasi}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached

    url = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={lokasi}&days={FORECAST_DAYS}"
    logger.info("Fetching weather data for: %s", lokasi)
    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    set_cache(cache_key, data)
    logger.info("Cached weather data for: %s", lokasi)
    return data


# Fungsi normalisasi label untuk hasil API eksternal
def normalize_condition(label):
    label = label.lower()
    if 'clear' in label or 'sunny' in label:
        return 'Clear'
    elif 'partly cloudy' in label or 'cloudy' in label:
        return 'Cloudy'
    elif 'overcast' in label:
        return 'Overcast'
    elif 'patchy rain' in label or 'light rain' in label:
        return 'Patchy Rain Possible'
    elif 'moderate rain' in label or (label.strip() == 'rain'):
        return 'Rain'
    elif 'heavy rain' in label:
        return 'Heavy Rain'
    elif 'thunderstorm' in label:
        return 'Thunderstorm'
    elif 'mist' in label or 'fog' in label:
        return 'Fog'
    elif 'snow' in label:
        return 'Snow'
    else:
        return 'Other'


# Build feature dict from a forecast hour
def build_feature_dict(jam, fitur_order):
    """Build feature dict from hourly data, matching training features."""
    # Extract temporal features
    hour_val = pd.to_datetime(jam['time']).hour
    month_val = pd.to_datetime(jam['time']).month
    day_of_year = pd.to_datetime(jam['time']).dayofyear

    feat = {
        'temp_c': jam.get('temp_c', 25.0),
        'humidity': jam.get('humidity', 80),
        'wind_kph': jam.get('wind_kph', 10),
        'cloud': jam.get('cloud', 50),             # FIXED: cloud cover, not daily_chance_of_rain
        'precip_mm': jam.get('precip_mm', 0.0),
        'pressure_mb': jam.get('pressure_mb', 1010.0),
        'uv': jam.get('uv', 6.0),
        'dewpoint_c': jam.get('dewpoint_c', 20.0),
        'feelslike_c': jam.get('feelslike_c', 25.0),
        'vis_km': jam.get('vis_km', 10.0),
        'gust_kph': jam.get('gust_kph', 15.0),
        # Temporal engineered features
        'hour_sin': math.sin(2 * math.pi * hour_val / 24),
        'hour_cos': math.cos(2 * math.pi * hour_val / 24),
        'doy_sin': math.sin(2 * math.pi * day_of_year / 366),
        'doy_cos': math.cos(2 * math.pi * day_of_year / 366),
        'month': month_val,
        'season': 1 if month_val in [12, 1, 2] else 2 if month_val in [3, 4, 5] else 3 if month_val in [6, 7, 8] else 4,
        'day_of_week': pd.to_datetime(jam['time']).dayofweek,
        'day': pd.to_datetime(jam['time']).day,
        # Wind direction encoding
        'wind_dir_sin': 0.0,
        'wind_dir_cos': 0.0,
    }

    # Encode wind direction
    wind_dir = jam.get('wind_dir', '')
    if wind_dir:
        wind_dir_map = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        deg = wind_dir_map.get(wind_dir, 0)
        feat['wind_dir_sin'] = math.sin(2 * math.pi * deg / 360)
        feat['wind_dir_cos'] = math.cos(2 * math.pi * deg / 360)

    # Only keep features that are in the trained feature_order
    return {k: feat[k] for k in fitur_order if k in feat}


# Proses prediksi
def process_forecast(forecast_data, ml_model, label_encoder):
    hasil = []

    for hari in forecast_data:
        try:
            tanggal = hari['date']
            hours = hari.get('hour', [])
            logger.info("Memproses tanggal: %s", tanggal)

            hourly_results = []
            for jam in hours:
                jam_time = pd.to_datetime(jam['time'])
                hour_val = jam_time.hour

                # Only predict at specific hours, or predict all hours
                if hour_val not in PREDICTION_HOURS:
                    continue

                # Build features
                fitur_dict = build_feature_dict(jam, FEATURE_ORDER)
                fitur = pd.DataFrame([fitur_dict])

                # Predict
                prediksi = ml_model.predict(fitur)
                label_en = label_encoder.inverse_transform(prediksi)[0]
                label_id = TERJEMAHAN_CUACA.get(label_en, label_en)

                hourly_results.append({
                    'jam': hour_val,
                    'waktu': jam_time.strftime('%H:%M'),
                    'prediksi_cuaca': label_id,
                    'detail': {
                        'suhu': fitur_dict.get('temp_c'),
                        'kelembapan': fitur_dict.get('humidity'),
                        'kecepatan_angin': fitur_dict.get('wind_kph'),
                        'kemungkinan_hujan': jam.get('chance_of_rain', fitur_dict.get('cloud')),
                        'visibilitas_km': fitur_dict.get('vis_km'),
                    }
                })

            if hourly_results:
                # Backward compat — default = jam 12 (siang)
                default = next((h for h in hourly_results if h['jam'] == 12), hourly_results[0])
                hasil.append({
                    'tanggal': tanggal,
                    'prediksi_cuaca': default['prediksi_cuaca'],
                    'detail': default['detail'],
                    'prediksi': hourly_results,  # multi-jam detail
                })

        except Exception as e:
            logger.error("Gagal prediksi pada %s: %s", tanggal, e)
            continue

    return hasil


# Log prediction for future retraining
def log_prediction(lokasi, hasil_prediksi, status="success"):
    """Append prediction to log CSV."""
    try:
        file_exists = os.path.isfile(PREDICTION_LOG)
        with open(PREDICTION_LOG, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'lokasi', 'status', 'jumlah_hari'])
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                lokasi,
                status,
                len(hasil_prediksi)
            ])
    except Exception as e:
        logger.warning("Gagal log prediksi: %s", e)


# Validate location parameter
def validate_lokasi(lokasi):
    if not lokasi:
        return False, "Parameter 'lokasi' wajib diisi"
    if not isinstance(lokasi, str):
        return False, "Parameter 'lokasi' harus berupa string"
    if len(lokasi) > 100:
        return False, "Parameter 'lokasi' maksimal 100 karakter"
    # Only allow letters, spaces, commas, dots, dashes
    if not re.match(r'^[a-zA-Z\s,.\-]+$', lokasi):
        return False, "Parameter 'lokasi' hanya boleh huruf, spasi, koma, titik, dan strip"
    return True, ""


# ===================== ENDPOINTS =====================

@app.route('/predict-cuaca', methods=['POST'])
def predict():
    if not model or not le or not FEATURE_ORDER:
        return jsonify({"error": "Model belum dimuat"}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body harus JSON"}), 400

    lokasi = data.get('lokasi')

    # Validate
    valid, msg = validate_lokasi(lokasi)
    if not valid:
        return jsonify({"error": msg}), 400

    try:
        weather_data = get_weather_data(lokasi)
        forecast = weather_data['forecast']['forecastday']
        hasil = process_forecast(forecast, model, le)

        response = {
            'lokasi': lokasi,
            'hasil': hasil,
            'akurasi_model': f"{AKURASI_MODEL:.2f}%" if AKURASI_MODEL else None,
            'terakhir_model_diperbarui': LAST_UPDATED if LAST_UPDATED else None
        }

        log_prediction(lokasi, hasil)
        return jsonify(response)

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else 500
        logger.error("HTTP error dari API: %s", e)
        return jsonify({"error": f"Gagal mengambil data cuaca dari API (HTTP {status_code})"}), status_code
    except requests.exceptions.ConnectionError:
        logger.error("Koneksi ke API gagal")
        return jsonify({"error": "Gagal terhubung ke API cuaca"}), 502
    except KeyError as e:
        logger.error("Data dari API tidak sesuai format: %s", e)
        return jsonify({"error": "Data cuaca dari API tidak lengkap"}), 502
    except Exception as e:
        logger.error("Gagal memproses prediksi: %s", e)
        return jsonify({"error": "Gagal mengambil atau memproses data cuaca"}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = {
        'status': 'ok' if model and le and FEATURE_ORDER else 'degraded',
        'model_loaded': model is not None,
        'encoder_loaded': le is not None,
        'feature_order_loaded': FEATURE_ORDER is not None,
        'api_key_configured': API_KEY is not None,
        'accuracy': f"{AKURASI_MODEL:.2f}%" if AKURASI_MODEL else None,
        'last_updated': LAST_UPDATED if LAST_UPDATED else None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    http_code = 200 if status['status'] == 'ok' else 503
    return jsonify(status), http_code


if __name__ == '__main__':
    app.run(port=5000, debug=False)
