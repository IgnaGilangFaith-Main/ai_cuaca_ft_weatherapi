import os
import requests
import json
import pandas as pd
import time
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

API_KEY = os.getenv('API_KEY') # WeatherAPI key

# Multi-city support — each entry: {"name": "Kebumen", "latlon": "-7.694172256470859, 109.69516925410542"}
LOKASI = [
    {"name": "Kebumen", "latlon": "-7.694172256470859, 109.69516925410542"},
    # Tambah kota lain di sini
]

TANGGAL_MULAI = '2025-01-01'
TANGGAL_SELESAI = date.today().strftime('%Y-%m-%d')
NAMA_FILE = 'data_cuaca_histori.csv'
MAX_RETRY = 5
RETRY_SLEEP = 5  # detik
MAX_WORKERS = 5  # parallel threads


def request_with_retry(url):
    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"  🔁 Percobaan {attempt} gagal: {e}")
            if attempt < MAX_RETRY:
                print(f"  ⏳ Menunggu {RETRY_SLEEP} detik sebelum retry...")
                time.sleep(RETRY_SLEEP)
            else:
                print("  ❌ Gagal setelah beberapa percobaan.")
                return None


def fetch_single_date(lokasi_name, tanggal_str):
    """Fetch weather data for one date, returns list of hourly dicts."""
    url = f"https://api.weatherapi.com/v1/history.json?key={API_KEY}&q={lokasi_name}&dt={tanggal_str}"
    response = request_with_retry(url)
    if response is None:
        print(f"  ❌ Gagal ambil data tanggal: {tanggal_str} setelah retry.")
        return []

    try:
        data = response.json()
        if 'forecast' in data and 'forecastday' in data['forecast'] and len(data['forecast']['forecastday']) > 0:
            hourly = data['forecast']['forecastday'][0]['hour']
            rows = []
            for jam in hourly:
                rows.append({
                    'time': jam['time'],
                    'hour': pd.to_datetime(jam['time']).hour,
                    'temp_c': jam.get('temp_c', 0),
                    'humidity': jam.get('humidity', 0),
                    'wind_kph': jam.get('wind_kph', 0),
                    'cloud': jam.get('cloud', 0),
                    'pressure_mb': jam.get('pressure_mb', 0),
                    'uv': jam.get('uv', 0),
                    'precip_mm': jam.get('precip_mm', 0),
                    'dewpoint_c': jam.get('dewpoint_c', 0),
                    # ——— New fields ———
                    'feelslike_c': jam.get('feelslike_c', 0),
                    'vis_km': jam.get('vis_km', 0),
                    'gust_kph': jam.get('gust_kph', 0),
                    'wind_dir': jam.get('wind_dir', ''),
                    # ———————
                    'condition': jam['condition']['text'],
                    'lokasi': lokasi_name,
                })
            print(f"  ✅ {tanggal_str} — {len(hourly)} jam")
            return rows
        else:
            print(f"  ⚠️  Tidak ada data untuk tanggal: {tanggal_str}")
            return []
    except json.JSONDecodeError:
        print(f"  ❌ JSON error untuk tanggal: {tanggal_str}")
        return []


def ambil_data():
    # Cek file jika ada untuk resume
    if os.path.exists(NAMA_FILE):
        df_exist = pd.read_csv(NAMA_FILE)
        tanggal_sudah = set(
            df_exist['lokasi'].astype(str) + '|' +
            pd.to_datetime(df_exist['time']).dt.date.astype(str)
        )
        print(f"🔄 Mode resume: {len(tanggal_sudah)} hari sudah diambil sebelumnya.")
    else:
        df_exist = pd.DataFrame()
        tanggal_sudah = set()

    all_data = []
    tanggal_awal = datetime.strptime(TANGGAL_MULAI, '%Y-%m-%d')
    tanggal_akhir = datetime.strptime(TANGGAL_SELESAI, '%Y-%m-%d')
    jumlah_hari = (tanggal_akhir - tanggal_awal).days + 1

    tasks = []
    for lokasi in LOKASI:
        print(f"\n📍 Mengambil data untuk {lokasi['name']} ({lokasi['latlon']}) "
              f"dari {TANGGAL_MULAI} hingga {TANGGAL_SELESAI}...")

        for i in range(jumlah_hari):
            tanggal = tanggal_awal + timedelta(days=i)
            tanggal_str = tanggal.strftime('%Y-%m-%d')
            key = f"{lokasi['name']}|{tanggal_str}"

            if key in tanggal_sudah:
                print(f"  ⏩ {lokasi['name']} — {tanggal_str} sudah ada, lewati.")
                continue

            tasks.append((lokasi['name'], tanggal_str))

    # Parallel fetch
    if tasks:
        print(f"\n⏳ Mengambil {len(tasks)} task dengan {MAX_WORKERS} thread parallel...\n")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_map = {
                executor.submit(fetch_single_date, name, tgl): (name, tgl)
                for name, tgl in tasks
            }
            for future in as_completed(future_map):
                name, tgl = future_map[future]
                try:
                    rows = future.result()
                    all_data.extend(rows)
                except Exception as e:
                    print(f"  ❌ {name} — {tgl} error: {e}")

        # Simpan
        if all_data:
            df_baru = pd.DataFrame(all_data)
            if not df_exist.empty:
                df_final = pd.concat([df_exist, df_baru], ignore_index=True)
                df_final = df_final.drop_duplicates(subset=['time', 'lokasi'])
            else:
                df_final = df_baru
            df_final.to_csv(NAMA_FILE, index=False)
            print(f"\n✅ Selesai. {len(df_baru)} baris baru ditambahkan ke `{NAMA_FILE}` "
                  f"(total {len(df_final)} baris)")
        elif not df_exist.empty:
            print(f"ℹ️  Tidak ada data baru. Data lama tetap di `{NAMA_FILE}`")
        else:
            print("❌ Tidak ada data yang disimpan.")
    else:
        print("ℹ️  Semua data sudah terambbil. Tidak ada tugas baru.")


if __name__ == '__main__':
    ambil_data()
