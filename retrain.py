#!/usr/bin/env python3
"""
Retrain script — update model with new data incrementally.
Usage:
    python retrain.py                          # retrain with all CSV data
    python retrain.py --csv data_baru.csv      # retrain with additional CSV
    python retrain.py --full                   # force full retrain from scratch
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import joblib
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from utils import normalize_condition, get_season, engineer_features, ALL_FEATURES

# === Config ===
BASE_CSV = 'data_cuaca_histori.csv'
MODEL_DIR = 'model'
MODEL_PATH = f'{MODEL_DIR}/model_cuaca.pkl'
ENCODER_PATH = f'{MODEL_DIR}/label_encoder.pkl'
FEATURE_ORDER_PATH = f'{MODEL_DIR}/feature_order.pkl'
AKURASI_PATH = f'{MODEL_DIR}/accuracy.txt'
LAST_UPDATED_PATH = f'{MODEL_DIR}/last_updated.txt'


def retrain(args):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    print("📂 Loading data...")
    if args.csv and os.path.exists(args.csv):
        df_new = pd.read_csv(args.csv)
        print(f"   Data baru: {len(df_new)} baris dari {args.csv}")
        if os.path.exists(BASE_CSV):
            df_base = pd.read_csv(BASE_CSV)
            df = pd.concat([df_base, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=['time', 'lokasi'])
            print(f"   Gabung: {len(df_base)} + {len(df_new)} = {len(df)} baris")
        else:
            df = df_new
    else:
        df = pd.read_csv(BASE_CSV)
        print(f"   {len(df)} baris dari {BASE_CSV}")

    # Feature engineering
    print("🔧 Feature engineering...")
    df = engineer_features(df)

    # Normalize labels
    df['condition_normalized'] = df['condition'].apply(normalize_condition)

    # Drop NaN
    available_features = [c for c in ALL_FEATURES if c in df.columns]
    df = df.dropna(subset=available_features + ['condition_normalized'])

    X = df[available_features]
    y = df['condition_normalized']

    # Label encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Time series split (80/20)
    n = len(df)
    cutoff = int(n * 0.8)
    X_train = X.iloc[:cutoff]
    X_test = X.iloc[cutoff:]
    y_train = y_encoded[:cutoff]
    y_test = y_encoded[cutoff:]

    print(f"📊 Split: {len(X_train)} train, {len(X_test)} test")

    # Load existing model for warm start, or create new
    if not args.full and os.path.exists(MODEL_PATH):
        print("🔄 Load model existing sebagai warm start...")
        try:
            existing_model = joblib.load(MODEL_PATH)
            # Retrain with new data
            model = RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight='balanced_subsample',
                warm_start=True
            )
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"   ⚠️  Gagal load model: {e}, training dari awal...")
            model = RandomForestClassifier(
                n_estimators=200, random_state=42, class_weight='balanced_subsample'
            )
            model.fit(X_train, y_train)
    else:
        print("🆕 Training model baru...")
        model = RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight='balanced_subsample'
        )
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    akurasi = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)

    print(f"\n✅ Retrain selesai — Akurasi: {akurasi:.2%}")
    print("\n📋 Classification Report:\n", report)

    # Save versioned
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_ver_path = f'{MODEL_DIR}/model_cuaca_v{timestamp}.pkl'
    joblib.dump(model, model_ver_path)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    joblib.dump(available_features, FEATURE_ORDER_PATH)

    with open(AKURASI_PATH, 'w') as f:
        f.write(str(round(akurasi * 100, 2)))

    last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LAST_UPDATED_PATH, 'w') as f:
        f.write(last_updated)

    print(f"📁 Model versi {timestamp} disimpan ke {model_ver_path}")
    print(f"🕒 Model diperbarui pada: {last_updated}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain weather prediction model')
    parser.add_argument('--csv', help='New CSV file with additional data')
    parser.add_argument('--full', action='store_true', help='Force full retrain from scratch')
    args = parser.parse_args()

    retrain(args)
