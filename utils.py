"""
Shared utilities for weather prediction project.
Functions extracted for reuse across train_model.py, app.py, retrain.py
"""

import numpy as np
import pandas as pd


# Mapping label normalisasi
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


# Season classifier
def get_season(dt):
    m = dt.month
    if m in [12, 1, 2]:
        return 1  # Monsoon
    elif m in [3, 4, 5]:
        return 2  # Transition 1
    elif m in [6, 7, 8]:
        return 3  # Dry
    else:
        return 4  # Transition 2


# Wind direction mapping
WIND_DIR_MAP = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
}

BASE_FEATURES = [
    'temp_c', 'humidity', 'wind_kph', 'cloud', 'precip_mm',
    'pressure_mb', 'uv', 'dewpoint_c',
    'feelslike_c', 'vis_km', 'gust_kph',
]

ENGINEERED_FEATURES = [
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
    'month', 'season', 'day_of_week', 'day',
    'wind_dir_sin', 'wind_dir_cos',
]

ALL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


def engineer_features(df):
    """Add engineered temporal features to dataframe. Returns sorted, modified df."""
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear
    df['day'] = df['time'].dt.day
    df['day_of_week'] = df['time'].dt.dayofweek

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 366)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 366)
    df['season'] = df['time'].apply(get_season)

    # Handle new columns that might not exist in old CSV
    for col in ['feelslike_c', 'vis_km', 'gust_kph']:
        if col not in df.columns:
            df[col] = 0.0

    if 'wind_dir' in df.columns:
        df['wind_dir_deg'] = df['wind_dir'].map(WIND_DIR_MAP).fillna(0)
    else:
        df['wind_dir_deg'] = 0.0

    df['wind_dir_sin'] = np.sin(2 * np.pi * df['wind_dir_deg'] / 360)
    df['wind_dir_cos'] = np.cos(2 * np.pi * df['wind_dir_deg'] / 360)

    return df
