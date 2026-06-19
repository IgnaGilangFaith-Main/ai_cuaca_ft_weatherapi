import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder
import joblib

from utils import normalize_condition, engineer_features, ALL_FEATURES

# ========== 0. Konfigurasi ==========
USE_GRID_SEARCH = True       # False = pakai default param
TRY_XGBOOST = True            # Coba XGBoost sebagai alternatif
N_FEATURES_TOP = 15           # Feature importance: plot top N

# ========== 1. Load & engineering fitur ==========
df = pd.read_csv('data_cuaca_histori.csv')
df = engineer_features(df)

# ========== 2. Normalisasi label ==========
df['condition_normalized'] = df['condition'].apply(normalize_condition)

print("Distribusi label setelah normalisasi:")
print(df['condition_normalized'].value_counts())
print()

# ========== 3. Siapkan fitur dan label ==========
AVAILABLE_FEATURES = [c for c in ALL_FEATURES if c in df.columns]
print(f"Fitur digunakan ({len(AVAILABLE_FEATURES)}): {AVAILABLE_FEATURES}")

df = df.dropna(subset=AVAILABLE_FEATURES + ['condition_normalized'])

X = df[AVAILABLE_FEATURES]
y = df['condition_normalized']

# ========== 4. Label encoding ==========
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ========== 5. TimeSeriesSplit (no data leakage) ==========
n = len(df)
cutoff = int(n * 0.8)

X_train = X.iloc[:cutoff]
X_test = X.iloc[cutoff:]
y_train = y_encoded[:cutoff]
y_test = y_encoded[cutoff:]

print(f"\n📊 Split: {len(X_train)} train, {len(X_test)} test "
      f"(cutoff = {df['time'].iloc[cutoff]})")

# TimeSeriesSplit cross-val untuk grid search
tscv = TimeSeriesSplit(n_splits=3)

# ========== 6. Hyperparameter Tuning (GridSearch) ==========
if USE_GRID_SEARCH:
    print("\n🔍 GridSearchCV untuk RandomForest...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced_subsample')
    grid = GridSearchCV(rf_base, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    print(f"✅ Best params: {grid.best_params_}")
    print(f"🎯 Best CV accuracy: {grid.best_score_:.2%}")
else:
    model = RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight='balanced_subsample'
    )
    model.fit(X_train, y_train)

# ========== 7. Feature Importance ==========
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n📈 Feature Importance (top 10):")
for i in range(min(10, len(AVAILABLE_FEATURES))):
    print(f"  {i+1}. {AVAILABLE_FEATURES[indices[i]]}: {importances[indices[i]]:.4f}")

top_n = min(N_FEATURES_TOP, len(AVAILABLE_FEATURES))
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), importances[indices][:top_n][::-1], align='center')
plt.yticks(range(top_n), [AVAILABLE_FEATURES[i] for i in indices[:top_n]][::-1])
plt.xlabel('Feature Importance')
plt.title('Feature Importance — Random Forest')
plt.tight_layout()
os.makedirs('model', exist_ok=True)
plt.savefig('model/feature_importance.png')
plt.close()
print("📁 Plot feature importance disimpan ke model/feature_importance.png")

# ========== 8. Learning Curve ==========
train_sizes_abs = np.linspace(int(len(X_train) * 0.1), int(len(X_train) * 0.9), 8).astype(int)

train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []

for train_size in train_sizes_abs:
    X_tr = X_train.iloc[:train_size]
    y_tr = y_train[:train_size]
    X_vl = X_train.iloc[train_size:]
    y_vl = y_train[train_size:]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_vl)) < 2:
        continue

    params = model.get_params()
    params.pop('random_state', None)
    m = RandomForestClassifier(**params, random_state=42)
    m.fit(X_tr, y_tr)

    train_acc_list.append(accuracy_score(y_tr, m.predict(X_tr)))
    val_acc_list.append(accuracy_score(y_vl, m.predict(X_vl)))

    try:
        classes = np.arange(len(le.classes_))
        train_loss_list.append(log_loss(y_tr, m.predict_proba(X_tr), labels=classes))
        val_loss_list.append(log_loss(y_vl, m.predict_proba(X_vl), labels=classes))
    except Exception:
        train_loss_list.append(None)
        val_loss_list.append(None)

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(train_sizes_abs[:len(train_loss_list)], train_loss_list, marker='o', label='Training Loss')
plt.plot(train_sizes_abs[:len(val_loss_list)], val_loss_list, marker='o', label='Validation Loss')
plt.xlabel('Training Samples')
plt.ylabel('Log Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('model/loss_curve.png')
plt.close()

# Plot accuracy
plt.figure(figsize=(8, 5))
plt.plot(train_sizes_abs[:len(train_acc_list)], train_acc_list, marker='o', label='Training Accuracy')
plt.plot(train_sizes_abs[:len(val_acc_list)], val_acc_list, marker='o', label='Validation Accuracy')
plt.xlabel('Training Samples')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('model/accuracy_curve.png')
plt.close()

# ========== 9. Evaluasi ==========
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
laporan = classification_report(y_test, y_pred, target_names=le.classes_, labels=np.arange(len(le.classes_)), zero_division=0)
print(f"\n✅ RandomForest — Akurasi: {akurasi:.2%}")
print("\n📋 Classification Report:\n", laporan)

# ========== 9A. XGBoost Comparison ==========
if TRY_XGBOOST:
    try:
        from xgboost import XGBClassifier
        print("\n🚀 Mencoba XGBoost...")

        xgb_param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
        }
        xgb_base = XGBClassifier(
            random_state=42, eval_metric='mlogloss',
            num_class=len(le.classes_)
        )
        # StratifiedKFold biar setiap fold punya semua kelas
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        xgb_grid = GridSearchCV(xgb_base, xgb_param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=1)
        xgb_grid.fit(X_train, y_train)

        xgb_model = xgb_grid.best_estimator_
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        xgb_report = classification_report(y_test, xgb_pred, target_names=le.classes_, labels=np.arange(len(le.classes_)), zero_division=0)

        print(f"\n✅ XGBoost — Best params: {xgb_grid.best_params_}")
        print(f"✅ XGBoost — Akurasi: {xgb_accuracy:.2%}")
        print("\n📋 Classification Report (XGBoost):\n", xgb_report)

        if xgb_accuracy > akurasi:
            print(f"\n🏆 XGBoost ({xgb_accuracy:.2%}) > RandomForest ({akurasi:.2%}) — pakai XGBoost")
            model = xgb_model
            akurasi = xgb_accuracy
            y_pred = xgb_pred
        else:
            print(f"\n🏆 RandomForest ({akurasi:.2%}) >= XGBoost ({xgb_accuracy:.2%}) — pakai RandomForest")

    except ImportError:
        print("⚠️  XGBoost tidak terinstall, skip.")

# ========== 10. Simpan model dan encoder ==========
os.makedirs('model', exist_ok=True)

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
model_versioned_path = f'model/model_cuaca_v{timestamp}.pkl'
feature_order = AVAILABLE_FEATURES

joblib.dump(model, model_versioned_path)
joblib.dump(le, 'model/label_encoder.pkl')
joblib.dump(feature_order, 'model/feature_order.pkl')
joblib.dump(model, 'model/model_cuaca.pkl')

print(f"\n📁 Model versi {timestamp} disimpan ke {model_versioned_path}")
print(f"📁 Model default disimpan ke model/model_cuaca.pkl")

# ========== 11. Simpan akurasi ==========
with open('model/accuracy.txt', 'w') as f:
    f.write(str(round(akurasi * 100, 2)))

# ========== 12. Confusion Matrix ==========
cm = confusion_matrix(y_test, y_pred)
cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
labels = le.classes_

last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
with open('model/last_updated.txt', 'w') as f:
    f.write(last_updated)
print(f"🕒 Model terakhir diperbarui pada: {last_updated}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('model/confusion_matrix.png')
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Normalized Confusion Matrix')
plt.tight_layout()
plt.savefig('model/confusion_matrix_normalized.png')
plt.close()
