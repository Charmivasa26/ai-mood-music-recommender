"""
AI Mood-Based Music Recommender — CNN-LSTM FIXED v3
ROOT CAUSE FIX: The labels are DETERMINISTIC rule-based.
The model must learn those exact rules. We engineer features
that make the rules linearly separable, then use CNN-LSTM on top.
"""

import os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense,
    Dropout, BatchNormalization, Concatenate, Flatten
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.makedirs("artifacts", exist_ok=True)

# ─── 1. Load Dataset ──────────────────────────────────────────
print("="*60)
print("  STEP 1: Loading Dataset")
print("="*60)
df = pd.read_csv("dataset.csv")
print(f"  Records: {len(df)}")
print(df['manual_vibe'].value_counts().to_string())

# ─── 2. FIX: Merge tiny class ─────────────────────────────────
# Focus/Instrumental has only 18 samples — merge into Mixed Vibe
df['manual_vibe'] = df['manual_vibe'].replace('Focus/Instrumental', 'Mixed Vibe')
print(f"\n  After merge:")
print(df['manual_vibe'].value_counts().to_string())

# ─── 3. Feature Engineering ───────────────────────────────────
# KEY INSIGHT: Labels are rule-based on energy, acousticness,
# danceability, instrumentalness. We create engineered features
# that directly expose these decision boundaries to the model.
print("\n" + "="*60)
print("  STEP 2: Feature Engineering")
print("="*60)

# Base audio features
BASE_COLS = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
]

# Engineer rule-signal features (these directly encode the labeling logic)
df['energy_sq']          = df['energy'] ** 2
df['dance_energy']       = df['danceability'] * df['energy']
df['acoustic_inv_energy']= df['acousticness'] * (1 - df['energy'])
df['valence_dance']      = df['valence'] * df['danceability']
df['high_energy_flag']   = (df['energy'] > 0.8).astype(float)
df['acoustic_flag']      = (df['acousticness'] > 0.7).astype(float)
df['groovy_flag']        = (df['danceability'] > 0.7).astype(float)
df['instr_flag']         = (df['instrumentalness'] > 0.5).astype(float)

ENGINEERED_COLS = [
    'energy_sq', 'dance_energy', 'acoustic_inv_energy',
    'valence_dance', 'high_energy_flag', 'acoustic_flag',
    'groovy_flag', 'instr_flag'
]

FEATURE_COLS = BASE_COLS + ENGINEERED_COLS
print(f"  Total features: {len(FEATURE_COLS)}")
print(f"  {FEATURE_COLS}")

# ─── 4. Preprocessing ─────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3: Preprocessing")
print("="*60)

df = df.dropna(subset=FEATURE_COLS + ['manual_vibe'])

X     = df[FEATURE_COLS].values
y_raw = df['manual_vibe'].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

le          = LabelEncoder()
y_int       = le.fit_transform(y_raw)
NUM_CLASSES = len(le.classes_)
y_onehot    = to_categorical(y_int, num_classes=NUM_CLASSES)
print(f"  Classes ({NUM_CLASSES}): {list(le.classes_)}")

# Reshape: (N, num_features, 1) for CNN-LSTM
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# 70/15/15 stratified split
X_temp, X_test, y_temp, y_test, yi_temp, yi_test = train_test_split(
    X_reshaped, y_onehot, y_int,
    test_size=0.15, random_state=SEED, stratify=y_int
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.176, random_state=SEED, stratify=yi_temp
)
print(f"  Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Class weights
cw_arr  = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
cw_dict = dict(enumerate(cw_arr))
print(f"\n  Class weights:")
for k, v in cw_dict.items():
    print(f"    {le.classes_[k]}: {v:.3f}")

# ─── 5. Build CNN-LSTM Model ──────────────────────────────────
print("\n" + "="*60)
print("  STEP 4: Building CNN-LSTM Model")
print("="*60)

def build_model(input_shape, num_classes):
    inp = Input(shape=input_shape, name="features")

    # CNN Branch A: fine-grained patterns (kernel=2)
    a = Conv1D(128, 2, activation='relu', padding='same')(inp)
    a = BatchNormalization()(a)
    a = Conv1D(128, 2, activation='relu', padding='same')(a)
    a = BatchNormalization()(a)

    # CNN Branch B: coarser patterns (kernel=4)
    b = Conv1D(128, 4, activation='relu', padding='same')(inp)
    b = BatchNormalization()(b)
    b = Conv1D(128, 4, activation='relu', padding='same')(b)
    b = BatchNormalization()(b)

    # Merge branches
    x = Concatenate(axis=-1)([a, b])
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # 2-layer LSTM to capture feature interaction sequences
    x = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)(x)
    x = LSTM(64,  return_sequences=False, dropout=0.2)(x)

    # Classification head
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation='softmax', name="output")(x)

    return Model(inputs=inp, outputs=out, name="CNN_LSTM_v3")

model = build_model((X_train.shape[1], 1), NUM_CLASSES)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ─── 6. Train ─────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5: Training")
print("="*60)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=8, min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    class_weight=cw_dict,
    callbacks=callbacks,
    verbose=1
)

# ─── 7. Evaluate ──────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6: Evaluation")
print("="*60)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n  ✅ Test Accuracy : {test_acc*100:.2f}%")
print(f"     Test Loss     : {test_loss:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n  Classification Report:")
print(classification_report(y_true, y_pred, target_names=le.classes_))

# ─── 8. Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['accuracy'],     label='Train', color='#4CAF50')
axes[0].plot(history.history['val_accuracy'], label='Val',   color='#2196F3')
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history.history['loss'],     label='Train', color='#F44336')
axes[1].plot(history.history['val_loss'], label='Val',   color='#FF9800')
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("artifacts/training_history.png", dpi=150)
plt.close()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix"); plt.tight_layout()
plt.savefig("artifacts/confusion_matrix.png", dpi=150)
plt.close()

# ─── 9. Save Artifacts ────────────────────────────────────────
model.save("artifacts/cnn_lstm_model.keras")
with open("artifacts/scaler.pkl",        "wb") as f: pickle.dump(scaler, f)
with open("artifacts/label_encoder.pkl", "wb") as f: pickle.dump(le, f)
with open("artifacts/feature_cols.pkl",  "wb") as f: pickle.dump(FEATURE_COLS, f)
df.to_csv("artifacts/processed_dataset.csv", index=False)

print("\n  ✅ All artifacts saved.")
print("="*60)
print(f"  FINAL TEST ACCURACY: {test_acc*100:.2f}%")
print("  Run: streamlit run app.py")
print("="*60)