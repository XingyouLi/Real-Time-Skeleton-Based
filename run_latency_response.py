import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

base = "processed_data"
X_train_feat = np.load(os.path.join(base, "X_train_feat.npy"))
X_val_feat = np.load(os.path.join(base, "X_val_feat.npy"))
X_test_feat = np.load(os.path.join(base, "X_test_feat.npy"))
y_train = np.load(os.path.join(base, "y_train_seq.npy"))
y_val = np.load(os.path.join(base, "y_val_seq.npy"))
y_test = np.load(os.path.join(base, "y_test_seq.npy"))

N_train, T, D = X_train_feat.shape
N_val, N_test = X_val_feat.shape[0], X_test_feat.shape[0]
NUM_CLASSES = 9

scaler = StandardScaler()
train_norm = scaler.fit_transform(X_train_feat.reshape(-1, D)).reshape(N_train, T, D)
val_norm = scaler.transform(X_val_feat.reshape(-1, D)).reshape(N_val, T, D)
test_norm = scaler.transform(X_test_feat.reshape(-1, D)).reshape(N_test, T, D)

tf.random.set_seed(42)
np.random.seed(42)
model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(T, D)),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(train_norm, y_train, validation_data=(val_norm, y_val),
          epochs=50, batch_size=32, callbacks=[early_stop], verbose=1)

print("\n===== STREAMING INFERENCE LATENCY ANALYSIS =====")
print("Simulating frame-by-frame streaming inference...")
print("Recording how many frames are needed before the model makes")
print("its first correct prediction for each test sample.\n")

correct_delays = []
incorrect_samples = []

for i in range(min(500, N_test)):
    true_label = y_test[i]
    found_correct = False
    for t in range(1, T + 1):
        partial_seq = np.zeros((1, T, D))
        partial_seq[0, :t, :] = test_norm[i, :t, :]
        pred = np.argmax(model.predict(partial_seq, verbose=0), axis=1)[0]
        if pred == true_label:
            delay_frames = t
            delay_ms = t * (1000.0 / 30.0)
            correct_delays.append({
                'sample': i, 'true_label': int(true_label),
                'delay_frames': delay_frames, 'delay_ms': delay_ms
            })
            found_correct = True
            break
    if not found_correct:
        incorrect_samples.append({'sample': i, 'true_label': int(true_label)})

correct_arr = np.array([d['delay_frames'] for d in correct_delays])
print(f"\nTotal samples evaluated: {min(500, N_test)}")
print(f"Correctly classified within {T} frames: {len(correct_delays)}")
print(f"Not correctly classified within {T} frames: {len(incorrect_samples)}")
if len(correct_arr) > 0:
    print(f"\nMean frames to first correct: {correct_arr.mean():.1f} frames ({correct_arr.mean() * 1000 / 30:.0f} ms)")
    print(f"Median frames to first correct: {np.median(correct_arr):.1f} frames ({np.median(correct_arr) * 1000 / 30:.0f} ms)")
    print(f"Std frames to first correct: {correct_arr.std():.1f} frames")
    print(f"25th percentile: {np.percentile(correct_arr, 25):.1f} frames")
    print(f"75th percentile: {np.percentile(correct_arr, 75):.1f} frames")
    print(f"Min frames: {correct_arr.min():.0f} frames")
    print(f"Max frames: {correct_arr.max():.0f} frames")

with open('latency_response_results.json', 'w') as f:
    json.dump({
        'total_samples': min(500, N_test),
        'correct_count': len(correct_delays),
        'incorrect_count': len(incorrect_samples),
        'mean_frames': float(correct_arr.mean()) if len(correct_arr) > 0 else None,
        'median_frames': float(np.median(correct_arr)) if len(correct_arr) > 0 else None,
        'std_frames': float(correct_arr.std()) if len(correct_arr) > 0 else None,
        'mean_ms': float(correct_arr.mean() * 1000 / 30) if len(correct_arr) > 0 else None,
        'median_ms': float(np.median(correct_arr) * 1000 / 30) if len(correct_arr) > 0 else None
    }, f, indent=2)
print("\nSaved to latency_response_results.json")
print("Done!")