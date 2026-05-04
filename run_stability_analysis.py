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

print("\n===== STREAMING STABILITY ANALYSIS =====")
print("For samples correctly classified within 30 frames,")
print("we measure prediction stability after the first correct prediction.")
print("Stability is defined as the proportion of the remaining frames")
print("in which the prediction remains correct.\n")

stability_scores = []
samples_with_stability = 0
total_correct = 0

for i in range(min(500, N_test)):
    true_label = y_test[i]
    first_correct_frame = None
    predictions = []
    
    for t in range(1, T + 1):
        partial_seq = np.zeros((1, T, D))
        partial_seq[0, :t, :] = test_norm[i, :t, :]
        pred = np.argmax(model.predict(partial_seq, verbose=0), axis=1)[0]
        predictions.append(pred)
        if first_correct_frame is None and pred == true_label:
            first_correct_frame = t
    
    if first_correct_frame is not None:
        total_correct += 1
        remaining_frames = T - first_correct_frame
        if remaining_frames > 0:
            correct_after = 0
            for j in range(first_correct_frame, T):
                if predictions[j] == true_label:
                    correct_after += 1
            stability = correct_after / remaining_frames
            stability_scores.append(stability)
            samples_with_stability += 1

stability_arr = np.array(stability_scores)

print(f"Total samples evaluated: {min(500, N_test)}")
print(f"Correctly classified within 30 frames: {total_correct}")
print(f"Samples with at least 1 remaining frame for stability: {samples_with_stability}")
print(f"\nPost-first-correct prediction stability:")
print(f"  Mean stability: {stability_arr.mean():.3f} ({stability_arr.mean()*100:.1f}%)")
print(f"  Median stability: {np.median(stability_arr):.3f} ({np.median(stability_arr)*100:.1f}%)")
print(f"  Std stability: {stability_arr.std():.3f}")

bins = [0.0, 0.5, 0.8, 0.9, 0.95, 1.0]
print(f"\nStability distribution:")
for i in range(len(bins) - 1):
    count = np.sum((stability_arr >= bins[i]) & (stability_arr < bins[i+1]))
    print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {count} samples ({count/len(stability_arr)*100:.1f}%)")

count_perfect = np.sum(stability_arr >= 0.999)
print(f"\n  Perfect stability (>= 0.999): {count_perfect} samples ({count_perfect/len(stability_arr)*100:.1f}%)")

with open('stability_analysis_results.json', 'w') as f:
    json.dump({
        'total_evaluated': min(500, N_test),
        'total_correct': total_correct,
        'samples_with_stability': samples_with_stability,
        'mean_stability': float(stability_arr.mean()),
        'median_stability': float(np.median(stability_arr)),
        'std_stability': float(stability_arr.std()),
        'perfect_stability_count': int(count_perfect),
        'perfect_stability_pct': float(count_perfect / len(stability_arr) * 100)
    }, f, indent=2)
print("\nSaved to stability_analysis_results.json")
print("Done!")