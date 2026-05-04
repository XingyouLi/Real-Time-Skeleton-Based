import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, GlobalAveragePooling1D, Dropout, Dense, Flatten
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

N_train, T = X_train_feat.shape[0], X_train_feat.shape[1]
N_val, N_test = X_val_feat.shape[0], X_test_feat.shape[0]
NUM_CLASSES = 9
SEEDS = [42, 123, 456, 789, 1024]

# 三种特征配置
feature_configs = {
    'Angles Only': (8, lambda X: X[:, :, :8]),
    'Delta Only': (8, lambda X: X[:, :, 8:]),
    'Angles+Delta': (16, lambda X: X),
}

def build_mlp(input_dim):
    return Sequential([
        Flatten(input_shape=(T, input_dim)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

def build_lstm(input_dim):
    return Sequential([
        LSTM(64, return_sequences=True, input_shape=(T, input_dim)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

def build_cnn(input_dim):
    return Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=(T, input_dim)),
        Conv1D(128, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

def build_tcn(input_dim):
    return Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='causal', input_shape=(T, input_dim)),
        Conv1D(64, kernel_size=3, activation='relu', padding='causal', dilation_rate=2),
        Conv1D(64, kernel_size=3, activation='relu', padding='causal', dilation_rate=4),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

architectures = {
    'MLP': build_mlp,
    'LSTM': build_lstm,
    '1D-CNN': build_cnn,
    'TCN': build_tcn,
}

results = {}
for feat_name, (dim, extract_fn) in feature_configs.items():
    train_data = extract_fn(X_train_feat)
    val_data = extract_fn(X_val_feat)
    test_data = extract_fn(X_test_feat)
    for arch_name, build_fn in architectures.items():
        key = f"{feat_name} + {arch_name}"
        accs, f1s = [], []
        print(f"\n{'='*50}")
        print(f"Training: {key}")
        print(f"{'='*50}")
        for seed in SEEDS:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            scaler = StandardScaler()
            d = train_data.shape[-1]
            train_norm = scaler.fit_transform(train_data.reshape(-1, d)).reshape(N_train, T, d)
            val_norm = scaler.transform(val_data.reshape(-1, d)).reshape(N_val, T, d)
            test_norm = scaler.transform(test_data.reshape(-1, d)).reshape(N_test, T, d)
            model = build_fn(d)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(train_norm, y_train, validation_data=(val_norm, y_val),
                      epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
            pred = np.argmax(model.predict(test_norm, verbose=0), axis=1)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average='weighted')
            accs.append(acc)
            f1s.append(f1)
            print(f"  Seed {seed}: acc={acc:.4f}")
        results[key] = {
            'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
            'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s))
        }
        print(f"  Mean: {results[key]['acc_mean']:.4f} ± {results[key]['acc_std']:.4f}")

print("\n\n===== CROSS ANALYSIS SUMMARY =====")
print(f"{'Configuration':<30s} {'Accuracy':>12s} {'Weighted F1':>12s}")
print("-" * 55)
for key, res in results.items():
    print(f"{key:<30s} {res['acc_mean']:.4f}±{res['acc_std']:.4f}  {res['f1_mean']:.4f}±{res['f1_std']:.4f}")

with open('cross_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved to cross_analysis_results.json")
print("Done!")