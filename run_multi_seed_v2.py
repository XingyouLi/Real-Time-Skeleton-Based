import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, GlobalAveragePooling1D, Dropout, Dense
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
N_val = X_val_feat.shape[0]
N_test = X_test_feat.shape[0]
NUM_CLASSES = 9

SEEDS = [42, 123, 456, 789, 1024]

results = {
    'angles_acc': [], 'angles_f1': [],
    'delta_acc': [], 'delta_f1': [],
    'fused_acc': [], 'fused_f1': [],
    'cnn_acc': [], 'cnn_f1': [],
    'tcn_acc': [], 'tcn_f1': [],
    'upper_acc': [], 'upper_f1': [],
    'lower_acc': [], 'lower_f1': [],
    'svm_acc': [], 'svm_f1': [],
    'rf_acc': [], 'rf_f1': [],
}
per_seed = {}
best_seed = None
best_fused_acc = 0.0
best_confusion = None

for seed in SEEDS:
    print(f"\n{'='*50}")
    print(f"Seed {seed}")
    print(f"{'='*50}")
    tf.random.set_seed(seed)
    np.random.seed(seed)
    per_seed[seed] = {}

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_feat.reshape(-1, D)).reshape(N_train, T, D)
    X_val_norm = scaler.transform(X_val_feat.reshape(-1, D)).reshape(N_val, T, D)
    X_test_norm = scaler.transform(X_test_feat.reshape(-1, D)).reshape(N_test, T, D)

    # ---------- LSTM (Angles+Delta) ----------
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(T, D)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_norm, y_train, validation_data=(X_val_norm, y_val),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
    pred = np.argmax(model.predict(X_test_norm, verbose=0), axis=1)
    fused_acc = accuracy_score(y_test, pred)
    fused_f1 = f1_score(y_test, pred, average='weighted')
    results['fused_acc'].append(fused_acc)
    results['fused_f1'].append(fused_f1)
    per_seed[seed]['fused_acc'] = float(fused_acc)
    if fused_acc > best_fused_acc:
        best_fused_acc = fused_acc
        best_seed = seed
        best_confusion = confusion_matrix(y_test, pred)
    print(f"  Angles+Delta (LSTM): acc={fused_acc:.4f}")

    # ---------- 1D-CNN ----------
    model_cnn = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=(T, D)),
        Conv1D(128, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop_cnn = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_cnn.fit(X_train_norm, y_train, validation_data=(X_val_norm, y_val),
                  epochs=50, batch_size=32, callbacks=[early_stop_cnn], verbose=0)
    pred_cnn = np.argmax(model_cnn.predict(X_test_norm, verbose=0), axis=1)
    cnn_acc = accuracy_score(y_test, pred_cnn)
    cnn_f1 = f1_score(y_test, pred_cnn, average='weighted')
    results['cnn_acc'].append(cnn_acc)
    results['cnn_f1'].append(cnn_f1)
    per_seed[seed]['cnn_acc'] = float(cnn_acc)
    print(f"  1D-CNN: acc={cnn_acc:.4f}")

    # ---------- TCN ----------
    model_tcn = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', padding='causal', input_shape=(T, D)),
        Conv1D(64, kernel_size=3, activation='relu', padding='causal', dilation_rate=2),
        Conv1D(64, kernel_size=3, activation='relu', padding='causal', dilation_rate=4),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model_tcn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop_tcn = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_tcn.fit(X_train_norm, y_train, validation_data=(X_val_norm, y_val),
                  epochs=50, batch_size=32, callbacks=[early_stop_tcn], verbose=0)
    pred_tcn = np.argmax(model_tcn.predict(X_test_norm, verbose=0), axis=1)
    tcn_acc = accuracy_score(y_test, pred_tcn)
    tcn_f1 = f1_score(y_test, pred_tcn, average='weighted')
    results['tcn_acc'].append(tcn_acc)
    results['tcn_f1'].append(tcn_f1)
    per_seed[seed]['tcn_acc'] = float(tcn_acc)
    print(f"  TCN: acc={tcn_acc:.4f}")

    # ---------- Angles Only ----------
    angles_train = X_train_norm[:, :, :8]
    angles_val = X_val_norm[:, :, :8]
    angles_test = X_test_norm[:, :, :8]
    model_a = Sequential([LSTM(64, return_sequences=True, input_shape=(T, 8)),
                          Dropout(0.3), LSTM(64), Dropout(0.3),
                          Dense(NUM_CLASSES, activation='softmax')])
    model_a.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop_a = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_a.fit(angles_train, y_train, validation_data=(angles_val, y_val),
                epochs=50, batch_size=32, callbacks=[early_stop_a], verbose=0)
    pred_a = np.argmax(model_a.predict(angles_test, verbose=0), axis=1)
    angles_acc = accuracy_score(y_test, pred_a)
    angles_f1 = f1_score(y_test, pred_a, average='weighted')
    results['angles_acc'].append(angles_acc)
    results['angles_f1'].append(angles_f1)
    per_seed[seed]['angles_acc'] = float(angles_acc)
    print(f"  Angles Only: acc={angles_acc:.4f}")

    # ---------- Delta Only ----------
    delta_train = X_train_norm[:, :, 8:]
    delta_val = X_val_norm[:, :, 8:]
    delta_test = X_test_norm[:, :, 8:]
    model_d = Sequential([LSTM(64, return_sequences=True, input_shape=(T, 8)),
                          Dropout(0.3), LSTM(64), Dropout(0.3),
                          Dense(NUM_CLASSES, activation='softmax')])
    model_d.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop_d = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_d.fit(delta_train, y_train, validation_data=(delta_val, y_val),
                epochs=50, batch_size=32, callbacks=[early_stop_d], verbose=0)
    pred_d = np.argmax(model_d.predict(delta_test, verbose=0), axis=1)
    delta_acc = accuracy_score(y_test, pred_d)
    delta_f1 = f1_score(y_test, pred_d, average='weighted')
    results['delta_acc'].append(delta_acc)
    results['delta_f1'].append(delta_f1)
    per_seed[seed]['delta_acc'] = float(delta_acc)
    print(f"  Delta Only: acc={delta_acc:.4f}")

    # ---------- Angle Subsets ----------
    # Upper Only (4 angles + 4 diffs = 8 dims)
    upper_train = np.concatenate([X_train_norm[:,:,:4], X_train_norm[:,:,8:12]], axis=-1)
    upper_val = np.concatenate([X_val_norm[:,:,:4], X_val_norm[:,:,8:12]], axis=-1)
    upper_test = np.concatenate([X_test_norm[:,:,:4], X_test_norm[:,:,8:12]], axis=-1)
    model_up = Sequential([LSTM(64, return_sequences=True, input_shape=(T, 8)),
                           Dropout(0.3), LSTM(64), Dropout(0.3),
                           Dense(NUM_CLASSES, activation='softmax')])
    model_up.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop_up = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_up.fit(upper_train, y_train, validation_data=(upper_val, y_val),
                 epochs=50, batch_size=32, callbacks=[early_stop_up], verbose=0)
    pred_up = np.argmax(model_up.predict(upper_test, verbose=0), axis=1)
    upper_acc = accuracy_score(y_test, pred_up)
    upper_f1 = f1_score(y_test, pred_up, average='weighted')
    results['upper_acc'].append(upper_acc)
    results['upper_f1'].append(upper_f1)
    per_seed[seed]['upper_acc'] = float(upper_acc)
    print(f"  Upper Only: acc={upper_acc:.4f}")

    # Lower Only (4 angles + 4 diffs = 8 dims)
    lower_train = np.concatenate([X_train_norm[:,:,4:8], X_train_norm[:,:,12:16]], axis=-1)
    lower_val = np.concatenate([X_val_norm[:,:,4:8], X_val_norm[:,:,12:16]], axis=-1)
    lower_test = np.concatenate([X_test_norm[:,:,4:8], X_test_norm[:,:,12:16]], axis=-1)
    model_lo = Sequential([LSTM(64, return_sequences=True, input_shape=(T, 8)),
                           Dropout(0.3), LSTM(64), Dropout(0.3),
                           Dense(NUM_CLASSES, activation='softmax')])
    model_lo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop_lo = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_lo.fit(lower_train, y_train, validation_data=(lower_val, y_val),
                 epochs=50, batch_size=32, callbacks=[early_stop_lo], verbose=0)
    pred_lo = np.argmax(model_lo.predict(lower_test, verbose=0), axis=1)
    lower_acc = accuracy_score(y_test, pred_lo)
    lower_f1 = f1_score(y_test, pred_lo, average='weighted')
    results['lower_acc'].append(lower_acc)
    results['lower_f1'].append(lower_f1)
    per_seed[seed]['lower_acc'] = float(lower_acc)
    print(f"  Lower Only: acc={lower_acc:.4f}")

    # ---------- SVM & RF ----------
    X_train_2d = X_train_feat.reshape(N_train, -1)
    X_test_2d = X_test_feat.reshape(N_test, -1)
    scaler_ml = StandardScaler()
    X_train_std = scaler_ml.fit_transform(X_train_2d)
    X_test_std = scaler_ml.transform(X_test_2d)

    svm = SVC(kernel='linear', C=1.0, random_state=seed)
    svm.fit(X_train_std, y_train)
    svm_pred = svm.predict(X_test_std)
    results['svm_acc'].append(accuracy_score(y_test, svm_pred))
    results['svm_f1'].append(f1_score(y_test, svm_pred, average='weighted'))

    rf = RandomForestClassifier(n_estimators=100, random_state=seed)
    rf.fit(X_train_std, y_train)
    rf_pred = rf.predict(X_test_std)
    results['rf_acc'].append(accuracy_score(y_test, rf_pred))
    results['rf_f1'].append(f1_score(y_test, rf_pred, average='weighted'))

# -------------------------------
summary = {}
for key in results:
    arr = np.array(results[key])
    summary[key] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr))}

print("\n\n===== UNIFIED FINAL RESULTS (Mean ± Std over 5 seeds) =====")
print(f"Angles Only:   Acc={summary['angles_acc']['mean']:.4f}±{summary['angles_acc']['std']:.4f}, F1={summary['angles_f1']['mean']:.4f}±{summary['angles_f1']['std']:.4f}")
print(f"Delta Only:    Acc={summary['delta_acc']['mean']:.4f}±{summary['delta_acc']['std']:.4f}, F1={summary['delta_f1']['mean']:.4f}±{summary['delta_f1']['std']:.4f}")
print(f"Angles+Delta:  Acc={summary['fused_acc']['mean']:.4f}±{summary['fused_acc']['std']:.4f}, F1={summary['fused_f1']['mean']:.4f}±{summary['fused_f1']['std']:.4f}")
print(f"Upper Only:    Acc={summary['upper_acc']['mean']:.4f}±{summary['upper_acc']['std']:.4f}, F1={summary['upper_f1']['mean']:.4f}±{summary['upper_f1']['std']:.4f}")
print(f"Lower Only:    Acc={summary['lower_acc']['mean']:.4f}±{summary['lower_acc']['std']:.4f}, F1={summary['lower_f1']['mean']:.4f}±{summary['lower_f1']['std']:.4f}")
print(f"1D-CNN:        Acc={summary['cnn_acc']['mean']:.4f}±{summary['cnn_acc']['std']:.4f}, F1={summary['cnn_f1']['mean']:.4f}±{summary['cnn_f1']['std']:.4f}")
print(f"TCN:           Acc={summary['tcn_acc']['mean']:.4f}±{summary['tcn_acc']['std']:.4f}, F1={summary['tcn_f1']['mean']:.4f}±{summary['tcn_f1']['std']:.4f}")
print(f"SVM:           Acc={summary['svm_acc']['mean']:.4f}±{summary['svm_acc']['std']:.4f}, F1={summary['svm_f1']['mean']:.4f}±{summary['svm_f1']['std']:.4f}")
print(f"Random Forest: Acc={summary['rf_acc']['mean']:.4f}±{summary['rf_acc']['std']:.4f}, F1={summary['rf_f1']['mean']:.4f}±{summary['rf_f1']['std']:.4f}")
print(f"\nBest seed for Angles+Delta LSTM: {best_seed} (acc={best_fused_acc:.4f})")

summary['best_seed'] = best_seed
summary['best_confusion'] = best_confusion.tolist()
summary['per_seed'] = {str(k): v for k, v in per_seed.items()}
with open('unified_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved to unified_results.json")