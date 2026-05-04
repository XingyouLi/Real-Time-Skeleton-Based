import numpy as np
import os
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

SEQ_LEN = 30
STRIDE = 15
DATA_DIR = "results"
NUM_CLASSES = 9
SEEDS = [42, 123, 456, 789, 1024]
CV_SPLITS = [42, 123, 456]

classes = [
    "WallPushups", "JumpingJack", "PushUps",
    "GolfSwing", "Lunges", "TennisSwing",
    "TaiChi", "PlayingGuitar", "BrushingTeeth"
]

def build_sequences_for_classes(classes, data_dir):
    video_data = []
    labels = []
    video_names_list = []
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls_name)
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            frames = np.load(fpath)
            if frames.ndim != 3 or frames.shape[1:] != (33, 4):
                continue
            n_frames = frames.shape[0]
            if n_frames < SEQ_LEN:
                continue
            for start in range(0, n_frames - SEQ_LEN + 1, STRIDE):
                seq = frames[start:start + SEQ_LEN]
                video_data.append(seq)
                labels.append(cls_idx)
            n_seqs = (n_frames - SEQ_LEN) // STRIDE + 1
            video_names_list.extend([fname] * n_seqs)
    return np.array(video_data), np.array(labels), np.array(video_names_list)

def extract_16d_features(sequences):
    N, T, K, D = sequences.shape
    L_S, R_S = 11, 12
    L_E, R_E = 13, 14
    L_W, R_W = 15, 16
    L_H, R_H = 23, 24
    L_K, R_K = 25, 26
    L_A, R_A = 27, 28
    angles = np.zeros((N, T, 8))
    for i in range(N):
        for t in range(T):
            def g(idx):
                return sequences[i, t, idx, :2]
            def ang(A, B, C):
                v1 = A - B
                v2 = C - B
                cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos = np.clip(cos, -1.0, 1.0)
                return np.arccos(cos)
            angles[i, t, 0] = ang(g(L_S), g(L_E), g(L_W))
            angles[i, t, 1] = ang(g(R_S), g(R_E), g(R_W))
            angles[i, t, 2] = ang(g(L_E), g(L_S), g(L_H))
            angles[i, t, 3] = ang(g(R_E), g(R_S), g(R_H))
            angles[i, t, 4] = ang(g(L_S), g(L_H), g(L_K))
            angles[i, t, 5] = ang(g(R_S), g(R_H), g(R_K))
            angles[i, t, 6] = ang(g(L_H), g(L_K), g(L_A))
            angles[i, t, 7] = ang(g(R_H), g(R_K), g(R_A))
    diff = np.zeros_like(angles)
    diff[:, 1:, :] = angles[:, 1:, :] - angles[:, :-1, :]
    return np.concatenate([angles, diff], axis=-1)

def build_cnn(input_dim):
    model = Sequential([
        Conv1D(64, kernel_size=5, activation='relu', input_shape=(30, input_dim)),
        Conv1D(128, kernel_size=3, activation='relu'),
        GlobalAveragePooling1D(),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_dim):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(30, input_dim)),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def run_model(train_data, val_data, test_data, model_type, seed, input_dim):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler()
    train_norm = scaler.fit_transform(train_data.reshape(-1, input_dim)).reshape(-1, 30, input_dim)
    val_norm = scaler.transform(val_data.reshape(-1, input_dim)).reshape(-1, 30, input_dim)
    test_norm = scaler.transform(test_data.reshape(-1, input_dim)).reshape(-1, 30, input_dim)
    if model_type == 'cnn':
        model = build_cnn(input_dim)
    else:
        model = build_lstm(input_dim)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(train_norm, train_labels, validation_data=(val_norm, val_labels),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
    pred = np.argmax(model.predict(test_norm, verbose=0), axis=1)
    acc = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred, average='weighted')
    return acc, f1

# 加载全量原始序列
print("加载全量序列数据...")
all_X, all_y, all_names = build_sequences_for_classes(classes, DATA_DIR)
all_raw = all_X[:, :, :, :2].reshape(all_X.shape[0], all_X.shape[1], -1)
all_feat = extract_16d_features(all_X)
print(f"总序列数: {all_X.shape[0]}, 原始坐标维度: {all_raw.shape[-1]}, 融合特征维度: {all_feat.shape[-1]}")

cv_results = {}
for cv_idx, split_seed in enumerate(CV_SPLITS):
    print(f"\n{'='*60}")
    print(f"交叉验证 第{cv_idx+1}/3折 (split seed={split_seed})")
    print(f"{'='*60}")

    unique_videos = np.unique(all_names)
    train_vids, temp_vids = train_test_split(unique_videos, test_size=0.3, random_state=split_seed)
    val_vids, test_vids = train_test_split(temp_vids, test_size=0.5, random_state=split_seed)
    print(f"训练视频: {len(train_vids)}, 验证视频: {len(val_vids)}, 测试视频: {len(test_vids)}")

    train_mask = np.isin(all_names, train_vids)
    val_mask = np.isin(all_names, val_vids)
    test_mask = np.isin(all_names, test_vids)

    global train_labels, val_labels, test_labels
    train_labels = all_y[train_mask]
    val_labels = all_y[val_mask]
    test_labels = all_y[test_mask]

    raw_train = all_raw[train_mask]
    raw_val = all_raw[val_mask]
    raw_test = all_raw[test_mask]
    feat_train = all_feat[train_mask]
    feat_val = all_feat[val_mask]
    feat_test = all_feat[test_mask]

    fold_results = {
        'raw_cnn_acc': [], 'raw_lstm_acc': [],
        'feat_cnn_acc': [], 'feat_lstm_acc': [],
    }

    for seed in SEEDS:
        print(f"  Seed {seed}...", end=' ', flush=True)

        acc, _ = run_model(raw_train, raw_val, raw_test, 'cnn', seed, raw_train.shape[-1])
        fold_results['raw_cnn_acc'].append(acc)

        acc, _ = run_model(raw_train, raw_val, raw_test, 'lstm', seed, raw_train.shape[-1])
        fold_results['raw_lstm_acc'].append(acc)

        acc, _ = run_model(feat_train, feat_val, feat_test, 'cnn', seed, feat_train.shape[-1])
        fold_results['feat_cnn_acc'].append(acc)

        acc, _ = run_model(feat_train, feat_val, feat_test, 'lstm', seed, feat_train.shape[-1])
        fold_results['feat_lstm_acc'].append(acc)

        print(f"raw_cnn={fold_results['raw_cnn_acc'][-1]:.4f}, "
              f"raw_lstm={fold_results['raw_lstm_acc'][-1]:.4f}, "
              f"feat_cnn={fold_results['feat_cnn_acc'][-1]:.4f}, "
              f"feat_lstm={fold_results['feat_lstm_acc'][-1]:.4f}")

    cv_results[f"split_{split_seed}"] = fold_results

# 汇总
print("\n\n===== 交叉验证最终结果 =====")
summary = {}
for key in ['raw_cnn_acc', 'raw_lstm_acc', 'feat_cnn_acc', 'feat_lstm_acc']:
    all_vals = []
    for split_key in cv_results:
        all_vals.extend(cv_results[split_key][key])
    arr = np.array(all_vals)
    summary[key] = {'mean': float(np.mean(arr)), 'std': float(np.std(arr))}
    print(f"{key}: {summary[key]['mean']:.4f} ± {summary[key]['std']:.4f}")

summary['cv_results'] = cv_results
with open('cv_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved to cv_results.json")
print("Done!")