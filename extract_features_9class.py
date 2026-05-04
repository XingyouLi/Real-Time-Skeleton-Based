import numpy as np
import os

# -------------------------------
# 1. 加载原始序列
# -------------------------------
base = "processed_data"
X_train_seq = np.load(os.path.join(base, "X_train_seq.npy"))
X_val_seq = np.load(os.path.join(base, "X_val_seq.npy"))
X_test_seq = np.load(os.path.join(base, "X_test_seq.npy"))

print(f"Train: {X_train_seq.shape}, Val: {X_val_seq.shape}, Test: {X_test_seq.shape}")

# -------------------------------
# 2. 角度计算函数
# -------------------------------
def compute_angles_batch(sequence):
    N, T, K, D = sequence.shape
    angles = np.zeros((N, T, 8))
    L_SHOULDER, R_SHOULDER = 11, 12
    L_ELBOW, R_ELBOW = 13, 14
    L_WRIST, R_WRIST = 15, 16
    L_HIP, R_HIP = 23, 24
    L_KNEE, R_KNEE = 25, 26
    L_ANKLE, R_ANKLE = 27, 28

    for i in range(N):
        for t in range(T):
            def get(idx):
                return sequence[i, t, idx, :2]
            angles[i, t, 0] = angle_3pts(get(L_SHOULDER), get(L_ELBOW), get(L_WRIST))
            angles[i, t, 1] = angle_3pts(get(R_SHOULDER), get(R_ELBOW), get(R_WRIST))
            angles[i, t, 2] = angle_3pts(get(L_ELBOW), get(L_SHOULDER), get(L_HIP))
            angles[i, t, 3] = angle_3pts(get(R_ELBOW), get(R_SHOULDER), get(R_HIP))
            angles[i, t, 4] = angle_3pts(get(L_SHOULDER), get(L_HIP), get(L_KNEE))
            angles[i, t, 5] = angle_3pts(get(R_SHOULDER), get(R_HIP), get(R_KNEE))
            angles[i, t, 6] = angle_3pts(get(L_HIP), get(L_KNEE), get(L_ANKLE))
            angles[i, t, 7] = angle_3pts(get(R_HIP), get(R_KNEE), get(R_ANKLE))
    return angles

def angle_3pts(A, B, C):
    v1 = A - B
    v2 = C - B
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)

# -------------------------------
# 3. 特征提取
# -------------------------------
def extract_features(sequences):
    angles = compute_angles_batch(sequences)
    diff = np.zeros_like(angles)
    diff[:, 1:, :] = angles[:, 1:, :] - angles[:, :-1, :]
    return np.concatenate([angles, diff], axis=-1)

print("Extracting features...")
X_train_feat = extract_features(X_train_seq)
X_val_feat = extract_features(X_val_seq)
X_test_feat = extract_features(X_test_seq)

print(f"Train feat: {X_train_feat.shape}")
print(f"Val feat: {X_val_feat.shape}")
print(f"Test feat: {X_test_feat.shape}")

np.save(os.path.join(base, "X_train_feat.npy"), X_train_feat)
np.save(os.path.join(base, "X_val_feat.npy"), X_val_feat)
np.save(os.path.join(base, "X_test_feat.npy"), X_test_feat)
print("Saved to processed_data/")
print("Done!")