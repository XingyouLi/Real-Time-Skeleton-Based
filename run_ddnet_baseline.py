import numpy as np
import os
import json
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, GlobalAveragePooling1D, Dropout, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

base = "processed_data"
X_train_seq = np.load(os.path.join(base, "X_train_seq.npy"))
X_val_seq = np.load(os.path.join(base, "X_val_seq.npy"))
X_test_seq = np.load(os.path.join(base, "X_test_seq.npy"))
y_train = np.load(os.path.join(base, "y_train_seq.npy"))
y_val = np.load(os.path.join(base, "y_val_seq.npy"))
y_test = np.load(os.path.join(base, "y_test_seq.npy"))

N_train, T, K, _ = X_train_seq.shape
N_val = X_val_seq.shape[0]
N_test = X_test_seq.shape[0]
NUM_CLASSES = 9
SEEDS = [42, 123, 456, 789, 1024]

# DD-Net 输入：关节位置 (J×3) 和 关节速度 (J×3)
# 我们用33个关键点的(x,y)坐标，所以J=33, C=2
# 位置：每帧的(x,y)坐标，展平为66维
# 速度：相邻帧坐标差，展平为66维
J = 33
C = 2

def compute_position_and_velocity(sequences):
    N, T, J, D = sequences.shape
    pos = sequences[:, :, :, :2].reshape(N, T, -1)
    vel = np.zeros_like(pos)
    vel[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]
    return pos, vel

pos_train, vel_train = compute_position_and_velocity(X_train_seq)
pos_val, vel_val = compute_position_and_velocity(X_val_seq)
pos_test, vel_test = compute_position_and_velocity(X_test_seq)

input_dim = J * C  # 66

def build_ddnet():
    input_pos = Input(shape=(T, input_dim), name='input_pos')
    input_vel = Input(shape=(T, input_dim), name='input_vel')

    # 位置分支
    x_pos = Conv1D(64, kernel_size=3, padding='same')(input_pos)
    x_pos = BatchNormalization()(x_pos)
    x_pos = ReLU()(x_pos)
    x_pos = Conv1D(128, kernel_size=3, padding='same')(x_pos)
    x_pos = BatchNormalization()(x_pos)
    x_pos = ReLU()(x_pos)
    x_pos = GlobalAveragePooling1D()(x_pos)

    # 速度分支
    x_vel = Conv1D(64, kernel_size=3, padding='same')(input_vel)
    x_vel = BatchNormalization()(x_vel)
    x_vel = ReLU()(x_vel)
    x_vel = Conv1D(128, kernel_size=3, padding='same')(x_vel)
    x_vel = BatchNormalization()(x_vel)
    x_vel = ReLU()(x_vel)
    x_vel = GlobalAveragePooling1D()(x_vel)

    # 融合
    x = Concatenate()([x_pos, x_vel])
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=[input_pos, input_vel], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

ddnet_results = {'acc': [], 'f1': []}
ddnet_params = build_ddnet().count_params()

for seed in SEEDS:
    print(f"\nSeed {seed}")
    tf.random.set_seed(seed)
    np.random.seed(seed)

    scaler_pos = StandardScaler()
    scaler_vel = StandardScaler()
    pos_train_norm = scaler_pos.fit_transform(pos_train.reshape(-1, input_dim)).reshape(N_train, T, input_dim)
    pos_val_norm = scaler_pos.transform(pos_val.reshape(-1, input_dim)).reshape(N_val, T, input_dim)
    pos_test_norm = scaler_pos.transform(pos_test.reshape(-1, input_dim)).reshape(N_test, T, input_dim)
    vel_train_norm = scaler_vel.fit_transform(vel_train.reshape(-1, input_dim)).reshape(N_train, T, input_dim)
    vel_val_norm = scaler_vel.transform(vel_val.reshape(-1, input_dim)).reshape(N_val, T, input_dim)
    vel_test_norm = scaler_vel.transform(vel_test.reshape(-1, input_dim)).reshape(N_test, T, input_dim)

    model = build_ddnet()
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit([pos_train_norm, vel_train_norm], y_train,
              validation_data=([pos_val_norm, vel_val_norm], y_val),
              epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)

    pred = np.argmax(model.predict([pos_test_norm, vel_test_norm], verbose=0), axis=1)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='weighted')
    ddnet_results['acc'].append(acc)
    ddnet_results['f1'].append(f1)
    print(f"  DD-Net: acc={acc:.4f}")

# 推理速度测试
model = build_ddnet()
pos_sample = np.random.randn(32, T, input_dim).astype('float32')
vel_sample = np.random.randn(32, T, input_dim).astype('float32')
for _ in range(10):
    model([pos_sample, vel_sample], training=False)
start = time.time()
for _ in range(100):
    model([pos_sample, vel_sample], training=False)
ddnet_time = (time.time() - start) / 100 * 1000

acc_arr = np.array(ddnet_results['acc'])
f1_arr = np.array(ddnet_results['f1'])
print(f"\n===== DD-NET RESULTS =====")
print(f"Accuracy: {acc_arr.mean():.4f} ± {acc_arr.std():.4f}")
print(f"F1: {f1_arr.mean():.4f} ± {f1_arr.std():.4f}")
print(f"Parameters: {ddnet_params}")
print(f"Batch inference time: {ddnet_time:.2f} ms")

with open('ddnet_results.json', 'w') as f:
    json.dump({
        'acc_mean': float(acc_arr.mean()),
        'acc_std': float(acc_arr.std()),
        'f1_mean': float(f1_arr.mean()),
        'f1_std': float(f1_arr.std()),
        'params': ddnet_params,
        'time_ms': ddnet_time
    }, f, indent=2)
print("Saved to ddnet_results.json")