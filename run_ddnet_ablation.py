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
X_train_feat = np.load(os.path.join(base, "X_train_feat.npy"))
X_val_feat = np.load(os.path.join(base, "X_val_feat.npy"))
X_test_feat = np.load(os.path.join(base, "X_test_feat.npy"))
X_train_seq = np.load(os.path.join(base, "X_train_seq.npy"))
X_val_seq = np.load(os.path.join(base, "X_val_seq.npy"))
X_test_seq = np.load(os.path.join(base, "X_test_seq.npy"))
y_train = np.load(os.path.join(base, "y_train_seq.npy"))
y_val = np.load(os.path.join(base, "y_val_seq.npy"))
y_test = np.load(os.path.join(base, "y_test_seq.npy"))

N_train, T = X_train_feat.shape[0], X_train_feat.shape[1]
N_val = X_val_feat.shape[0]
N_test = X_test_feat.shape[0]
NUM_CLASSES = 9
SEEDS = [42, 123, 456, 789, 1024]

J, C = 33, 2
input_dim_raw = J * C

def compute_position_and_velocity(sequences):
    N_frames, T_f, J_f, D_f = sequences.shape
    pos = sequences[:, :, :, :2].reshape(N_frames, T_f, -1)
    vel = np.zeros_like(pos)
    vel[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]
    return pos, vel

pos_train, vel_train = compute_position_and_velocity(X_train_seq)
pos_val, vel_val = compute_position_and_velocity(X_val_seq)
pos_test, vel_test = compute_position_and_velocity(X_test_seq)

# 16维特征：角度+差分
angles_train = X_train_feat[:, :, :8]
diff_train = X_train_feat[:, :, 8:]
angles_val = X_val_feat[:, :, :8]
diff_val = X_val_feat[:, :, 8:]
angles_test = X_test_feat[:, :, :8]
diff_test = X_test_feat[:, :, 8:]

def build_ddnet_single_stream(input_dim, name_prefix):
    inp = Input(shape=(T, input_dim), name=f'{name_prefix}_input')
    x = Conv1D(64, kernel_size=3, padding='same', name=f'{name_prefix}_conv1')(inp)
    x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = ReLU(name=f'{name_prefix}_relu1')(x)
    x = Conv1D(128, kernel_size=3, padding='same', name=f'{name_prefix}_conv2')(x)
    x = BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = ReLU(name=f'{name_prefix}_relu2')(x)
    x = GlobalAveragePooling1D(name=f'{name_prefix}_pool')(x)
    return inp, x

def build_ddnet_variant(stream1_dim, stream2_dim, variant_name):
    inp1, x1 = build_ddnet_single_stream(stream1_dim, f'{variant_name}_s1')
    inp2, x2 = build_ddnet_single_stream(stream2_dim, f'{variant_name}_s2')
    x = Concatenate(name=f'{variant_name}_concat')([x1, x2])
    x = Dropout(0.3, name=f'{variant_name}_drop1')(x)
    x = Dense(128, activation='relu', name=f'{variant_name}_dense')(x)
    x = Dropout(0.3, name=f'{variant_name}_drop2')(x)
    output = Dense(NUM_CLASSES, activation='softmax', name=f'{variant_name}_out')(x)
    model = Model(inputs=[inp1, inp2], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

configs = {
    'DD-Net (Pos+Vel, 66-dim)': {
        'stream1_train': pos_train, 'stream1_val': pos_val, 'stream1_test': pos_test,
        'stream2_train': vel_train, 'stream2_val': vel_val, 'stream2_test': vel_test,
        'dim1': input_dim_raw, 'dim2': input_dim_raw,
        'name': 'posvel'
    },
    'DD-Net Angle+Diff (16-dim)': {
        'stream1_train': angles_train, 'stream1_val': angles_val, 'stream1_test': angles_test,
        'stream2_train': diff_train, 'stream2_val': diff_val, 'stream2_test': diff_test,
        'dim1': 8, 'dim2': 8,
        'name': 'anglediff'
    }
}

results = {}
for config_name, config in configs.items():
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"{'='*60}")
    accs, f1s = [], []
    for seed in SEEDS:
        tf.random.set_seed(seed)
        np.random.seed(seed)
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        s1_train = scaler1.fit_transform(config['stream1_train'].reshape(-1, config['dim1'])).reshape(N_train, T, config['dim1'])
        s1_val = scaler1.transform(config['stream1_val'].reshape(-1, config['dim1'])).reshape(N_val, T, config['dim1'])
        s1_test = scaler1.transform(config['stream1_test'].reshape(-1, config['dim1'])).reshape(N_test, T, config['dim1'])
        s2_train = scaler2.fit_transform(config['stream2_train'].reshape(-1, config['dim2'])).reshape(N_train, T, config['dim2'])
        s2_val = scaler2.transform(config['stream2_val'].reshape(-1, config['dim2'])).reshape(N_val, T, config['dim2'])
        s2_test = scaler2.transform(config['stream2_test'].reshape(-1, config['dim2'])).reshape(N_test, T, config['dim2'])
        model = build_ddnet_variant(config['dim1'], config['dim2'], config['name'])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit([s1_train, s2_train], y_train,
                  validation_data=([s1_val, s2_val], y_val),
                  epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
        pred = np.argmax(model.predict([s1_test, s2_test], verbose=0), axis=1)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average='weighted')
        accs.append(acc)
        f1s.append(f1)
        print(f"  Seed {seed}: acc={acc:.4f}")
    acc_arr = np.array(accs)
    f1_arr = np.array(f1s)
    results[config_name] = {
        'acc_mean': float(acc_arr.mean()),
        'acc_std': float(acc_arr.std()),
        'f1_mean': float(f1_arr.mean()),
        'f1_std': float(f1_arr.std())
    }
    print(f"  Mean: {acc_arr.mean():.4f} ± {acc_arr.std():.4f}")

# 汇总
print("\n\n===== DD-NET ABLATION RESULTS =====")
for name, res in results.items():
    print(f"{name}: Acc={res['acc_mean']:.4f}±{res['acc_std']:.4f}, F1={res['f1_mean']:.4f}±{res['f1_std']:.4f}")

# 关键对比
posvel_acc = results['DD-Net (Pos+Vel, 66-dim)']['acc_mean']
anglediff_acc = results['DD-Net Angle+Diff (16-dim)']['acc_mean']
gap_input = posvel_acc - anglediff_acc
print(f"\nGap due to input representation: {gap_input:.4f}")

# 与单流1D-CNN对比
# 之前的结果：16维融合特征1D-CNN = 0.8513
single_cnn_acc = 0.8513
gap_arch = anglediff_acc - single_cnn_acc
print(f"Gap due to architecture (DD-Net vs 1D-CNN on 16-dim): {gap_arch:.4f}")

summary = {k: v for k, v in results.items()}
summary['input_gap'] = float(gap_input)
summary['architecture_gap'] = float(gap_arch)
with open('ddnet_ablation_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved to ddnet_ablation_results.json")
print("Done!")