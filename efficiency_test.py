import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, GlobalAveragePooling1D, Dropout, Dense
import time
import numpy as np

T, D = 30, 16
NUM_CLASSES = 9
x = np.random.randn(32, T, D).astype('float32')

# ---------- LSTM ----------
lstm = Sequential([
    LSTM(64, return_sequences=True, input_shape=(T, D)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])
lstm_params = lstm.count_params()

for _ in range(5):
    lstm(x, training=False)

start = time.time()
for _ in range(100):
    lstm(x, training=False)
lstm_time = (time.time() - start) / 100 * 1000

# ---------- 1D-CNN ----------
cnn = Sequential([
    Conv1D(64, kernel_size=5, activation='relu', input_shape=(T, D)),
    Conv1D(128, kernel_size=3, activation='relu'),
    GlobalAveragePooling1D(),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])
cnn_params = cnn.count_params()

for _ in range(5):
    cnn(x, training=False)

start = time.time()
for _ in range(100):
    cnn(x, training=False)
cnn_time = (time.time() - start) / 100 * 1000

# ---------- TCN ----------
tcn = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', padding='causal', input_shape=(T, D)),
    Conv1D(64, kernel_size=3, activation='relu', padding='causal', dilation_rate=2),
    Conv1D(64, kernel_size=3, activation='relu', padding='causal', dilation_rate=4),
    GlobalAveragePooling1D(),
    Dropout(0.3),
    Dense(NUM_CLASSES, activation='softmax')
])
tcn_params = tcn.count_params()

for _ in range(5):
    tcn(x, training=False)

start = time.time()
for _ in range(100):
    tcn(x, training=False)
tcn_time = (time.time() - start) / 100 * 1000

# ---------- 输出 ----------
print(f"{'Model':<10} {'Params':<12} {'Time (ms/batch)'}")
print("-" * 40)
print(f"{'LSTM':<10} {lstm_params:<12} {lstm_time:.2f}")
print(f"{'1D-CNN':<10} {cnn_params:<12} {cnn_time:.2f}")
print(f"{'TCN':<10} {tcn_params:<12} {tcn_time:.2f}")