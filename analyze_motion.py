import numpy as np
import os
import json

base = "processed_data"
X_train_feat = np.load(os.path.join(base, "X_train_feat.npy"))
X_val_feat = np.load(os.path.join(base, "X_val_feat.npy"))
X_test_feat = np.load(os.path.join(base, "X_test_feat.npy"))
y_train = np.load(os.path.join(base, "y_train_seq.npy"))
y_val = np.load(os.path.join(base, "y_val_seq.npy"))
y_test = np.load(os.path.join(base, "y_test_seq.npy"))

classes = [
    "WallPushups", "JumpingJack", "PushUps",
    "GolfSwing", "Lunges", "TennisSwing",
    "TaiChi", "PlayingGuitar", "BrushingTeeth"
]

angles_idx = list(range(8))
diff_idx = list(range(8, 16))

print("=" * 70)
print("MOTION MAGNITUDE ANALYSIS PER CLASS")
print("=" * 70)
print(f"{'Class':<20s} {'Diff Mean':>10s} {'Diff Std':>10s} {'Angles Std':>10s} {'Accuracy':>10s}")
print("-" * 60)

# 从 unified_results.json 中读取每类准确率
with open('unified_results.json', 'r') as f:
    unified = json.load(f)
best_cm = np.array(unified['best_confusion'])
per_class_correct = np.diag(best_cm)
per_class_total = best_cm.sum(axis=1)
per_class_acc = per_class_correct / per_class_total

for cls_idx, cls_name in enumerate(classes):
    mask = y_test == cls_idx
    cls_feat = X_test_feat[mask]
    diff_feat = cls_feat[:, :, diff_idx]
    angles_feat = cls_feat[:, :, angles_idx]

    diff_mean = np.mean(np.abs(diff_feat))
    diff_std = np.std(diff_feat)
    angles_std = np.std(angles_feat)

    print(f"{cls_name:<20s} {diff_mean:10.4f} {diff_std:10.4f} {angles_std:10.4f} {per_class_acc[cls_idx]:10.4f}")

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

diff_means = []
diff_stds = []
accs = []
for cls_idx in range(9):
    mask = y_test == cls_idx
    cls_feat = X_test_feat[mask]
    diff_feat = cls_feat[:, :, diff_idx]
    diff_means.append(np.mean(np.abs(diff_feat)))
    diff_stds.append(np.std(diff_feat))
    accs.append(per_class_acc[cls_idx])

print(f"Correlation: Diff Mean vs Accuracy = {np.corrcoef(diff_means, accs)[0,1]:.4f}")
print(f"Correlation: Diff Std vs Accuracy = {np.corrcoef(diff_stds, accs)[0,1]:.4f}")

# TaiChi vs JumpingJack 对比
tc_idx = classes.index("TaiChi")
jj_idx = classes.index("JumpingJack")
tc_feat = X_test_feat[y_test == tc_idx]
jj_feat = X_test_feat[y_test == jj_idx]
tc_diff = tc_feat[:, :, diff_idx]
jj_diff = jj_feat[:, :, diff_idx]
print(f"\nTaiChi diff mean: {np.mean(np.abs(tc_diff)):.4f}")
print(f"JumpingJack diff mean: {np.mean(np.abs(jj_diff)):.4f}")
print(f"Ratio (JJ/TC): {np.mean(np.abs(jj_diff))/np.mean(np.abs(tc_diff)):.2f}x")

print("\nDone!")