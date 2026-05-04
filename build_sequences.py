import numpy as np
import os
from sklearn.model_selection import train_test_split

SEQ_LEN = 30
STRIDE = 15
DATA_DIR = "results"
OUT_DIR = "processed_data"

classes = [
    "WallPushups",
    "JumpingJack",
    "PushUps",
    "GolfSwing",
    "Lunges",
    "TennisSwing",
    "TaiChi",
    "PlayingGuitar",
    "BrushingTeeth"
]

video_data = []
labels = []
skipped = []

for cls_idx, cls_name in enumerate(classes):
    cls_dir = os.path.join(DATA_DIR, cls_name)
    files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
    print(f"{cls_name}: {len(files)} files")
    for fname in files:
        fpath = os.path.join(cls_dir, fname)
        frames = np.load(fpath)
        if frames.ndim != 3 or frames.shape[1:] != (33, 4):
            print(f"  Skipping {fname}: shape {frames.shape} (not raw keypoints)")
            skipped.append(fpath)
            continue
        n_frames = frames.shape[0]
        if n_frames < SEQ_LEN:
            print(f"  Skipping {fname}: {n_frames} frames < {SEQ_LEN}")
            skipped.append(fpath)
            continue
        for start in range(0, n_frames - SEQ_LEN + 1, STRIDE):
            seq = frames[start:start + SEQ_LEN]
            video_data.append(seq)
            labels.append(cls_idx)

print(f"\nSkipped {len(skipped)} invalid files:")
for s in skipped:
    print(f"  {s}")

X = np.array(video_data)
y = np.array(labels)
print(f"\nTotal valid sequences: {X.shape}")

video_names = []
for cls_name in classes:
    cls_dir = os.path.join(DATA_DIR, cls_name)
    files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".npy")])
    for fname in files:
        fpath = os.path.join(cls_dir, fname)
        frames = np.load(fpath)
        if frames.ndim != 3 or frames.shape[1:] != (33, 4):
            continue
        n_frames = frames.shape[0]
        n_seqs = (n_frames - SEQ_LEN) // STRIDE + 1 if n_frames >= SEQ_LEN else 0
        video_names.extend([fname] * n_seqs)

video_names = np.array(video_names)
unique_videos = np.unique(video_names)
train_vids, temp_vids = train_test_split(unique_videos, test_size=0.3, random_state=42)
val_vids, test_vids = train_test_split(temp_vids, test_size=0.5, random_state=42)
print(f"Train videos: {len(train_vids)}, Val videos: {len(val_vids)}, Test videos: {len(test_vids)}")

train_mask = np.isin(video_names, train_vids)
val_mask = np.isin(video_names, val_vids)
test_mask = np.isin(video_names, test_vids)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

os.makedirs(OUT_DIR, exist_ok=True)
np.save(os.path.join(OUT_DIR, "X_train_seq.npy"), X_train)
np.save(os.path.join(OUT_DIR, "X_val_seq.npy"), X_val)
np.save(os.path.join(OUT_DIR, "X_test_seq.npy"), X_test)
np.save(os.path.join(OUT_DIR, "y_train_seq.npy"), y_train)
np.save(os.path.join(OUT_DIR, "y_val_seq.npy"), y_val)
np.save(os.path.join(OUT_DIR, "y_test_seq.npy"), y_test)
print("Saved to processed_data/")
print("Done!")