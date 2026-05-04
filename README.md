# ATDF-Net: Real-Time Skeleton-Based Action Recognition on Consumer CPUs

This repository contains the official implementation and experimental data for the paper:

**"Real-Time Skeleton-Based Action Recognition on Consumer CPUs: Latency Analysis, System Bottleneck Characterization, and Feature Contribution Benchmarking"**

Submitted to the *Journal of Real-Time Image Processing*.

## Overview

We present a systematic empirical study of lightweight skeleton-based action recognition under CPU-only deployment constraints. The system fuses spatial joint angles and frame-to-frame angle differences into a compact 16-dimensional vector processed by small sequential networks (MLP, LSTM, 1D-CNN, TCN). On a nine-class UCF101 subset with MediaPipe Pose as the visual front-end, the 1D-CNN pipeline achieves 0.8514 accuracy at 116 FPS on a consumer-grade CPU.

## Key Findings

- **Latency bottleneck:** MediaPipe keypoint extraction consumes 84.5% of per-frame latency (8.2 ms). The classifier requires only 1.1 ms.
- **Feature contribution:** Static joint angles carry the dominant discriminative information. Adding temporal difference features provides at most a 1.6 percentage point gain, which does not reach statistical significance.
- **Streaming latency:** Median first-correct response delay is 18 frames (600 ms). Post-first-correct prediction stability is 96.7%.
- **Data redundancy:** Overlapping sliding windows do not systematically inflate performance (at most 0.4 percentage point difference on 1D-CNN).

## Repository Structure

### Core Pipeline Scripts

| File | Description |
|------|-------------|
| `build_sequences.py` | Construct 30-frame sliding window sequences from raw keypoints |
| `extract_features_9class.py` | Compute 8 joint angles + 8 temporal differences (16-dim features) |
| `run_multi_seed_v2.py` | Unified training script: ablation, angle subsets, classifier comparison (5 seeds) |

### Analysis and Control Experiments

| File | Description |
|------|-------------|
| `run_architecture_feature_cross.py` | Architecture × feature cross-analysis (MLP, LSTM, 1D-CNN, TCN × Angles, Delta, Fused) |
| `run_cross_validation.py` | 3-fold cross-validation on core configurations |
| `run_ddnet_baseline.py` | DD-Net baseline implementation |
| `run_ddnet_ablation.py` | DD-Net input ablation (Pos+Vel vs. Angle+Diff) |
| `run_no_overlap_control.py` | Data redundancy control: non-overlapping vs. overlapping windows |
| `run_latency_response.py` | Streaming inference latency characterization |
| `run_stability_analysis.py` | Post-first-correct prediction stability analysis |
| `analyze_motion.py` | Per-class motion magnitude analysis |
| `efficiency_test.py` | Model parameter count and inference time measurement |

### Preprocessing

| File | Description |
|------|-------------|
| `src/extract_all_keypoints.py` | Extract 33 MediaPipe keypoints from raw videos |

### Experimental Results (JSON)

| File | Description |
|------|-------------|
| `unified_results.json` | Primary experiment: ablation, angle subsets, classifier comparison |
| `cross_analysis_results.json` | Architecture × feature cross-analysis results |
| `cv_results.json` | 3-fold cross-validation results |
| `ddnet_results.json` | DD-Net baseline results |
| `ddnet_ablation_results.json` | DD-Net input ablation results |
| `no_overlap_results.json` | Non-overlapping vs. overlapping window comparison |
| `latency_response_results.json` | Streaming inference latency characterization |
| `stability_analysis_results.json` | Post-first-correct prediction stability |

### Figures

| File | Description |
|------|-------------|
| `photo/figure1_pipeline.png` | System architecture diagram |
| `photo/angle_illustration.png` | Eight joint angles illustration |
| `photo/feature_fusion_pipeline.png` | Feature fusion pipeline diagram |

### Processed Data

The `processed_data/` directory contains pre-extracted sequences and features for the nine-class UCF101 subset used in all experiments:

- `X_train_seq.npy`, `X_val_seq.npy`, `X_test_seq.npy` — Raw keypoint sequences (shape: N × 30 × 33 × 4)
- `X_train_feat.npy`, `X_val_feat.npy`, `X_test_feat.npy` — 16-dimensional fused features (shape: N × 30 × 16)
- `y_train_seq.npy`, `y_val_seq.npy`, `y_test_seq.npy` — Class labels

Data split: 70% train / 15% validation / 15% test, split by video identity.

## Dataset

The UCF101 dataset is publicly available at:
[https://www.crcv.ucf.edu/data/UCF101.php](https://www.crcv.ucf.edu/data/UCF101.php)

The nine action classes used in this study are: WallPushups, JumpingJack, PushUps, GolfSwing, Lunges, TennisSwing, TaiChi, PlayingGuitar, BrushingTeeth.

## Environment

- Python 3.9+
- TensorFlow 2.x
- MediaPipe 0.10.9
- NumPy, SciPy, scikit-learn, matplotlib, seaborn

All experiments run on an Intel Core i5-8300H CPU without GPU acceleration.

## Hardware Specifications

Tests were conducted on the following consumer-grade laptop:

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i5-8300H @ 2.30 GHz (4 cores, 8 threads) |
| RAM | 16 GB DDR4 |
| Storage | 477 GB SSD |
| GPU | NVIDIA GeForce GTX 1060 (6 GB, not used) |

## License

MIT License. See `LICENSE` file for details.