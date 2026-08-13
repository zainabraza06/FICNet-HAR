<h1 align="center">
  🏃 FALL-HAR: Hierarchical Fall Detection &amp; Activity Recognition
</h1>

<p align="center">
  <em>A professional, modular, local-ready codebase for multi-stage wearable sensor classification using the MobiAct Dataset v2.0</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white" alt="sklearn"/>
  <img src="https://img.shields.io/badge/LOSO-Validated-green" alt="LOSO"/>
  <img src="https://img.shields.io/badge/Edge-Profiled-blueviolet" alt="Edge"/>
</p>

---

## 📖 Overview

This repository implements a **three-stage hierarchical classification system** for fall detection and activity recognition using inertial measurement unit (IMU) data from the [MobiAct Dataset v2.0](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/). The system is designed for deployment on resource-constrained wearable / edge devices.

### The Three-Stage Architecture

```
Raw IMU Signal (Accelerometer + Gyroscope)
        │
        ▼
┌──────────────────────────┐
│  STAGE 1: Fall Gate      │  Binary: FALL vs ADL
│  SVM-RBF / Fusion        │  Acc ≈ 0.9869
└──────────┬───────────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
┌─────────┐  ┌──────────────────────────┐
│FALL     │  │  ADL                     │
│         │  │                          │
└────┬────┘  └──────────┬───────────────┘
     │                  │
     ▼                  ▼
┌──────────────┐  ┌──────────────────────────┐
│ STAGE 2a     │  │ STAGE 2b                 │
│ Fall Subtypes│  │ ADL Classification       │
│ 4-class      │  │ 11-class                 │
│ Acc ≈ 0.8609 │  │ Acc ≈ 0.9370             │
└──────────────┘  └──────────────────────────┘
```

### Key Features

- **Hierarchical gating** with SVM confidence thresholds (0.70–0.95)
- **Dual-branch Fusion Network** combining BiLSTM temporal modelling with hand-crafted flat feature vectors
- **Cross-domain features**: Temporal Attention Entropy, Relative Position Encoding (RPE), Multi-scale FPN, Kalman Sensor Fusion
- **Streaming evaluation** with window purity stratification (pure vs. boundary windows)
- **Complete ablation studies** across feature sets, model architectures, and cross-domain additions
- **Edge profiling** — CPU inference latency, model size, throughput, and power estimates
- **Leave-One-Subject-Out (LOSO)** cross-validation throughout

---

## 📂 Repository Structure

```
FALL-HAR/
│
├── data/
│   └── MobiAct_Dataset_v2.0/
│       └── Annotated Data/
│           ├── BSC/          # Back-stroke to the chair (Fall)
│           ├── FKL/          # Forward-kneeling (Fall)
│           ├── FOL/          # Forward-falling (Fall)
│           ├── SDL/          # Side-lying (Fall)
│           ├── STD/          # Standing (ADL)
│           ├── WAL/          # Walking (ADL)
│           ├── JOG/          # Jogging (ADL)
│           ├── JUM/          # Jumping (ADL)
│           ├── STU/          # Stairs up (ADL)
│           ├── STN/          # Stairs down (ADL)
│           ├── SCH/          # Bending to pick object (ADL)
│           ├── SIT/          # Sitting on chair (ADL)
│           ├── CHU/          # Car getting in (ADL)
│           ├── CSI/          # Car sitting (ADL)
│           └── CSO/          # Car getting out (ADL)
│
├── src/                          # 🔬 Core library
│   ├── __init__.py
│   ├── data_loader.py            # File loading, segmentation, streaming builder, static dataset builders
│   ├── features.py               # Unified feature extractor with all flags
│   ├── models.py                 # SimpleBiLSTM, DualBranchFusionNet, StreamingFusionNet, classical models
│   └── evaluation.py             # LOSO runners, purity metrics, hierarchical helpers
│
├── scripts/                      # 🚀 Runnable pipelines
│   ├── run_stage1_ablations.py   # Stage 1: Binary Fall vs ADL — full ablation
│   ├── run_stage2a_ablations.py  # Stage 2a: Fall Subtypes (4-class) — full ablation
│   ├── run_stage2b_ablations.py  # Stage 2b: ADL Classification (11-class) — full ablation
│   ├── run_streaming_ablations.py# Streaming: purity-stratified window ablation
│   ├── run_hierarchical_ablations.py # End-to-end confidence gating ablation
│   └── run_edge_analysis.py      # Edge profiling: CPU latency, model size, paper tables
│
├── results/                      # 📊 Auto-generated output (CSV + JSON)
│   ├── stage1_all_ablations/
│   ├── stage2a_all_ablations/
│   ├── stage2b_all_ablations/
│   ├── streaming_ablation/
│   ├── hierarchical/
│   └── edge_analysis/
│
├── configs/                      # Hyperparameter / path configs (optional overrides)
├── tests/                        # Smoke tests
├── smoke_test.py                 # Quick sanity check before full run
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| **Name** | MobiAct Dataset v2.0 |
| **Subjects** | 67 |
| **Fall types** | 4 (BSC, FKL, FOL, SDL) |
| **ADL types** | 11 (STD, WAL, JOG, JUM, STU, STN, SCH, SIT, CHU, CSI, CSO) |
| **Sensors** | Accelerometer, Gyroscope, Orientation (pitch, roll) |
| **Sampling rate** | 87 Hz |
| **Format** | Per-subject CSV (`{CODE}_{SUBJ}_{TRIAL}_annotated.csv`) |

**Download:** [MobiAct Dataset v2.0](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/)

---

## ⚙️ Setup

### 1. Clone the Repository
```bash
git clone https://github.com/zainabraza06/FALL-HAR.git
cd FALL-HAR
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

The full dependency list:

```
numpy
pandas
scipy
scikit-learn
torch
```

For CUDA-accelerated training, install the correct PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Place the Dataset

Extract the MobiAct Dataset v2.0 archive and place it so the path resolves to:

```
data/MobiAct_Dataset_v2.0/Annotated Data/BSC/BSC_1_1_annotated.csv
```

The `src/data_loader.py` automatically resolves paths relative to the project root — **no hardcoded paths are needed**.

### 5. Verify Setup
```bash
python smoke_test.py
```

This runs a quick sanity check to confirm the dataset is reachable and the feature pipeline is functional.

---

## 🚀 Running the Pipelines

All scripts are run from the **project root directory**.

### Stage 1 — Binary Fall Detection
```bash
python scripts/run_stage1_ablations.py
```
**What it does:**
- Ablates 9 feature set configurations (Profile only → Full + Cross-domain)
- Compares 6 models: LDA, KNN-3, SVM-RBF, RandomForest, BiLSTM, Fusion
- Performs statistical significance testing (paired t-test, 5 seeds)
- Reports gyroscope contribution delta
- **Outputs:** `results/stage1_all_ablations/`

---

### Stage 2a — Fall Subtype Classification (4-class)
```bash
python scripts/run_stage2a_ablations.py
```
**What it does:**
- Classifies BSC / FKL / FOL / SDL using fall-specific peak features (onset slope, settle time, secondary peaks)
- Full model and feature set ablation
- Per-class performance breakdown
- **Outputs:** `results/stage2a_all_ablations/`

---

### Stage 2b — ADL Classification (11-class)
```bash
python scripts/run_stage2b_ablations.py
```
**What it does:**
- Classifies 11 daily activities using spectral (Welch PSD) + gyroscopic features
- Gyroscope and orientation contribution ablation
- Statistical significance testing
- **Outputs:** `results/stage2b_all_ablations/`

---

### Streaming Evaluation
```bash
python scripts/run_streaming_ablations.py
```
**What it does:**
- Constructs sliding windows (`window=200 samples`, `stride=100`) over raw continuous sensor streams
- Labels each window by majority vote and tracks **purity** (fraction of homogeneous samples)
- Compares a Naive Classical baseline (Random Forest, no memory) against the `StreamingFusionNet` (dual-LSTM)
- Reports accuracy stratified by **pure windows** (≥ 90% purity) vs. **boundary windows** (< 90%)
- **Outputs:** `results/streaming_ablation/`

---

### Hierarchical System Evaluation
```bash
python scripts/run_hierarchical_ablations.py
```
**What it does:**
- Assembles the full three-stage pipeline end-to-end
- Applies **confidence gating** at Stage 1: only routes high-confidence predictions into Stage 2; low-confidence predictions remain at binary output
- Ablates 6 confidence thresholds: 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
- Reports weighted hierarchical accuracy across all combinations
- **Outputs:** `results/hierarchical/`

---

### Edge Deployment Analysis
```bash
python scripts/run_edge_analysis.py
```
**What it does:**
- Trains all best models on **CPU only** (fair edge simulation)
- Measures inference latency (500 runs, warm-up of 50) using `time.perf_counter`
- Calculates model size in MB (parameter count × 4 bytes for PyTorch; pickle dump for sklearn)
- Estimates throughput (samples/sec) and power consumption tier (mW)
- Generates **paper-ready Table 1** (accuracy) and **Table 2** (edge metrics)
- **Outputs:** `results/edge_analysis/`

---

## 🔬 Model Architecture

### Dual-Branch Fusion Network

The core deep learning model used across all stages combines temporal and hand-crafted features via cross-attention:

```
Binned IMU Sequence       Flat Feature Vector
  (B × N_bins × D)          (B × F)
        │                       │
   BiLSTM                    MLP + BN
        │                       │
   DL Representation         ML Representation
        │                       │
        └──────┬────────────────┘
               │
         Cross-Attention
         (DL attends ML, ML attends DL)
               │
          Concatenate
               │
          Classifier
```

### Feature Sets

| Feature Group | Dims | Stages | Description |
|---------------|------|--------|-------------|
| Binned Profile | 15 | All | Per-axis energy distribution across 5 temporal bins |
| Accelerometer Stats | 5 | S1, S2a | Skewness, kurtosis, variance, mean, max jerk |
| Spectral (Welch PSD) | 6 | S2b, Streaming | Dominant frequency, power concentration |
| Fall-Specific | 10 | S2a | Onset slope, peak axis, settle time, secondary peaks |
| Gyroscope | 8 | S2b, Streaming | Angular energy distribution, autocorrelation |
| **Temporal Attention Entropy** | 3 | All | Energy concentration across sub-segments |
| **Relative Position Encoding (RPE)** | 10 | All | Lag-autocorrelation, weighted positional statistics |
| **Multi-scale FPN** | 10 | Optional | Features at 1×, 2×, 4× downsampling |
| **Kalman Sensor Fusion** | 10 | Optional | Fused acc+gyro signal with adaptive gain |

### Classical Baselines

| Model | Notes |
|-------|-------|
| LDA | Linear Discriminant Analysis with uniform class priors |
| KNN-3 | k=3, distance-weighted |
| SVM-RBF | RBF kernel, C=1.0, γ=scale, probability outputs |
| RandomForest | 200 trees, max depth 8 |

---

## 📊 Results Summary

### Overall Performance (LOSO Cross-Validation)

| Stage | Task | Best Model | Accuracy | Balanced Acc | Macro F1 |
|-------|------|-----------|----------|-------------|---------|
| Stage 1 | Binary Fall vs ADL | SVM-RBF | **0.9869** | 0.9840 | 0.9845 |
| Stage 2a | Fall Subtypes (4-class) | Fusion | **0.8609** | 0.8609 | 0.8605 |
| Stage 2b | ADL Classification (11-class) | Fusion | **0.9370** | 0.9352 | 0.9385 |
| Streaming | Windowed ADL | StreamingFusionNet | **0.9270** | 0.9000 | — |

### Edge Deployment (CPU Simulation)

| Stage | Model | Inference (ms/100) | Size (MB) | Throughput | Power |
|-------|-------|-------------------|-----------|------------|-------|
| Stage 1 | SVM-RBF | < 1 ms | < 0.5 MB | High | 50 mW |
| Stage 2a | Fusion | < 5 ms | < 1 MB | Medium | 100 mW |
| Stage 2b | Fusion | < 5 ms | < 1 MB | Medium | 100 mW |
| Streaming | StreamingFusionNet | < 10 ms | < 1 MB | Medium | 200 mW |

### Cross-Domain Feature Impact (Stage 1)

| Addition | Accuracy Δ |
|----------|-----------|
| + Temporal Attention | +0.8% |
| + RPE | +0.6% |
| + Attention + RPE | **+1.2%** |
| + FPN | +0.3% |
| + Kalman | +0.4% |

---

## 🧩 Module Reference

### `src/data_loader.py`

| Function | Description |
|----------|-------------|
| `get_segment(code, subj, trial)` | Load and return the labelled segment for an activity |
| `build_stage1_dataset(all_codes, fall_codes)` | Build binary Fall/ADL dataset arrays |
| `build_stage2a_dataset(fall_codes)` | Build fall subtype dataset arrays |
| `build_stage2b_dataset(adl_codes)` | Build ADL 11-class dataset arrays |
| `build_subject_stream(codes, subject, ...)` | Build a continuous sensor stream for one subject |
| `window_stream(acc, gyro, ...)` | Slide windows over a stream and compute purity |
| `build_streaming_dataset(codes, subjects, ...)` | Build windowed streaming dataset for all subjects |

### `src/features.py`

| Function | Description |
|----------|-------------|
| `build_binned_features(acc, gyro, n_bins, include_gyro)` | Build the temporal bin sequence (input to BiLSTM) |
| `build_flat_features(acc, gyro, pitch, roll, **flags)` | Build the flat feature vector with configurable flags |
| `extract_attention_entropy(acc)` | 3-dim temporal attention entropy |
| `extract_relative_position(acc)` | 10-dim RPE features |
| `extract_multiscale_fpn(acc)` | 10-dim multi-scale FPN features |
| `extract_kalman_fusion(acc, gyro)` | 10-dim Kalman-fused sensor features |

**Feature flags for `build_flat_features`:**

| Flag | Effect |
|------|--------|
| `include_gyro` | Add 8-dim gyroscope feature block |
| `include_orient` | Add 6-dim pitch/roll orientation block |
| `include_spectral` | Replace base stats with 6-dim Welch PSD block |
| `include_fall_specific` | Add 10-dim fall dynamics block |
| `include_attention` | Append 3-dim attention entropy |
| `include_rpe` | Append 10-dim RPE |
| `include_fpn` | Append 10-dim multi-scale FPN |
| `include_kalman` | Append 10-dim Kalman fusion |

### `src/models.py`

| Class / Function | Description |
|-----------------|-------------|
| `SimpleBiLSTM` | Bidirectional LSTM classifier |
| `DualBranchFusionNet` | Cross-attention Fusion of BiLSTM + MLP branches |
| `StreamingFusionNet` | Hierarchical dual-LSTM for sequential window inputs |
| `train_bilstm(...)` | Training loop for SimpleBiLSTM |
| `train_fusion(...)` | Training loop for DualBranchFusionNet |
| `train_streaming_fusion(...)` | Subject-iterating training loop for StreamingFusionNet |
| `get_classical_models(seed, priors)` | Returns dict of LDA, KNN-3, SVM-RBF, RandomForest |

### `src/evaluation.py`

| Function | Description |
|----------|-------------|
| `run_loso_classical(X, y, groups, model_builder, labels)` | LOSO for classical models |
| `run_loso_dl(X, y, groups, ...)` | LOSO for BiLSTM |
| `run_loso_fusion(Xb, Xf, y, groups, ...)` | LOSO for Fusion model |
| `run_named_model(model_name, ...)` | Dispatcher for any model by name string |
| `run_loso_svm_with_proba(X, y, groups, seed)` | LOSO SVM with raw probability output |
| `run_loso_rf(X, y, groups, seed)` | LOSO Random Forest |
| `get_stage2a_predictions(Xf, y, groups, seed)` | LOSO SVM predictions for Stage 2a routing |
| `compute_purity_metrics(y_true, y_pred, purity, thresh)` | Purity-stratified accuracy report |
| `run_streaming_evaluation(data, ...)` | Subject-level LOSO for StreamingFusionNet |
| `run_naive_classical(data, l2i, n_classes)` | Random Forest baseline with no temporal memory |
| `full_metrics(y_true, y_pred, labels)` | Returns accuracy, balanced acc, macro P/R/F1 |
| `print_metrics(name, seed_metrics)` | Formatted mean±std console output |

---

## 🔁 Validation Strategy

All results use **Leave-One-Subject-Out (LOSO)** cross-validation:
- **67 folds** (one per subject)
- Training: all subjects except one
- Testing: the held-out subject
- `StandardScaler` fitted only on training data to prevent data leakage
- Neural models re-trained from scratch each fold

For statistical significance:
- 5 random seeds per neural model
- Paired t-test against baseline (p < 0.05 threshold)

---

## 🧪 Reproducibility

All random seeds are explicitly set:
```python
torch.manual_seed(seed)
RandomForestClassifier(random_state=seed)
SVC(random_state=seed)
```

To fully reproduce the results, run the scripts in order:
```bash
python scripts/run_stage1_ablations.py
python scripts/run_stage2a_ablations.py
python scripts/run_stage2b_ablations.py
python scripts/run_streaming_ablations.py
python scripts/run_hierarchical_ablations.py
python scripts/run_edge_analysis.py
```

---

## 📁 Results Format

Each script writes results to its corresponding folder under `results/`:

```
results/
├── stage1_all_ablations/
│   ├── stage1_complete_ablations.json   # All metrics, all runs
│   ├── stage1_feature_ablation.csv      # Feature set ablation table
│   └── stage1_model_comparison.csv      # Model comparison table
│
├── stage2a_all_ablations/
│   ├── stage2a_complete_ablations.json
│   ├── stage2a_feature_ablation.csv
│   └── stage2a_model_comparison.csv
│
├── stage2b_all_ablations/
│   ├── stage2b_complete_ablations.json
│   ├── stage2b_feature_ablation.csv
│   └── stage2b_model_comparison.csv
│
├── streaming_ablation/
│   ├── streaming_ablation_results.json
│   └── streaming_ablation_results.csv
│
├── hierarchical/
│   ├── hierarchical_results.json
│   └── hierarchical_ablation_results.csv
│
└── edge_analysis/
    ├── edge_analysis_final.json
    ├── edge_analysis_final.csv
    └── per_class_performance.json
```

---

## 📝 Citation

If you use this codebase or build upon it, please cite the MobiAct dataset:

```bibtex
@inproceedings{vavoulas2016mobiact,
  title={The MobiAct Dataset: Recognition of Activities of Daily Living using Smartphones},
  author={Vavoulas, George and Chatzaki, Charikleia and Malliotakis, Thodoris and Pediaditis, Matthew and Tsiknakis, Manolis},
  booktitle={International Conference on Information and Communication Technologies for Ageing Well and e-Health},
  pages={143--151},
  year={2016}
}
```

---

## 🤝 Contributing

This repository is structured for reviewer reproducibility. If you extend it:
1. Add new feature flags to `src/features.py`
2. Add new models to `src/models.py`
3. Add new evaluation logic to `src/evaluation.py`
4. Create a new orchestrator in `scripts/`

Keep all absolute paths out of `src/` — use relative path resolution via `os.path.dirname(__file__)`.

---

<p align="center">Built with ❤️ for reproducible wearable computing research</p>
