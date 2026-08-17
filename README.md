# FALL-HAR: A Lightweight Hierarchical System for Smartphone-Based Fall Detection and Activity Recognition

![FALL-HAR Architecture](architectural_diagram_fall_har.png)

## Overview

**FALL-HAR** is a lightweight hierarchical framework designed for smartphone-based human activity recognition (HAR), specifically focusing on the critical task of fall detection. Unlike traditional flat classification models that treat fall detection as just another activity class, FALL-HAR isolates fall detection into its own stage. This asymmetry is driven by the fact that missed falls have far greater consequences than misclassified activities of daily living (ADLs), and falls exhibit substantially greater cross-subject signal consistency.

By decomposing the problem, FALL-HAR allows the critical fall detection stage to be optimized and evaluated independently from finer-grained activity discrimination. 

### Key Contributions
1. **Hierarchical Fall-ADL Framework**: Separates binary fall detection (Stage 1) from 4-class fall-subtype (Stage 2a) and 11-class ADL recognition (Stage 2b), optimizing features for each specific task.
2. **Subject-Independent LOSO Evaluation**: Employs a rigorous Leave-One-Subject-Out (LOSO) cross-validation protocol to ensure true generalization to unseen individuals, avoiding data leakage common in random k-fold splits.
3. **Continuous Streaming Evaluation**: Assesses the system on continuous sliding-window streams to better reflect real-world, unsegmented activity, particularly at activity transitions.
4. **Edge-Ready & Lightweight**: Achieves sub-1.5 ms inference latency and sub-0.1 MB model size, making it highly suitable for resource-constrained edge devices like smartphones.

---

## Architecture Flow

The system is structured as a three-stage hierarchy:

1. **Stage 1 (Binary FALL/ADL Detection)**:
   - **Goal**: Safety-critical isolation of falls from ADLs.
   - **Model**: SVM-RBF.
   - **Features**: Profile, kinematic statistics, and Relative-Position Encoding (RPE).
   - **Performance**: 98.93% LOSO Accuracy.

2. **Stage 2a (Fall Subtype Classification)**:
   - **Goal**: Classifies falls into 4 types (BSC, FKL, FOL, SDL).
   - **Model**: Dual-branch Fusion Network.
   - **Features**: Profile, kinematic statistics, and fall-specific kinematics.
   - **Performance**: 85.86% LOSO Accuracy.

3. **Stage 2b (ADL Subtype Recognition)**:
   - **Goal**: Classifies ADLs into 11 categories (STD, WAL, JOG, etc.).
   - **Model**: Dual-branch Fusion Network.
   - **Features**: Profile, statistics, RPE, and gyroscope-derived statistics.
   - **Performance**: 93.87% LOSO Accuracy.

---

## Dataset and Methodology

### Dataset
This project uses the **MobiAct v2.0 dataset**, containing smartphone accelerometer, gyroscope, and orientation data from 67 participants. Data was collected via a Samsung Galaxy S3 carried in a trouser pocket at ~87 Hz. 
The label space covers:
- **4 Fall Types**: BSC, FKL, FOL, SDL
- **11 ADLs**: STD, WAL, JOG, JUM, STU, STN, SCH, SIT, CHU, CSI, CSO

### Models Evaluated
- **Classical Models**: LDA, KNN-3, SVM-RBF
- **Sequential Neural**: Bidirectional LSTM (BiLSTM)
- **Dual-Branch Fusion**: Combines a BiLSTM over binned temporal data with a Multilayer Perceptron (MLP) over flat features, fused via per-channel gated cross-attention.

### Feature Extraction
The framework uses a greedy, significance-gated feature selection process to determine the optimal representation for each specific stage. Features include:
- Binned representations & kinematic statistics
- Fall-specific kinematics (e.g., impact peaks, settling time)
- Temporal-attention entropy
- Relative-Position Encoding (RPE)
- Gyroscope & Spectral statistics

---

## Key Results

All evaluations strictly use **Leave-One-Subject-Out (LOSO) cross-validation**.

### 1. Stage-by-Stage Performance (Isolated)
| Stage | Task | Selected Model | Best Features | Accuracy (LOSO) | F1 Score |
|-------|------|----------------|---------------|-----------------|----------|
| **1** | Fall vs. ADL | SVM-RBF | Profile + Stats + RPE | **98.93%** | 98.68% |
| **2a** | 4-class Fall | Fusion | Profile + Stats + Fall-specific | **85.86%** | 85.83% |
| **2b** | 11-class ADL | Fusion | Profile + Stats + RPE + Gyro | **93.87%** | 93.68% |

### 2. End-to-End Hierarchical Performance
When evaluated continuously with prediction-based routing (Stage 1 controls routing to Stage 2a/2b):
- **End-to-End Accuracy**: **90.50% ± 0.58%**
- **Balanced Accuracy**: 90.42% ± 0.46%
- *Error Analysis*: Only ~10% of total pipeline errors were due to misrouting at Stage 1. The vast majority of residual error lies in the lower-stakes subtype discrimination, validating the hierarchical isolation of the safety-critical fall detector.

### 3. Continuous Streaming Evaluation
Testing on unsegmented, continuous data (200-sample sliding windows, 50% overlap):
- **Classical Baseline (RF)**: 73.55% overall (40.75% at transitions)
- **Fusion Baseline**: 92.25% overall (71.58% at transitions)
- **Fusion + RPE**: **93.04% overall**
*The sequential Fusion architecture dramatically outperforms memory-free classical approaches, particularly during complex activity transitions.*

### 4. Computational Footprint
The models are highly optimized for edge deployment:
- **Inference Latency**: < 1.5 ms per window
- **Model Storage**: < 0.1 MB per stage

---

## Conclusion
FALL-HAR demonstrates that decoupling fall detection from general activity recognition yields a highly reliable, safety-first pipeline. By applying task-specific feature selection and rigorous subject-independent LOSO validation, the system achieves near-state-of-the-art binary fall detection while maintaining strong multi-class ADL recognition and excellent edge-device efficiency.
