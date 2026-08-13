# MobiAct Project

This repository contains the codebase for evaluating various feature sets and models for Fall vs ADL classification, Fall Subtypes, and ADL activities using the MobiAct Dataset v2.0.

## Project Structure

```
MobiAct_Project/
├── data/
│   └── MobiAct_Dataset_v2.0/    # Place the extracted dataset here
├── src/
│   ├── data_loader.py           # Handles data loading and segmentation
│   ├── features.py              # Unified feature extractors (binned, flat, cross-domain, spectral)
│   ├── models.py                # Deep learning and classical models
│   └── evaluation.py            # LOSO validation and evaluation metrics
├── scripts/
│   ├── run_stage1_ablations.py    # Stage 1: Binary Fall vs ADL
│   ├── run_stage2a_ablations.py   # Stage 2a: Fall Subtypes (4-class)
│   ├── run_stage2b_ablations.py   # Stage 2b: ADL Classification (11-class)
│   ├── run_streaming_ablations.py # Streaming Evaluation (Windowing & Purity)
│   ├── run_hierarchical_ablations.py # End-to-End Hierarchical System
│   └── run_edge_analysis.py          # Edge Deployment Profiling (CPU latency, model size)
├── configs/                     # Hyperparameters and paths
├── tests/                       # Unit/smoke tests
├── results/                     # Output directory for results and logs
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd MobiAct_Project
```

### 2. Install dependencies
It is recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Data Preparation
1. Download `MobiAct_Dataset_v2.0`.
2. Extract the dataset and place the `MobiAct_Dataset_v2.0` folder inside the `data/` directory of this repository. The path `data/MobiAct_Dataset_v2.0/Annotated Data/` should exist.

### 4. Running the Pipelines

#### Stage 1: Binary Fall vs ADL
```bash
python scripts/run_stage1_ablations.py
```

#### Stage 2a: Fall Subtypes (4-class)
```bash
python scripts/run_stage2a_ablations.py
```

#### Stage 2b: ADL Classification (11-class)
```bash
python scripts/run_stage2b_ablations.py
```

#### Streaming Evaluation (Purity Stratification)
```bash
python scripts/run_streaming_ablations.py
```

#### End-to-End Hierarchical System
```bash
python scripts/run_hierarchical_ablations.py
```

#### Edge Deployment Analysis (CPU Profiling)
```bash
python scripts/run_edge_analysis.py
```

Results (JSON, CSV summaries) for each stage will be saved in their respective directories within the `results/` folder.
