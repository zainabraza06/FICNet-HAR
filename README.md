# MobiAct Project

This repository contains the codebase for evaluating various feature sets and models for Fall vs ADL (Activities of Daily Living) classification using the MobiAct Dataset v2.0.

## Project Structure

```
MobiAct_Project/
├── data/
│   └── MobiAct_Dataset_v2.0/    # Place the extracted dataset here
├── src/
│   ├── data_loader.py           # Handles data loading and segmentation
│   ├── features.py              # Feature extractors (binned, flat, cross-domain)
│   ├── models.py                # Deep learning and classical models
│   └── evaluation.py            # LOSO validation and evaluation metrics
├── scripts/
│   └── run_stage1_ablations.py  # Stage 1 binary classification script
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

### 4. Run Stage 1 Ablations
```bash
python scripts/run_stage1_ablations.py
```
This will run all feature and model ablations for Stage 1 (Binary Fall vs ADL) and save the results inside the `results/stage1_all_ablations/` folder.
