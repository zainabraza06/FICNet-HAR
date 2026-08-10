# MobiAct HAR — Subject-Independent Fall & Activity Recognition

Preprocessing and feature engineering pipeline for a three-stage
LOSO-validated fall/activity recognition system on the MobiAct v2.0
dataset.

## Architecture
1. **Stage 1 — Fall Gate** (binary): 5-bin |acc| energy-distribution
   profile. Alignment-free, subject-invariant by construction.
2. **Stage 2a — Fall Subtype** (4-class: BSC/FKL/FOL/SDL): 34-dim
   feature vector (per-axis energy profile + scalar + gyroscope).
   LOSO accuracy: 0.816
3. **Stage 2b — ADL Classification** (9- or 11-class): 35-dim
   feature vector (profile + spectral + gyroscope + orientation).
   LOSO accuracy: 0.925 (9-class) / 0.884 (11-class, 0.865 balanced)

## Key Finding
Gyroscope-derived axis-distribution and moment features
consistently outperform accelerometer-only features for
fine-grained activity/fall-subtype disambiguation (effect sizes
up to Cohen's d = 2.97), a signal largely unused in prior MobiAct
literature.

## What was tried and rejected (see `src/features.py` docstring)
Orientation canonicalization, body-metric conditioning, cross-modal
jerk×gyro fusion, hierarchical hard-routing classification, and
K-shot personalization calibration were all empirically tested and
did not improve on the final pipeline — documented for transparency.

## Usage

### Running Stage 1 Evaluation
To run the full subject-independent Leave-One-Group-Out (LOSO) evaluation of all Stage 1 models (5 classical models, SimpleBiLSTM, DualBranchFusionNet, Residual Tree Correction, significance tests, feature-channel ablation, and probability averaging ensemble):
```bash
python run_stage1.py
```

To run a quick dry-run with a subset of subjects and training epochs to verify execution:
```bash
python run_stage1.py --fast
```

### Feature Engineering Pipeline Example
```python
import numpy as np
from src.data_loader import MobiActLoader
from src.pipeline import build_stage2b_dataset
from src.evaluate import run_loso
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

loader = MobiActLoader('MobiAct_Dataset_v2.0/Annotated Data')
X, y, groups = build_stage2b_dataset(loader, subjects=list(range(1, 68)), n_classes=11)
results = run_loso(X, y, groups, clf_builder=lambda: LinearDiscriminantAnalysis(
    priors=np.ones(11)/11))
print("Accuracy:", results['accuracy'], "Balanced Accuracy:", results['balanced_accuracy'])
```

## Status
- Preprocessing and feature engineering: **finalized**.
- Stage 1 evaluation and training pipelines: **completed and verified**.
- Next: design and integrate Stage 2a/2b deep learning architectures.

