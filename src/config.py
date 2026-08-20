import os
import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'data', 'Annotated Data')
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_ROOT, exist_ok=True)

# ── Activity codes ────────────────────────────────────────────────────────────
FALL_CODES   = ['BSC', 'FKL', 'FOL', 'SDL']
ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
ALL_CODES    = FALL_CODES + ADL_CODES_11

# ── Task definitions ──────────────────────────────────────────────────────────
TASK_CONFIG = {
    'binary':    {
        'codes':    ALL_CODES,
        'labels':   ['ADL', 'FALL'],
        'label_fn': lambda code: 'FALL' if code in FALL_CODES else 'ADL',
    },
    'adl_only':  {
        'codes':    ADL_CODES_11,
        'labels':   sorted(ADL_CODES_11),
        'label_fn': lambda code: code,
    },
    'fall_only': {
        'codes':    FALL_CODES,
        'labels':   sorted(FALL_CODES),
        'label_fn': lambda code: code,
    },
    'flat15':    {
        'codes':    ALL_CODES,
        'labels':   sorted(ALL_CODES),
        'label_fn': lambda code: code,
    },
}

TASKS_TO_RUN = ['binary', 'adl_only', 'fall_only', 'flat15']

# ── Model / training settings ─────────────────────────────────────────────────
CLASSICAL_MODELS = {'LDA', 'KNN-3', 'SVM-RBF'}
MODELS           = ['LDA', 'KNN-3', 'SVM-RBF', 'BiLSTM', 'Fusion']
SEEDS            = [0, 1, 2, 3, 4]
ALPHA            = 0.05
EPOCHS           = 250
