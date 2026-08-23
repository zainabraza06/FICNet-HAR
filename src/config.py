import os
import torch

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT    = os.path.join(PROJECT_ROOT, 'data', 'Annotated Data')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', '01_primary_eval')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Activity codes ────────────────────────────────────────────────────────────
FALL_CODES   = ['BSC', 'FKL', 'FOL', 'SDL']
ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
ALL_CODES    = FALL_CODES + ADL_CODES_11

# ── Task definitions ──────────────────────────────────────────────────────────
TASK_CONFIG = {
    'binary':    {
        'codes':    ALL_CODES,
        'label_fn': lambda code: 'FALL' if code in FALL_CODES else 'ADL',
    },
    'adl_only':  {
        'codes':    ADL_CODES_11,
        'label_fn': lambda code: code,
    },
    'fall_only': {
        'codes':    FALL_CODES,
        'label_fn': lambda code: code,
    },
    'flat15':    {
        'codes':    ALL_CODES,
        'label_fn': lambda code: code,
    },
}

# ── Signal / windowing ────────────────────────────────────────────────────────
FS_NATIVE  = 200.0          # native sensor sample rate (Hz)
FS_TARGET  = 50.0           # resampled rate (Hz)
WINDOW_SEC = 1.0            # window duration (seconds)
OVERLAP    = 0.5            # fractional overlap between successive windows
MIN_WINDOW_FRACTION = 0.9   # discard windows shorter than this fraction of WIN_LEN

WIN_LEN  = int(round(WINDOW_SEC * FS_TARGET))      # samples per window
STEP     = int(round(WIN_LEN * (1 - OVERLAP)))     # stride in samples
MIN_LEN  = int(round(WIN_LEN * MIN_WINDOW_FRACTION))
CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# ── Training modes ────────────────────────────────────────────────────────────
MODES = ['erm', 'sam', 'fic', 'sam_fic']

# ── Hyperparameters ───────────────────────────────────────────────────────────
EPOCHS        = 30
BATCH_SIZE    = 512
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
PROGRESS_EVERY = 5          # print running train loss every N epochs

# Seeds: CV has split + training randomness; LOSO folds are deterministic so
# 1 seed covers training randomness at manageable compute cost.
SEEDS_CV   = [0, 1, 2]
SEEDS_LOSO = [0]

N_CV_FOLDS = 5

# ── SAM ───────────────────────────────────────────────────────────────────────
SAM_RHO = 0.05

# ── FIC / SAM+FIC ────────────────────────────────────────────────────────────
FIC_LOSS_WEIGHT = 0.3       # weight on consistency term relative to CE
FIC_VAR_THRESH  = 1e-6      # drop near-constant handcrafted features
FIC_CORR_THRESH = 0.95      # drop one feature from any pair with |corr| > threshold
