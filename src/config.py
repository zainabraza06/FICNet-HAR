import os
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
# Assuming the script runs from the project root (Mobiact_projecct)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(PROJECT_ROOT, 'data', 'Annotated Data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'stage1_complete')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Code and Classes
FALL_CODES = ['BSC', 'FKL', 'FOL', 'SDL']
ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
ALL_CODES = FALL_CODES + ADL_CODES_11
LABELS_S1 = ['ADL', 'FALL']

# Model and Feature Settings
SEEDS = [0, 1, 2, 3, 4]
CLASSICAL_MODELS = {'LDA', 'KNN-3', 'SVM-RBF'}
MODELS = ['LDA', 'KNN-3', 'SVM-RBF', 'BiLSTM', 'Fusion']
ALPHA = 0.05  # significance threshold for accepting a feature addition
