import os
import pandas as pd

# The local relative path to the dataset
# Assuming it's run from the project root
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'MobiAct_Dataset_v2.0', 'Annotated Data')

def load_file(code, subj, trial=1):
    path = os.path.join(DATA_ROOT, code, f"{code}_{subj}_{trial}_annotated.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

def get_segment(code, subj, trial=1):
    df = load_file(code, subj, trial)
    if df is None: return None
    seg = df[df['label'] == code].reset_index(drop=True)
    return seg if len(seg) > 0 else None
