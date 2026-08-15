import os
import pandas as pd
import numpy as np
from src.config import DATA_ROOT, ALL_CODES, FALL_CODES
from src.features.extractors import build_binned_features, build_flat_features

def load_file(code, subj, trial=1):
    path = os.path.join(DATA_ROOT, code, f"{code}_{subj}_{trial}_annotated.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

def get_segment(code, subj, trial=1):
    df = load_file(code, subj, trial)
    if df is None: return None
    seg = df[df['label'] == code].reset_index(drop=True)
    return seg if len(seg) > 0 else None

def build_dataset_stage1(groups, n_bins=5):
    Xb, Xf, y, g_out = [], [], [], []
    for code in ALL_CODES:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None: continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x','acc_y','acc_z']].values
            fb = build_binned_features(acc, n_bins)
            fc = build_flat_features(acc, groups)
            if fb is not None and fc is not None:
                Xb.append(fb)
                Xf.append(fc)
                y.append('FALL' if code in FALL_CODES else 'ADL')
                g_out.append(subj)
    return np.array(Xb), np.array(Xf), np.array(y), np.array(g_out)
