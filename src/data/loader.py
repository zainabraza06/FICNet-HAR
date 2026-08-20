import os
import numpy as np
import pandas as pd
from src.config import DATA_ROOT
from src.features.extractors import build_binned_features, build_flat_features


def load_file(code, subj, trial=1):
    path = os.path.join(DATA_ROOT, code, f"{code}_{subj}_{trial}_annotated.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def get_segment(code, subj, trial=1):
    df = load_file(code, subj, trial)
    if df is None:
        return None
    seg = df[df['label'] == code].reset_index(drop=True)
    return seg if len(seg) > 0 else None


def build_dataset(groups, codes_for_task, label_fn):
    """
    Generic dataset builder used by all four tasks.

    Parameters
    ----------
    groups         : list[str]  — feature groups to include (must be keys of REGISTRY)
    codes_for_task : list[str]  — activity codes for this task
    label_fn       : callable   — maps an activity code to its class label string

    Returns
    -------
    Xb : (N, n_bins, per_bin_dim)  binned tensor for BiLSTM / Fusion
    Xf : (N, flat_dim)             flat feature vector for classical / Fusion MLP
    y  : (N,)                      string labels
    g  : (N,)                      subject IDs (for LOSO split)
    """
    include_gyro_bins = 'gyro' in groups
    Xb, Xf, y, g = [], [], [], []

    for code in codes_for_task:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None:
                continue
            seg = seg.iloc[:1000]
            acc  = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values if 'gyro_x' in seg.columns \
                   else np.zeros((len(seg), 3))
            roll = seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg))

            fb = build_binned_features(acc, gyro, n_bins=5, include_gyro=include_gyro_bins)
            fc = build_flat_features(acc, gyro, roll, groups)
            if fb is not None and fc is not None:
                Xb.append(fb)
                Xf.append(fc)
                y.append(label_fn(code))
                g.append(subj)

    return np.array(Xb), np.array(Xf), np.array(y), np.array(g)
