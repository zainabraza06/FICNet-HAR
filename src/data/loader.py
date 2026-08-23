import os
import numpy as np
import pandas as pd
from src.config import (DATA_ROOT, FS_NATIVE, FS_TARGET, CHANNELS,
                        WIN_LEN, STEP, MIN_LEN)


def load_and_segment(fpath, code):
    """Read one annotated CSV and return only the rows labelled with `code`."""
    df = pd.read_csv(fpath)
    if 'label' not in df.columns:
        return None
    seg = df[df['label'] == code].reset_index(drop=True)
    return seg if len(seg) > 0 else None


def resample_segment(seg_df, native_fs=FS_NATIVE, target_fs=FS_TARGET, cols=CHANNELS):
    """
    Duration-preserving linear interpolation from native_fs to target_fs.

    Returns an (n_target, len(cols)) float32 array.
    Missing columns are filled with zeros.
    """
    n = len(seg_df)
    duration_sec = n / native_fs
    n_target = max(1, int(round(duration_sec * target_fs)))
    t_native = np.linspace(0, duration_sec, n,        endpoint=False)
    t_target = np.linspace(0, duration_sec, n_target, endpoint=False)
    out = np.zeros((n_target, len(cols)), dtype=np.float32)
    for i, c in enumerate(cols):
        if c in seg_df.columns:
            out[:, i] = np.interp(t_target, t_native, seg_df[c].values)
    return out


def make_windows(signal_arr):
    """
    Slide a fixed-length window over a resampled signal array.

    Skips segments shorter than MIN_LEN. Returns a list of
    (WIN_LEN, n_channels) float32 arrays.
    """
    n = len(signal_arr)
    if n < MIN_LEN:
        return []
    windows = []
    start = 0
    while start + WIN_LEN <= n:
        windows.append(signal_arr[start:start + WIN_LEN])
        start += STEP
    return windows


def build_raw_dataset(codes, label_fn):
    """
    Scans DATA_ROOT for every `*_annotated.csv` matching the given codes,
    resamples to FS_TARGET, windows with WIN_LEN / STEP / MIN_LEN, and
    returns three parallel arrays.

    Returns
    -------
    X    : (N, WIN_LEN, 6)  float32  — raw 6-channel windows
    y    : (N,)             str      — class labels produced by label_fn
    subj : (N,)             int64    — subject IDs parsed from filenames
    """
    X, y, subj = [], [], []

    for code in codes:
        code_dir = os.path.join(DATA_ROOT, code)
        if not os.path.isdir(code_dir):
            continue
        for fname in sorted(os.listdir(code_dir)):
            if not fname.endswith('_annotated.csv'):
                continue
            parts = fname.replace('_annotated.csv', '').split('_')
            if len(parts) < 3:
                continue
            fcode, subject_id = parts[0], int(parts[1])
            seg = load_and_segment(os.path.join(code_dir, fname), fcode)
            if seg is None:
                continue
            resampled = resample_segment(seg)
            for w in make_windows(resampled):
                X.append(w)
                y.append(label_fn(fcode))
                subj.append(subject_id)

    return (np.array(X, dtype=np.float32),
            np.array(y),
            np.array(subj, dtype=np.int64))
