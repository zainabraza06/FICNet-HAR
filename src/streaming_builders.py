"""
Sliding-window dataset builders for the streaming ablation study.

Mirrors the structure of run_stage2b_streaming_final.py but adds
include_attention / include_rpe flags so the ablation runner can sweep
four feature variants without duplicating the windowing logic.

Constants (matching the validated streaming baseline)
-------------------------------------------------------
  WINDOW_SAMPLES = 200   (~2.3 s at 87 Hz)
  STRIDE         = 100   (~1.1 s — 50 % overlap)
  SUB_BINS       = 5     temporal bins inside each window
  PURITY_THRESHOLD = 0.9 fraction of dominant label in a window

Per-window feature layout
--------------------------
  Binned (for the recurrent branch):
    shape (SUB_BINS, per_bin_dim)
    per_bin_dim = 3 acc-energy fracs + mean + std + 3 gyro-energy fracs = 8

  Flat (for the MLP branch):
    15 per-axis acc-energy profile  (3 axes × 5 bins)
     4 acc-mag stats                (var, mean, skew, kurtosis)
     8 gyro features                (axis fracs × 3, skew, kurt, mean,
                                     roll_range, autocorr_peak)
    [3 attention features]          (if include_attention)
    [10 RPE features]               (if include_rpe)
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import correlate, find_peaks

from .streaming_features import (
    extract_attention_entropy_streaming,
    extract_relative_position_features_streaming,
)

# ============================================================
# Constants
# ============================================================
ADL_CODES_11     = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
WINDOW_SAMPLES   = 200
STRIDE           = 100
SUB_BINS         = 5
PURITY_THRESHOLD = 0.90
MAX_PER_ACTIVITY = 1000


# ============================================================
# Data helpers
# ============================================================

def _load_segment(data_root: str, code: str, subj: int, trial: int = 1):
    path = os.path.join(data_root, code, f"{code}_{subj}_{trial}_annotated.csv")
    if not os.path.exists(path):
        return None
    df  = pd.read_csv(path)
    seg = df[df['label'] == code].reset_index(drop=True)
    return seg if len(seg) > 0 else None


# ============================================================
# Per-window feature builders
# ============================================================

def _build_window_binned(
    acc_w: np.ndarray,
    gyro_w: np.ndarray,
    n_bins: int = SUB_BINS,
) -> np.ndarray:
    """
    Binned temporal features for one window — shape (n_bins, 8).

    Columns: acc_ex, acc_ey, acc_ez, acc_mean, acc_std,
             gyro_ex, gyro_ey, gyro_ez
    """
    bin_edges = np.linspace(0, len(acc_w), n_bins + 1).astype(int)
    bins = []
    for i in range(n_bins):
        a = acc_w[bin_edges[i]:bin_edges[i + 1]]
        if len(a) == 0:
            a = acc_w[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
        ae = (a ** 2).sum(axis=0); ae = ae / (ae.sum() + 1e-8)
        am = np.sqrt((a ** 2).sum(axis=1))
        feat = list(ae) + [am.mean(), am.std()]

        g = gyro_w[bin_edges[i]:bin_edges[i + 1]]
        if len(g) == 0:
            g = gyro_w[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
        ge = (g ** 2).sum(axis=0); ge = ge / (ge.sum() + 1e-8)
        feat += list(ge)
        bins.append(feat)
    return np.array(bins, dtype=np.float32)


def _build_window_flat(
    acc_w:   np.ndarray,
    gyro_w:  np.ndarray,
    pitch_w: np.ndarray,
    roll_w:  np.ndarray,
    n_bins:  int = SUB_BINS,
    include_attention: bool = False,
    include_rpe: bool = False,
) -> np.ndarray:
    """
    Flat feature vector for one window.

    Baseline (27 dims)
        15  per-axis acc-energy profile
         4  acc-mag stats (var, mean, skew, kurtosis)
         8  gyro features

    Optional extras (concatenated at the end)
        +3  attention entropy  (if include_attention)
        +10 RPE               (if include_rpe)
    """
    bin_edges = np.linspace(0, len(acc_w), n_bins + 1).astype(int)

    # 15-dim per-axis energy profile
    profile = []
    for axis in range(3):
        e  = acc_w[:, axis] ** 2
        ap = np.array([e[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(n_bins)])
        profile.append(ap / (ap.sum() + 1e-8))
    profile = np.concatenate(profile)

    # 4-dim acc-mag stats
    acc_mag  = np.sqrt((acc_w ** 2).sum(axis=1))
    acc_base = np.array([acc_mag.var(), acc_mag.mean(), skew(acc_mag), kurtosis(acc_mag)])

    # 8-dim gyro block
    gyro_mag = np.sqrt((gyro_w ** 2).sum(axis=1))
    ge = gyro_w ** 2; tge = ge.sum() + 1e-8
    gx, gy, gz = ge[:, 0].sum()/tge, ge[:, 1].sum()/tge, ge[:, 2].sum()/tge
    ac    = acc_mag - acc_mag.mean()
    acorr = correlate(ac, ac, mode='full')[len(ac) - 1:]
    acorr = acorr / (acorr[0] + 1e-8)
    pk, _ = find_peaks(acorr[5:], height=0.2)
    autocorr   = float(acorr[pk[0] + 5]) if len(pk) > 0 else 0.0
    roll_range = float(roll_w.max() - roll_w.min()) if len(roll_w) > 0 else 0.0
    gyro_vec   = np.array([gx, gy, gz,
                            skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean(),
                            roll_range, autocorr])

    parts = [profile, acc_base, gyro_vec]

    if include_attention:
        parts.append(extract_attention_entropy_streaming(acc_w))
    if include_rpe:
        parts.append(extract_relative_position_features_streaming(acc_w))

    return np.concatenate(parts).astype(np.float32)


# ============================================================
# Subject stream builder
# ============================================================

def build_subject_stream(
    data_root: str,
    codes: list[str],
    subject: int,
    max_per_activity: int = MAX_PER_ACTIVITY,
) -> tuple | None:
    """
    Concatenate all activity segments for one subject into a single
    continuous stream.

    Returns
    -------
    (acc, gyro, pitch, roll, labels) arrays or None if no data found.
    """
    acc_parts, gyro_parts   = [], []
    pitch_parts, roll_parts = [], []
    label_parts             = []

    for code in codes:
        seg = _load_segment(data_root, code, subject)
        if seg is None:
            continue
        seg = seg.iloc[:max_per_activity]
        acc_parts.append(seg[['acc_x', 'acc_y', 'acc_z']].values)
        gyro_parts.append(seg[['gyro_x', 'gyro_y', 'gyro_z']].values)
        pitch_parts.append(
            seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg))
        )
        roll_parts.append(
            seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg))
        )
        label_parts.extend([code] * len(seg))

    if not acc_parts:
        return None

    return (
        np.vstack(acc_parts),
        np.vstack(gyro_parts),
        np.concatenate(pitch_parts),
        np.concatenate(roll_parts),
        np.array(label_parts),
    )


# ============================================================
# Sliding-window segmentation
# ============================================================

def window_stream(
    acc_s:   np.ndarray,
    gyro_s:  np.ndarray,
    pitch_s: np.ndarray,
    roll_s:  np.ndarray,
    labels_s: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    stride:         int = STRIDE,
    sub_bins:       int = SUB_BINS,
    include_attention: bool = False,
    include_rpe: bool = False,
) -> tuple:
    """
    Slide a window over a subject stream and extract features.

    Returns
    -------
    X_bins    : (N, sub_bins, per_bin_dim)
    X_flat    : (N, flat_dim)
    y_labels  : (N,) str
    purities  : (N,) float — fraction of dominant label in each window
    or (None, None, None, None) if no windows were produced.
    """
    n = len(acc_s)
    X_bins, X_flat, y_labels, purities = [], [], [], []

    start = 0
    while start + window_samples <= n:
        end  = start + window_samples
        acc_w   = acc_s[start:end]
        gyro_w  = gyro_s[start:end]
        pitch_w = pitch_s[start:end]
        roll_w  = roll_s[start:end]
        lw      = labels_s[start:end]

        vals, counts = np.unique(lw, return_counts=True)
        majority = vals[np.argmax(counts)]
        purity   = float(counts.max() / len(lw))

        fb = _build_window_binned(acc_w, gyro_w, sub_bins)
        fc = _build_window_flat(acc_w, gyro_w, pitch_w, roll_w, sub_bins,
                                include_attention=include_attention,
                                include_rpe=include_rpe)

        X_bins.append(fb)
        X_flat.append(fc)
        y_labels.append(majority)
        purities.append(purity)
        start += stride

    if not X_bins:
        return None, None, None, None

    return (
        np.array(X_bins),
        np.array(X_flat),
        np.array(y_labels),
        np.array(purities),
    )


# ============================================================
# Full-dataset streaming builder
# ============================================================

def build_streaming_dataset(
    data_root: str,
    codes: list[str],
    subjects: list[int],
    max_per_activity: int = MAX_PER_ACTIVITY,
    include_attention: bool = False,
    include_rpe: bool = False,
    min_windows: int = 5,
) -> dict[int, tuple]:
    """
    Build the complete streaming dataset as a dict keyed by subject ID.

    Each value is  (X_bins, X_flat, y_labels, purities)  for that
    subject's continuous stream segmented into sliding windows.

    Parameters
    ----------
    data_root         : path to MobiAct 'Annotated Data' directory
    codes             : list of activity codes to include
    subjects          : list of subject IDs to process
    max_per_activity  : max samples per activity per subject
    include_attention : append attention entropy features to flat vector
    include_rpe       : append RPE features to flat vector
    min_windows       : skip subjects with fewer windows than this

    Returns
    -------
    dict { subject_id : (X_bins, X_flat, y_labels, purities) }
    """
    dataset: dict[int, tuple] = {}

    for subj in subjects:
        stream = build_subject_stream(data_root, codes, subj, max_per_activity)
        if stream is None:
            continue

        acc_s, gyro_s, pitch_s, roll_s, labels_s = stream
        X_bins, X_flat, y_labels, purities = window_stream(
            acc_s, gyro_s, pitch_s, roll_s, labels_s,
            include_attention=include_attention,
            include_rpe=include_rpe,
        )

        if X_bins is None or len(X_bins) < min_windows:
            continue

        dataset[subj] = (X_bins, X_flat, y_labels, purities)

    return dataset
