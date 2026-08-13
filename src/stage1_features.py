"""
Unified feature extraction for Stage 1 complete ablation study.

This module collects ALL cross-domain feature families tested across
the ablation series and exposes a single build_stage1_dataset() entry
point that accepts boolean flags for each family.

Feature inventory
-----------------
Baseline (always present):
  profile  — 15-dim per-axis energy profile  (3 axes × 5 bins)
  acc_stats — 5-dim acc-magnitude statistics  (skew, kurt, var, mean, max_jerk)

Optional groups (each flag adds dimensions to the flat vector):
  include_gyro      → +8  dims (gyro axis fracs × 3, skew, kurt, mean,
                                roll_range, autocorr_peak)
  include_attention → +3  dims (attention_entropy, concentration, uniformity)
  include_rpe       → +10 dims (lag autocorrs, change-rate ratios,
                                position-weighted stats, high-act entropy)
  include_fpn       → +10 dims (multi-scale stats at 3 temporal scales
                                + 2 cross-scale ratios, truncated to 10)
  include_kalman    → +10 dims (Kalman fused signal stats + gain/trust)
                       *** requires include_gyro to be meaningful ***

Empirical results (from ablation runs, documented in comments):
  Temporal Attention : +0.36% Stage 1         → WORKS
  Multi-Scale FPN    : -0.12% Stage 1         → FAILS
  Kalman Fusion      : -0.12% Stage 1         → FAILS  (gyro adds noise here)
  Relative Pos Enc   :  0.00% Stage 1         → NEUTRAL
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import correlate, find_peaks

from .cross_domain_features import (
    extract_attention_entropy,
    extract_multi_scale_features,
)
from .kalman_rpe_features import (
    kalman_fusion_features,
    extract_relative_position_features,
)

# ============================================================
# Constants
# ============================================================
FALL_CODES   = ['BSC', 'FKL', 'FOL', 'SDL']
ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
ALL_CODES    = FALL_CODES + ADL_CODES_11
N_BINS       = 5
MAX_SAMPLES  = 1000


# ============================================================
# Data helper
# ============================================================

def _load_segment(data_root: str, code: str, subj: int, trial: int = 1):
    path = os.path.join(data_root, code, f"{code}_{subj}_{trial}_annotated.csv")
    if not os.path.exists(path):
        return None
    df  = pd.read_csv(path)
    seg = df[df['label'] == code].reset_index(drop=True)
    return seg if len(seg) > 0 else None


# ============================================================
# Binned features (for recurrent branch)
# ============================================================

def build_binned_features(
    acc: np.ndarray,
    gyro: np.ndarray | None = None,
    n_bins: int = N_BINS,
    include_gyro: bool = False,
) -> np.ndarray | None:
    """
    Binned temporal features — shape (n_bins, per_bin_dim).

    per_bin_dim = 5 (acc_ex, acc_ey, acc_ez, acc_mean, acc_std)
               + 3 (gyro_ex, gyro_ey, gyro_ez)  when include_gyro
    """
    if len(acc) < n_bins:
        return None
    bin_edges = np.linspace(0, len(acc), n_bins + 1).astype(int)
    bins = []
    for i in range(n_bins):
        a = acc[bin_edges[i]:bin_edges[i + 1]]
        if len(a) == 0:
            a = acc[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
        ae = (a ** 2).sum(axis=0); ae = ae / (ae.sum() + 1e-8)
        am = np.sqrt((a ** 2).sum(axis=1))
        feat = list(ae) + [am.mean(), am.std()]
        if include_gyro and gyro is not None:
            g = gyro[bin_edges[i]:bin_edges[i + 1]]
            if len(g) == 0:
                g = gyro[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
            ge = (g ** 2).sum(axis=0); ge = ge / (ge.sum() + 1e-8)
            feat += list(ge)
        bins.append(feat)
    return np.array(bins, dtype=np.float32)


# ============================================================
# Flat features (for classical / MLP branch)
# ============================================================

def build_flat_features(
    acc: np.ndarray,
    gyro: np.ndarray | None  = None,
    pitch: np.ndarray | None = None,
    roll:  np.ndarray | None = None,
    n_bins: int = N_BINS,
    include_gyro:      bool = False,
    include_attention: bool = False,
    include_rpe:       bool = False,
    include_fpn:       bool = False,
    include_kalman:    bool = False,
) -> np.ndarray | None:
    """
    Flat feature vector for Stage 1 binary classification.

    Always included (20 dims):
        15  per-axis energy profile
         5  acc-magnitude stats

    Optional (each flag appends to the vector):
        include_gyro      → +8
        include_attention → +3   (extract_attention_entropy)
        include_rpe       → +10  (extract_relative_position_features)
        include_fpn       → +10  (extract_multi_scale_features, truncated)
        include_kalman    → +10  (kalman_fusion_features, needs gyro)
    """
    if len(acc) < n_bins:
        return None

    bin_edges = np.linspace(0, len(acc), n_bins + 1).astype(int)

    # 15-dim per-axis energy profile
    profile = []
    for axis in range(3):
        e  = acc[:, axis] ** 2
        ap = np.array([e[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(n_bins)])
        profile.append(ap / (ap.sum() + 1e-8))
    profile = np.concatenate(profile)

    # 5-dim acc-mag stats
    acc_mag   = np.sqrt((acc ** 2).sum(axis=1))
    acc_stats = np.array([
        skew(acc_mag), kurtosis(acc_mag),
        acc_mag.var(), acc_mag.mean(),
        np.max(np.abs(np.gradient(acc_mag))),
    ])

    parts = [profile, acc_stats]

    # Gyroscope block
    if include_gyro and gyro is not None:
        gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))
        ge = gyro ** 2; tge = ge.sum() + 1e-8
        gx, gy, gz = ge[:, 0].sum()/tge, ge[:, 1].sum()/tge, ge[:, 2].sum()/tge
        ac    = acc_mag - acc_mag.mean()
        acorr = correlate(ac, ac, mode='full')[len(ac) - 1:]
        acorr = acorr / (acorr[0] + 1e-8)
        pk, _ = find_peaks(acorr[5:], height=0.2)
        autocorr   = float(acorr[pk[0] + 5]) if len(pk) > 0 else 0.0
        roll_range = float(roll.max() - roll.min()) if (roll is not None and len(roll) > 0) else 0.0
        parts.append(np.array([gx, gy, gz,
                                skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean(),
                                roll_range, autocorr]))

    # Cross-domain features
    if include_attention:
        parts.append(extract_attention_entropy(acc))

    if include_rpe:
        parts.append(extract_relative_position_features(acc))

    if include_fpn:
        fpn = extract_multi_scale_features(acc)
        parts.append(fpn[:10])   # truncate to 10 dims for consistency

    if include_kalman and gyro is not None:
        parts.append(kalman_fusion_features(acc, gyro))

    return np.concatenate(parts).astype(np.float32)


# ============================================================
# Dataset builder
# ============================================================

def build_stage1_dataset(
    data_root: str,
    include_gyro:      bool = False,
    include_attention: bool = False,
    include_rpe:       bool = False,
    include_fpn:       bool = False,
    include_kalman:    bool = False,
) -> tuple:
    """
    Build the Stage 1 (binary FALL vs ADL) dataset.

    Parameters
    ----------
    data_root         : path to MobiAct 'Annotated Data' directory
    include_gyro      : add 8-dim gyroscope feature block
    include_attention : add 3-dim temporal attention entropy
    include_rpe       : add 10-dim relative position encoding
    include_fpn       : add 10-dim multi-scale FPN features
    include_kalman    : add 10-dim Kalman fusion features (needs gyro)

    Returns
    -------
    X_bins : (N, n_bins, per_bin_dim)
    X_flat : (N, flat_dim)
    y      : (N,) str — 'FALL' | 'ADL'
    groups : (N,) int — subject IDs for LOSO splitting
    """
    X_bins, X_flat, y, groups = [], [], [], []

    for code in ALL_CODES:
        label = 'FALL' if code in FALL_CODES else 'ADL'
        for subj in range(1, 68):
            seg = _load_segment(data_root, code, subj)
            if seg is None:
                continue
            seg   = seg.iloc[:MAX_SAMPLES]
            acc   = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro  = seg[['gyro_x', 'gyro_y', 'gyro_z']].values
            pitch = seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg))
            roll  = seg['roll'].values  if 'roll'  in seg.columns else np.zeros(len(seg))

            fb = build_binned_features(acc, gyro,
                                       include_gyro=include_gyro)
            fc = build_flat_features(acc, gyro, pitch, roll,
                                     include_gyro=include_gyro,
                                     include_attention=include_attention,
                                     include_rpe=include_rpe,
                                     include_fpn=include_fpn,
                                     include_kalman=include_kalman)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append(label);   groups.append(subj)

    return (np.array(X_bins), np.array(X_flat),
            np.array(y),      np.array(groups))
