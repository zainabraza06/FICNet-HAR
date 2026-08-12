"""
Enhanced feature builders and dataset constructors for cross-domain ablation.

Each build function accepts two boolean flags:
    include_attention  – append Temporal Attention Entropy features
    include_multiscale – append Multi-Scale FPN features

The four combinations (Baseline / +Attention / +FPN / +Both) are used
by the ablation runner to measure the marginal contribution of each
cross-domain adaptation against the best-performing baseline model for
that stage.

Stage mapping
-------------
Stage 1  (binary fall gate)      → SVM-RBF  best baseline
Stage 2a (fall subtype, 4-class) → DualBranchFusionNet best baseline
Stage 2b (ADL, 11-class)         → RandomForest best baseline
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from .cross_domain_features import extract_attention_entropy, extract_multi_scale_features

# ============================================================
# Dataset constants (mirrors run_stage* scripts)
# ============================================================
FALL_CODES    = ['BSC', 'FKL', 'FOL', 'SDL']
ADL_CODES_11  = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
N_BINS        = 5
MAX_SAMPLES   = 1000


# ============================================================
# Low-level data helpers
# ============================================================

def _load_segment(data_root: str, code: str, subj: int, trial: int = 1):
    """Load a single annotated CSV and return the labelled segment rows."""
    path = os.path.join(data_root, code, f"{code}_{subj}_{trial}_annotated.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    seg = df[df['label'] == code].reset_index(drop=True)
    return seg if len(seg) > 0 else None


# ============================================================
# Low-level feature builders
# ============================================================

def _build_binned(acc: np.ndarray, gyro: np.ndarray | None,
                  n_bins: int = N_BINS, include_gyro: bool = False) -> np.ndarray | None:
    """
    Build binned temporal features — [n_bins, per_bin_dim].

    per_bin_dim = 5 (acc energy fractions × 3 + mean + std)
                + 3 (gyro energy fractions, if include_gyro)
    """
    if len(acc) < n_bins:
        return None
    bin_edges = np.linspace(0, len(acc), n_bins + 1).astype(int)
    bins = []
    for i in range(n_bins):
        a = acc[bin_edges[i]:bin_edges[i + 1]]
        if len(a) == 0:
            a = acc[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
        ae = (a ** 2).sum(axis=0)
        ae = ae / (ae.sum() + 1e-8)
        am = np.sqrt((a ** 2).sum(axis=1))
        feat = list(ae) + [am.mean(), am.std()]
        if include_gyro and gyro is not None:
            g = gyro[bin_edges[i]:bin_edges[i + 1]]
            if len(g) == 0:
                g = gyro[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
            ge = (g ** 2).sum(axis=0)
            ge = ge / (ge.sum() + 1e-8)
            feat += list(ge)
        bins.append(feat)
    return np.array(bins, dtype=np.float32)


def _build_flat(acc: np.ndarray, gyro: np.ndarray | None = None,
                pitch: np.ndarray | None = None, roll: np.ndarray | None = None,
                n_bins: int = N_BINS, include_gyro: bool = False) -> np.ndarray | None:
    """
    Build a flat (1-D) feature vector.

    Dimensions
    ----------
    Always present (20 dims):
        15 per-axis energy profile  (3 axes × n_bins)
        5  acc-magnitude stats       (skew, kurt, var, mean, max_jerk)

    Optional gyro block (8 dims when include_gyro=True):
        3  gyro axis energy fracs  (gx, gy, gz)
        3  gyro mag stats          (skew, kurt, mean)
        1  roll range              (requires roll array)
        1  autocorr peak strength
    """
    if len(acc) < n_bins:
        return None
    bin_edges = np.linspace(0, len(acc), n_bins + 1).astype(int)

    # Per-axis energy profile
    profile = []
    for axis in range(3):
        e  = acc[:, axis] ** 2
        ap = np.array([e[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(n_bins)])
        profile.append(ap / (ap.sum() + 1e-8))
    profile = np.concatenate(profile)   # 15-dim

    # Accelerometer magnitude stats
    acc_mag = np.sqrt((acc ** 2).sum(axis=1))
    acc_stats = np.array([
        skew(acc_mag), kurtosis(acc_mag),
        acc_mag.var(), acc_mag.mean(),
        np.max(np.abs(np.gradient(acc_mag))),
    ])

    parts = [profile, acc_stats]

    if include_gyro and gyro is not None:
        from scipy.signal import correlate, find_peaks
        gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))
        ge  = gyro ** 2
        tge = ge.sum() + 1e-8
        gx, gy, gz = ge[:, 0].sum() / tge, ge[:, 1].sum() / tge, ge[:, 2].sum() / tge

        ac_signal = acc_mag - acc_mag.mean()
        acorr = correlate(ac_signal, ac_signal, mode='full')[len(ac_signal) - 1:]
        acorr = acorr / (acorr[0] + 1e-8)
        pk, _ = find_peaks(acorr[5:], height=0.2)
        autocorr = float(acorr[pk[0] + 5]) if len(pk) > 0 else 0.0

        roll_range = float(roll.max() - roll.min()) if (roll is not None and len(roll) > 0) else 0.0

        gyro_vec = np.array([gx, gy, gz,
                             skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean(),
                             roll_range, autocorr])
        parts.append(gyro_vec)

    return np.concatenate(parts).astype(np.float32)


# ============================================================
# Enhanced wrappers  (add cross-domain features on top)
# ============================================================

def _enhanced_binned(acc: np.ndarray, gyro: np.ndarray | None,
                     n_bins: int = N_BINS, include_gyro: bool = False,
                     include_attention: bool = False,
                     include_multiscale: bool = False) -> np.ndarray | None:
    """
    Binned features ± cross-domain augmentation.

    When augmentation is enabled, extra features are appended as an
    additional pseudo-bin row (padded / truncated to match per_bin_dim)
    so that the 3-D tensor shape fed to the BiLSTM stays consistent.
    """
    base = _build_binned(acc, gyro, n_bins=n_bins, include_gyro=include_gyro)
    if base is None:
        return None

    if not (include_attention or include_multiscale):
        return base

    extra: list[float] = []
    if include_attention:
        extra.extend(extract_attention_entropy(acc).tolist())
    if include_multiscale:
        extra.extend(extract_multi_scale_features(acc).tolist())

    per_bin_dim = base.shape[1]
    extra_arr = np.array(extra, dtype=np.float32)
    # Align length to per_bin_dim
    if len(extra_arr) < per_bin_dim:
        extra_arr = np.pad(extra_arr, (0, per_bin_dim - len(extra_arr)))
    else:
        extra_arr = extra_arr[:per_bin_dim]

    return np.vstack([base, extra_arr])   # (n_bins+1, per_bin_dim)


def _enhanced_flat(acc: np.ndarray, gyro: np.ndarray | None = None,
                   pitch: np.ndarray | None = None, roll: np.ndarray | None = None,
                   n_bins: int = N_BINS, include_gyro: bool = False,
                   include_attention: bool = False,
                   include_multiscale: bool = False) -> np.ndarray | None:
    """Flat features ± cross-domain augmentation (simply concatenated)."""
    base = _build_flat(acc, gyro, pitch, roll,
                       n_bins=n_bins, include_gyro=include_gyro)
    if base is None:
        return None

    parts = [base]
    if include_attention:
        parts.append(extract_attention_entropy(acc))
    if include_multiscale:
        parts.append(extract_multi_scale_features(acc))

    return np.concatenate(parts).astype(np.float32)


# ============================================================
# Per-stage dataset builders
# ============================================================

def build_stage1(data_root: str,
                 include_attention: bool = False,
                 include_multiscale: bool = False):
    """
    Stage 1 — binary fall gate (FALL vs ADL).

    Returns
    -------
    X_bins : np.ndarray  (N, n_bins[+1], per_bin_dim)
    X_flat : np.ndarray  (N, flat_dim)
    y      : np.ndarray  (N,)  str labels 'FALL' | 'ADL'
    groups : np.ndarray  (N,)  subject IDs for LOSO splitting
    """
    X_bins, X_flat, y, groups = [], [], [], []

    for code in FALL_CODES:
        for subj in range(1, 68):
            seg = _load_segment(data_root, code, subj)
            if seg is None:
                continue
            seg = seg.iloc[:MAX_SAMPLES]
            acc  = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values

            fb = _enhanced_binned(acc, gyro, include_attention=include_attention,
                                  include_multiscale=include_multiscale)
            fc = _enhanced_flat(acc, gyro, include_attention=include_attention,
                                include_multiscale=include_multiscale)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append('FALL');  groups.append(subj)

    for code in ADL_CODES_11:
        for subj in range(1, 68):
            seg = _load_segment(data_root, code, subj)
            if seg is None:
                continue
            seg = seg.iloc[:MAX_SAMPLES]
            acc  = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values

            fb = _enhanced_binned(acc, gyro, include_attention=include_attention,
                                  include_multiscale=include_multiscale)
            fc = _enhanced_flat(acc, gyro, include_attention=include_attention,
                                include_multiscale=include_multiscale)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append('ADL');   groups.append(subj)

    return (np.array(X_bins), np.array(X_flat),
            np.array(y),      np.array(groups))


def build_stage2a(data_root: str,
                  include_attention: bool = False,
                  include_multiscale: bool = False):
    """
    Stage 2a — fall subtype classification (BSC / FKL / FOL / SDL).

    Accelerometer only (no gyro), matching the best-baseline setup.
    """
    X_bins, X_flat, y, groups = [], [], [], []

    for code in FALL_CODES:
        for subj in range(1, 68):
            seg = _load_segment(data_root, code, subj)
            if seg is None:
                continue
            seg = seg.iloc[:MAX_SAMPLES]
            acc  = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values

            fb = _enhanced_binned(acc, gyro, include_attention=include_attention,
                                  include_multiscale=include_multiscale)
            fc = _enhanced_flat(acc, gyro, include_attention=include_attention,
                                include_multiscale=include_multiscale)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append(code);    groups.append(subj)

    return (np.array(X_bins), np.array(X_flat),
            np.array(y),      np.array(groups))


def build_stage2b(data_root: str,
                  include_attention: bool = False,
                  include_multiscale: bool = False):
    """
    Stage 2b — ADL 11-class recognition.

    Includes gyroscope and orientation columns (pitch / roll), matching
    the best-baseline Stage 2b setup.
    """
    X_bins, X_flat, y, groups = [], [], [], []

    for code in ADL_CODES_11:
        for subj in range(1, 68):
            seg = _load_segment(data_root, code, subj)
            if seg is None:
                continue
            seg = seg.iloc[:MAX_SAMPLES]
            acc   = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro  = seg[['gyro_x', 'gyro_y', 'gyro_z']].values
            pitch = seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg))
            roll  = seg['roll'].values  if 'roll'  in seg.columns else np.zeros(len(seg))

            fb = _enhanced_binned(acc, gyro, include_gyro=True,
                                  include_attention=include_attention,
                                  include_multiscale=include_multiscale)
            fc = _enhanced_flat(acc, gyro, pitch, roll, include_gyro=True,
                                include_attention=include_attention,
                                include_multiscale=include_multiscale)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append(code);    groups.append(subj)

    return (np.array(X_bins), np.array(X_flat),
            np.array(y),      np.array(groups))
