"""
Dataset builders for the Kalman Fusion + Relative Position Encoding ablation.

This module extends the first-round ablation builders
(src/ablation_builders.py) by keeping Temporal Attention always
enabled (it was the best variant from the first ablation) and then
testing Kalman Fusion and RPE features on top of that baseline.

Ablation variant mapping
------------------------
  Baseline (Attention)         → attention=True, kalman=False, rpe=False
  Kalman Fusion                → attention=True, kalman=True,  rpe=False
  Relative Position Encoding   → attention=True, kalman=False, rpe=True
  Kalman + RPE                 → attention=True, kalman=True,  rpe=True

Stage-specific notes
--------------------
  Stage 1  (binary, acc-only):  Kalman is skipped (no gyro in baseline).
  Stage 2a (fall subtypes):     Kalman is skipped to match acc-only
                                baseline; RPE is still tested.
  Stage 2b (ADL, acc+gyro):     Full grid — both Kalman and RPE tested.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from .cross_domain_features import extract_attention_entropy
from .kalman_rpe_features import kalman_fusion_features, extract_relative_position_features

# ============================================================
# Constants
# ============================================================
FALL_CODES   = ['BSC', 'FKL', 'FOL', 'SDL']
ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
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
# Low-level feature builders (reproduced locally for standalone use)
# ============================================================

def _build_binned(acc: np.ndarray, gyro: np.ndarray | None,
                  n_bins: int = N_BINS, include_gyro: bool = False) -> np.ndarray | None:
    if len(acc) < n_bins:
        return None
    bin_edges = np.linspace(0, len(acc), n_bins + 1).astype(int)
    bins = []
    for i in range(n_bins):
        a = acc[bin_edges[i]:bin_edges[i + 1]]
        if len(a) == 0:
            a = acc[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
        ae  = (a ** 2).sum(axis=0); ae = ae / (ae.sum() + 1e-8)
        am  = np.sqrt((a ** 2).sum(axis=1))
        feat = list(ae) + [am.mean(), am.std()]
        if include_gyro and gyro is not None:
            g = gyro[bin_edges[i]:bin_edges[i + 1]]
            if len(g) == 0:
                g = gyro[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
            ge = (g ** 2).sum(axis=0); ge = ge / (ge.sum() + 1e-8)
            feat += list(ge)
        bins.append(feat)
    return np.array(bins, dtype=np.float32)


def _build_flat(acc: np.ndarray, gyro: np.ndarray | None = None,
                pitch: np.ndarray | None = None, roll: np.ndarray | None = None,
                n_bins: int = N_BINS, include_gyro: bool = False) -> np.ndarray | None:
    if len(acc) < n_bins:
        return None
    from scipy.signal import correlate, find_peaks

    bin_edges = np.linspace(0, len(acc), n_bins + 1).astype(int)
    profile = []
    for axis in range(3):
        e  = acc[:, axis] ** 2
        ap = np.array([e[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(n_bins)])
        profile.append(ap / (ap.sum() + 1e-8))
    profile = np.concatenate(profile)

    acc_mag = np.sqrt((acc ** 2).sum(axis=1))
    acc_stats = np.array([
        skew(acc_mag), kurtosis(acc_mag),
        acc_mag.var(), acc_mag.mean(),
        np.max(np.abs(np.gradient(acc_mag))),
    ])
    parts = [profile, acc_stats]

    if include_gyro and gyro is not None:
        gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))
        ge = gyro ** 2; tge = ge.sum() + 1e-8
        gx, gy, gz = ge[:, 0].sum()/tge, ge[:, 1].sum()/tge, ge[:, 2].sum()/tge
        ac = acc_mag - acc_mag.mean()
        acorr = correlate(ac, ac, mode='full')[len(ac) - 1:]
        acorr = acorr / (acorr[0] + 1e-8)
        pk, _ = find_peaks(acorr[5:], height=0.2)
        autocorr  = float(acorr[pk[0] + 5]) if len(pk) > 0 else 0.0
        roll_range = float(roll.max() - roll.min()) if (roll is not None and len(roll) > 0) else 0.0
        gyro_vec  = np.array([gx, gy, gz,
                               skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean(),
                               roll_range, autocorr])
        parts.append(gyro_vec)

    return np.concatenate(parts).astype(np.float32)


# ============================================================
# Enhanced wrappers  (Attention always ON + optional Kalman/RPE)
# ============================================================

def _enhanced_binned(acc: np.ndarray, gyro: np.ndarray | None,
                     n_bins: int = N_BINS, include_gyro: bool = False,
                     include_kalman: bool = False,
                     include_rpe: bool = False) -> np.ndarray | None:
    """
    Binned features with Temporal Attention (always on) ± Kalman / RPE.

    Extra features are packed into one or two additional pseudo-bin
    rows (padded/truncated to per_bin_dim) so the BiLSTM tensor shape
    stays consistent across variants.
    """
    base = _build_binned(acc, gyro, n_bins=n_bins, include_gyro=include_gyro)
    if base is None:
        return None

    per_bin_dim = base.shape[1]

    # Temporal attention — always include (baseline for this round)
    attn = extract_attention_entropy(acc)

    extra: list[float] = list(attn)

    if include_kalman and gyro is not None:
        extra.extend(kalman_fusion_features(acc, gyro).tolist())
    if include_rpe:
        extra.extend(extract_relative_position_features(acc).tolist())

    # Pack into a single extra row aligned to per_bin_dim
    extra_arr = np.array(extra, dtype=np.float32)
    if len(extra_arr) < per_bin_dim:
        extra_arr = np.pad(extra_arr, (0, per_bin_dim - len(extra_arr)))
    else:
        extra_arr = extra_arr[:per_bin_dim]

    return np.vstack([base, extra_arr])


def _enhanced_flat(acc: np.ndarray, gyro: np.ndarray | None = None,
                   pitch: np.ndarray | None = None, roll: np.ndarray | None = None,
                   n_bins: int = N_BINS, include_gyro: bool = False,
                   include_kalman: bool = False,
                   include_rpe: bool = False) -> np.ndarray | None:
    """Flat features with Temporal Attention (always on) ± Kalman / RPE."""
    base = _build_flat(acc, gyro, pitch, roll, n_bins=n_bins, include_gyro=include_gyro)
    if base is None:
        return None

    parts = [base, extract_attention_entropy(acc)]

    if include_kalman and gyro is not None:
        parts.append(kalman_fusion_features(acc, gyro))
    if include_rpe:
        parts.append(extract_relative_position_features(acc))

    return np.concatenate(parts).astype(np.float32)


# ============================================================
# Per-stage dataset builders
# ============================================================

def build_stage1_krpe(data_root: str,
                      include_kalman: bool = False,
                      include_rpe: bool = False):
    """
    Stage 1 — binary fall gate (FALL vs ADL).

    Accelerometer only; Kalman is silently ignored (requires gyro).
    """
    X_bins, X_flat, y, groups = [], [], [], []

    for code in FALL_CODES + ADL_CODES_11:
        label = 'FALL' if code in FALL_CODES else 'ADL'
        for subj in range(1, 68):
            seg = _load_segment(data_root, code, subj)
            if seg is None:
                continue
            seg  = seg.iloc[:MAX_SAMPLES]
            acc  = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values

            # Kalman skipped at Stage 1 (acc-only baseline)
            fb = _enhanced_binned(acc, gyro, include_gyro=False,
                                  include_kalman=False,
                                  include_rpe=include_rpe)
            fc = _enhanced_flat(acc, gyro, include_gyro=False,
                                include_kalman=False,
                                include_rpe=include_rpe)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append(label);   groups.append(subj)

    return (np.array(X_bins), np.array(X_flat),
            np.array(y),      np.array(groups))


def build_stage2a_krpe(data_root: str,
                       include_kalman: bool = False,
                       include_rpe: bool = False):
    """
    Stage 2a — fall subtype classification (BSC / FKL / FOL / SDL).

    Accelerometer only to match prior baseline; Kalman silently skipped.
    """
    X_bins, X_flat, y, groups = [], [], [], []

    for code in FALL_CODES:
        for subj in range(1, 68):
            seg = _load_segment(data_root, code, subj)
            if seg is None:
                continue
            seg  = seg.iloc[:MAX_SAMPLES]
            acc  = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values

            fb = _enhanced_binned(acc, gyro, include_gyro=False,
                                  include_kalman=False,
                                  include_rpe=include_rpe)
            fc = _enhanced_flat(acc, gyro, include_gyro=False,
                                include_kalman=False,
                                include_rpe=include_rpe)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append(code);    groups.append(subj)

    return (np.array(X_bins), np.array(X_flat),
            np.array(y),      np.array(groups))


def build_stage2b_krpe(data_root: str,
                       include_kalman: bool = False,
                       include_rpe: bool = False):
    """
    Stage 2b — ADL 11-class recognition.

    Full sensor suite (acc + gyro + orientation).  Both Kalman and RPE
    are supported here.
    """
    X_bins, X_flat, y, groups = [], [], [], []

    for code in ADL_CODES_11:
        for subj in range(1, 68):
            seg = _load_segment(data_root, code, subj)
            if seg is None:
                continue
            seg   = seg.iloc[:MAX_SAMPLES]
            acc   = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro  = seg[['gyro_x', 'gyro_y', 'gyro_z']].values
            pitch = seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg))
            roll  = seg['roll'].values  if 'roll'  in seg.columns else np.zeros(len(seg))

            fb = _enhanced_binned(acc, gyro, include_gyro=True,
                                  include_kalman=include_kalman,
                                  include_rpe=include_rpe)
            fc = _enhanced_flat(acc, gyro, pitch, roll, include_gyro=True,
                                include_kalman=include_kalman,
                                include_rpe=include_rpe)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append(code);    groups.append(subj)

    return (np.array(X_bins), np.array(X_flat),
            np.array(y),      np.array(groups))
