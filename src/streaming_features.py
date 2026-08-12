"""
Streaming-aware cross-domain feature extractors for the Stage 2b ablation.

The functions here mirror their counterparts in cross_domain_features.py
and kalman_rpe_features.py but are adapted for short sliding windows
(default 200 samples / ~2.3 s at 87 Hz) rather than full activity
segments.  Key differences:

  * extract_attention_entropy_streaming() — guards against windows that
    are shorter than the requested segment size and returns zeros rather
    than raising.
  * extract_relative_position_features_streaming() — identical math to
    the segment-level version but explicitly documented for the streaming
    context; always returns exactly 10 values.

Both functions are wrappers around their parent implementations with the
added short-window safety guards expected by the streaming pipeline.
"""

import numpy as np


# ============================================================
# Helpers
# ============================================================

def _acc_mag(acc_window: np.ndarray) -> np.ndarray:
    """Return scalar acc magnitude for a (T, 3) window."""
    return np.sqrt((acc_window ** 2).sum(axis=1))


# ============================================================
# 1. Temporal Attention Entropy  (streaming variant)
# ============================================================

def extract_attention_entropy_streaming(
    acc_window: np.ndarray,
    segment_size: int = 50,
) -> np.ndarray:
    """
    Temporal attention entropy features from a sliding window.

    Parameters
    ----------
    acc_window   : np.ndarray, shape (W, 3)  — one sliding window
    segment_size : int  — samples per temporal segment (default 50).
                   If the window is shorter, a single segment is used.

    Returns
    -------
    np.ndarray, shape (3,)
        [attention_entropy, attention_concentration, attention_uniformity]
        Returns zeros if the window contains fewer than 2 samples.
    """
    if len(acc_window) < 2:
        return np.zeros(3, dtype=np.float32)

    mag = _acc_mag(acc_window)
    n_segments = max(1, len(mag) // segment_size)
    segs = np.array_split(mag, n_segments)

    energies = np.array([np.sum(s ** 2) for s in segs], dtype=np.float64)
    energies = energies / (energies.sum() + 1e-8)

    entropy       = float(-np.sum(energies * np.log(energies + 1e-8)))
    concentration = float(np.max(energies))
    uniformity    = float(1.0 - (np.max(energies) - np.min(energies)))

    return np.array([entropy, concentration, uniformity], dtype=np.float32)


# ============================================================
# 2. Relative Position Encoding  (streaming variant)
# ============================================================

def extract_relative_position_features_streaming(
    acc_window: np.ndarray,
    lags:  list[int] | None = None,
    steps: list[int] | None = None,
    high_pct: float = 80.0,
) -> np.ndarray:
    """
    Relative position encoding features from a sliding window.

    Parameters
    ----------
    acc_window : np.ndarray, shape (W, 3)
    lags       : autocorrelation lags  (default [1, 5, 10, 20])
    steps      : change-rate step sizes (default [1, 5, 10])
    high_pct   : percentile for 'high-activity' threshold (default 80)

    Returns
    -------
    np.ndarray, shape (10,)
        Always exactly 10 values; zeros when the window is too short.

        Index layout:
          0-3  : lag autocorrelations at lags [1, 5, 10, 20]
          4-6  : relative change-rate ratios at steps [1, 5, 10]
          7    : position-weighted mean of acc magnitude
          8    : position-weighted std of acc magnitude
          9    : Shannon entropy of high-activity sample positions
    """
    if lags  is None: lags  = [1, 5, 10, 20]
    if steps is None: steps = [1, 5, 10]

    mag = _acc_mag(acc_window)
    n   = len(mag)

    if n < 20:
        return np.zeros(10, dtype=np.float32)

    features: list[float] = []

    # Lag autocorrelations
    for lag in lags:
        if lag < n:
            c = np.corrcoef(mag[:-lag], mag[lag:])[0, 1]
            features.append(0.0 if np.isnan(c) else float(c))
        else:
            features.append(0.0)

    # Relative change-rate ratios
    diffs = np.diff(mag)
    for step in steps:
        if len(diffs) > step:
            later   = np.mean(np.abs(diffs[step:]))
            earlier = np.mean(np.abs(diffs[:step]))
            ratio   = float(later / (earlier + 1e-8))
            features.append(0.0 if np.isnan(ratio) else ratio)
        else:
            features.append(0.0)

    # Position-weighted statistics
    weights = np.arange(1, n + 1, dtype=np.float64) / n
    w_sum   = weights.sum()
    w_mean  = float(np.sum(weights * mag) / w_sum)
    w_std   = float(np.sqrt(np.sum(weights * (mag - w_mean) ** 2) / w_sum))
    features.extend([w_mean, w_std])

    # High-activity position entropy
    threshold = np.percentile(mag, high_pct)
    hi_idx    = np.where(mag > threshold)[0]
    if len(hi_idx) > 0:
        rel = (hi_idx + 1) / n
        pos_entropy = float(-np.sum(rel * np.log(rel + 1e-8)))
    else:
        pos_entropy = 0.0
    features.append(pos_entropy)

    # Guarantee exactly 10 values
    features = features[:10]
    while len(features) < 10:
        features.append(0.0)

    return np.array(features, dtype=np.float32)
