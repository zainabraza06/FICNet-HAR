"""
Kalman Fusion and Relative Position Encoding feature extractors.

These are the second wave of cross-domain adaptations, building on top
of the Temporal Attention features (best from the first ablation round).

1. Kalman Sensor Fusion  (Robotics)
   Inspired by the Kalman Filter used in robotics / navigation for
   optimal sensor fusion under Gaussian noise assumptions.
   Here we derive a simplified scalar Kalman gain from the noise levels
   of the accelerometer and gyroscope signals, fuse them into a single
   trajectory, and extract statistics from the fused signal along with
   confidence/noise-ratio features.

   Intuition
   ---------
   Falls produce a brief, high-amplitude acc spike while the gyroscope
   continues rotating afterwards; the noise-weighted fusion and
   resulting Kalman gain capture this asymmetric trust relationship.

2. Relative Position Encoding  (NLP Transformers)
   Inspired by Shaw et al. (2018) relative position representations
   used in Transformer self-attention.  Instead of absolute positional
   embeddings, RPE models the *relationship* between token positions.
   Here, "tokens" are time-steps of the acc-magnitude signal.

   Concrete features extracted:
     - Lag autocorrelations at lags {1, 5, 10, 20}
     - Relative change-rate ratios across step sizes {1, 5, 10}
     - Linearly position-weighted mean and std
     - Entropy of the temporal distribution of high-activity samples

   Intuition
   ---------
   ADL cycles (walking, jogging) have strong short-lag autocorrelations;
   falls have near-zero autocorrelation after the impact.  The position-
   weighted statistics capture whether the peak energy occurs near the
   start (fall) or is spread evenly (ADL).
"""

import numpy as np


# ============================================================
# 1. Kalman Sensor Fusion
# ============================================================

def kalman_fusion_features(acc_data: np.ndarray, gyro_data: np.ndarray) -> np.ndarray:
    """
    Simplified Kalman-inspired sensor fusion between accelerometer
    and gyroscope signals.

    Parameters
    ----------
    acc_data  : np.ndarray, shape (T, 3)
    gyro_data : np.ndarray, shape (T, 3)

    Returns
    -------
    np.ndarray, shape (10,)
        [fused_mean, fused_std, fused_max, fused_min, fused_median,
         fused_max_jerk, kalman_gain, gyro_trust, noise_ratio,
         acc_gyro_correlation]

    Notes
    -----
    kalman_gain   — scalar in [0, 1]: fraction of trust placed in the
                    accelerometer.  High gain → acc is smoother (low
                    noise) relative to gyro.
    gyro_trust    — 1 - kalman_gain
    noise_ratio   — acc_std / gyro_std: >1 means acc is noisier.
    acc_gyro_corr — Pearson correlation between |acc| and |gyro|
                    magnitudes; captures timing alignment of the two
                    sensor peaks.
    """
    acc_mag  = np.sqrt((acc_data  ** 2).sum(axis=1))
    gyro_mag = np.sqrt((gyro_data ** 2).sum(axis=1))

    acc_noise  = float(np.std(acc_mag))
    gyro_noise = float(np.std(gyro_mag))

    # Kalman gain: give more weight to the less noisy sensor
    kalman_gain = acc_noise / (acc_noise + gyro_noise + 1e-8)

    fused = kalman_gain * acc_mag + (1.0 - kalman_gain) * gyro_mag

    corr = (
        float(np.corrcoef(acc_mag, gyro_mag)[0, 1])
        if len(acc_mag) > 10
        else 0.0
    )
    if np.isnan(corr):
        corr = 0.0

    return np.array(
        [
            float(np.mean(fused)),
            float(np.std(fused)),
            float(np.max(fused)),
            float(np.min(fused)),
            float(np.median(fused)),
            float(np.max(np.abs(np.gradient(fused)))),
            kalman_gain,
            1.0 - kalman_gain,
            acc_noise / (gyro_noise + 1e-8),
            corr,
        ],
        dtype=np.float32,
    )


# ============================================================
# 2. Relative Position Encoding
# ============================================================

def extract_relative_position_features(
    acc_data: np.ndarray,
    lags: list[int] | None = None,
    steps: list[int] | None = None,
    high_pct: float = 80.0,
) -> np.ndarray:
    """
    Temporal relative-position features adapted from NLP Transformer
    relative position encoding (Shaw et al., 2018).

    Parameters
    ----------
    acc_data  : np.ndarray, shape (T, 3)
    lags      : autocorrelation lags to compute  (default [1, 5, 10, 20])
    steps     : step sizes for change-rate ratios (default [1, 5, 10])
    high_pct  : percentile threshold for 'high-activity' samples (default 80)

    Returns
    -------
    np.ndarray, shape (10,)
        [lag_corr × 4, rel_change_ratio × 3, pos_weighted_mean,
         pos_weighted_std, high_activity_pos_entropy]

    Notes
    -----
    Always returns exactly 10 values (zero-padded when signal is too
    short to compute a particular lag / step).
    """
    if lags  is None: lags  = [1, 5, 10, 20]
    if steps is None: steps = [1, 5, 10]

    acc_mag = np.sqrt((acc_data ** 2).sum(axis=1))
    n       = len(acc_mag)

    features: list[float] = []

    # --- Lag autocorrelations ---
    for lag in lags:
        if lag < n:
            c = np.corrcoef(acc_mag[:-lag], acc_mag[lag:])[0, 1]
            features.append(0.0 if np.isnan(c) else float(c))
        else:
            features.append(0.0)

    # --- Relative change-rate ratios ---
    diffs = np.diff(acc_mag)
    for step in steps:
        if len(diffs) > step:
            later  = np.mean(np.abs(diffs[step:]))
            earlier = np.mean(np.abs(diffs[:step]))
            features.append(float(later / (earlier + 1e-8)))
        else:
            features.append(0.0)

    # --- Position-weighted statistics ---
    weights      = np.arange(1, n + 1, dtype=np.float64) / n
    w_sum        = weights.sum()
    w_mean       = float(np.sum(weights * acc_mag) / w_sum)
    w_std        = float(
        np.sqrt(np.sum(weights * (acc_mag - w_mean) ** 2) / w_sum)
    )
    features.extend([w_mean, w_std])

    # --- High-activity position entropy ---
    threshold             = np.percentile(acc_mag, high_pct)
    high_idx              = np.where(acc_mag > threshold)[0]
    if len(high_idx) > 0:
        rel_pos           = (high_idx + 1) / n          # avoid log(0)
        pos_entropy       = float(-np.sum(rel_pos * np.log(rel_pos + 1e-8)))
    else:
        pos_entropy = 0.0
    features.append(pos_entropy)

    # Ensure exactly 10 values
    features = features[:10]
    while len(features) < 10:
        features.append(0.0)

    return np.array(features, dtype=np.float32)
