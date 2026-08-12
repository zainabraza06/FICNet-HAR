"""
Cross-domain feature extractors adapted from Computer Vision techniques.

Two novel feature families are implemented here:

1. Temporal Attention Entropy
   Inspired by spatial attention maps used in vision transformers and
   convolutional attention modules (CBAM, SE-Net).  Instead of pixels,
   we compute a soft "attention" distribution over temporal energy
   segments and derive entropy / concentration statistics from it.

   Intuition
   ---------
   Falls produce a sharp, localised energy spike → low entropy (focused
   attention).  Regular ADL cycles spread energy more evenly → higher
   entropy (distributed attention).

2. Multi-Scale Temporal Features  (Feature Pyramid Network analogy)
   Inspired by FPN backbones used in object detection.  FPNs build a
   pyramid of feature maps at multiple spatial resolutions and fuse
   information across scales.  Here we downsample the acceleration
   magnitude at several temporal scales and extract statistics at each
   level, then add cross-scale ratio features.

   Intuition
   ---------
   A fall may look very different at 1× vs 4× temporal downsampling;
   cross-scale ratios capture this coarse-vs-fine discrepancy which
   plain single-scale features miss.
"""

import numpy as np
from scipy.stats import skew, kurtosis


# ============================================================
# 1. Temporal Attention Entropy
# ============================================================

def extract_attention_entropy(acc_data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """
    Compute attention-entropy features from raw accelerometer data.

    Adapted from CV spatial-attention mechanisms (CBAM, SE-Net) applied
    to the temporal domain.

    Parameters
    ----------
    acc_data : np.ndarray, shape (T, 3)
        Raw tri-axial accelerometer readings.
    window_size : int
        Approximate number of samples per temporal segment.

    Returns
    -------
    np.ndarray, shape (3,)
        [attention_entropy, attention_concentration, attention_uniformity]

    Notes
    -----
    * attention_entropy       — Shannon entropy of the energy distribution.
                                Low → event is localised; high → spread evenly.
    * attention_concentration — Maximum segment energy fraction.
                                High → single dominant segment (typical of falls).
    * attention_uniformity    — 1 − (max − min) of the energy distribution.
                                High → flat distribution (typical of ADLs).
    """
    acc_mag = np.sqrt((acc_data ** 2).sum(axis=1))

    n_segments = max(1, len(acc_mag) // window_size)
    segments = np.array_split(acc_mag, n_segments)

    # Normalised energy distribution (acts as an attention map)
    segment_energies = np.array([np.sum(s ** 2) for s in segments])
    segment_energies = segment_energies / (segment_energies.sum() + 1e-8)

    attention_entropy = -np.sum(segment_energies * np.log(segment_energies + 1e-8))
    attention_concentration = float(np.max(segment_energies))
    attention_uniformity = 1.0 - float(np.max(segment_energies) - np.min(segment_energies))

    return np.array(
        [attention_entropy, attention_concentration, attention_uniformity],
        dtype=np.float32,
    )


# ============================================================
# 2. Multi-Scale Temporal Features  (FPN analogy)
# ============================================================

def extract_multi_scale_features(
    acc_data: np.ndarray,
    scales: list[int] | None = None,
) -> np.ndarray:
    """
    Extract statistics at multiple temporal scales, analogous to
    Feature Pyramid Networks (FPN) used in object detection.

    Parameters
    ----------
    acc_data : np.ndarray, shape (T, 3)
        Raw tri-axial accelerometer readings.
    scales : list[int]
        Downsampling factors.  Default ``[1, 2, 4]`` — fine, medium, coarse.

    Returns
    -------
    np.ndarray, shape (8 * len(scales) + len(scales) - 1,)
        Concatenated per-scale statistics followed by cross-scale ratios.

    Notes
    -----
    Per-scale statistics (8 per scale):
        mean, std, max, min, median, skewness, kurtosis, max_jerk

    Cross-scale ratios  (len(scales) − 1):
        mean(scale_i) / mean(scale_{i+1})   for each adjacent pair.
        Captures coarse-vs-fine amplitude discrepancy.
    """
    if scales is None:
        scales = [1, 2, 4]

    acc_mag = np.sqrt((acc_data ** 2).sum(axis=1))

    features: list[float] = []
    scale_means: list[float] = []

    for scale in scales:
        downsampled = acc_mag[::scale] if scale > 1 else acc_mag

        level_mean = float(np.mean(downsampled))
        scale_means.append(level_mean)

        features.extend([
            level_mean,
            float(np.std(downsampled)),
            float(np.max(downsampled)),
            float(np.min(downsampled)),
            float(np.median(downsampled)),
            float(skew(downsampled)),
            float(kurtosis(downsampled)),
            float(np.max(np.abs(np.gradient(downsampled)))),
        ])

    # Cross-scale ratios
    for i in range(len(scales) - 1):
        ratio = scale_means[i] / (scale_means[i + 1] + 1e-8)
        features.append(ratio)

    return np.array(features, dtype=np.float32)
