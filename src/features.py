"""
Finalized feature extraction pipeline — three stages.

Stage 1 (fall gate, binary):    5-bin |acc| energy profile
Stage 2a (fall subtype, 4-cls): 34-dim (15 profile + 10 scalar + 9 gyro)
Stage 2b (ADL, 11-cls):         35-dim (15 profile + 6 spectral/stat +
                                          8 gyro + 6 orientation)

All feature functions operate on a single labeled segment (a DataFrame
already filtered to one activity code, via MobiActLoader.get_segment).

Evidence trail (see project EDA log for full derivation):
- Raw-waveform correlation: weak (falls 0.03-0.11, ADLs ~0.01) — REJECTED
- Energy-distribution profile (n_bins=5): strong (falls 0.34-0.57,
  ADLs near 0) — ACCEPTED as Stage 1 representation
- Magnitude-only profile insufficient for fall SUBTYPE (correlation
  0.94-0.99 between some pairs) — per-axis profile required for 2a
- Gyroscope axis-fraction/moment features: consistently the strongest
  discriminators for hard confusion pairs (BSC/SDL d=2.11, CHU/CSO
  d=-2.97) — ACCEPTED, added to both 2a and 2b
- Orientation canonicalization (world-frame rotation): REJECTED —
  Android's magnetometer-derived azimuth is unreliable indoors
- Body-metric conditioning (height/weight/BMI): REJECTED — no
  significant correlation with fall-impact magnitude
- Cross-modal jerk*gyro joint signal: REJECTED — underperforms
  plain |acc| energy profile
- Hierarchical hard-routing classification: REJECTED — routing
  errors compound and underperform flat classification
- K-shot personalization calibration: REJECTED — confusion pairs
  are structural (physical similarity), not subject-specific
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, correlate, welch


# ============================================================
# Low-level signal utilities
# ============================================================

def _acc_magnitude(seg):
    return np.sqrt(seg['acc_x'] ** 2 + seg['acc_y'] ** 2 + seg['acc_z'] ** 2)


def _gyro_magnitude(seg):
    return np.sqrt(seg['gyro_x'] ** 2 + seg['gyro_y'] ** 2 + seg['gyro_z'] ** 2)


# ============================================================
# Stage 1: Fall gate representation
# ============================================================

def energy_profile(seg, n_bins: int = 5, axis: str | None = None, max_samples: int | None = None):
    """
    Binned, normalized energy-distribution profile.

    axis=None -> magnitude profile (n_bins,) — used for Stage 1 (fall gate)
    axis='per_axis' -> per-axis profile (3*n_bins,) — used for Stage 2a/2b

    max_samples caps the window (used for long ADL files like STD/WAL,
    which run for minutes; a representative chunk is used instead of
    the full file).
    """
    if max_samples is not None:
        seg = seg.iloc[:max_samples]
    if len(seg) < n_bins:
        return None

    acc = seg[['acc_x', 'acc_y', 'acc_z']].values
    bin_edges = np.linspace(0, len(acc), n_bins + 1).astype(int)

    if axis == 'per_axis':
        profile = []
        for a in range(3):
            energy = acc[:, a] ** 2
            binned = np.array([energy[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(n_bins)])
            total = binned.sum()
            profile.append(binned / (total + 1e-8))
        return np.concatenate(profile)  # (3*n_bins,)
    else:
        energy = (acc ** 2).sum(axis=1)
        binned = np.array([energy[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(n_bins)])
        total = binned.sum()
        return binned / (total + 1e-8)  # (n_bins,)


# ============================================================
# Stage 2a: Fall subtype scalar + gyro features
# ============================================================

def fall_subtype_scalar_features(seg) -> dict | None:
    """10 scalar features validated for fall-type discrimination
    (peak direction, sub-peak count, timing, jerk, distribution shape)."""
    if len(seg) < 10:
        return None

    acc = seg[['acc_x', 'acc_y', 'acc_z']].values
    gyro_mag = _gyro_magnitude(seg).values
    acc_mag = _acc_magnitude(seg).values

    feats = {}
    peak_idx = int(np.argmax(acc_mag))
    peak_vec = acc[peak_idx]
    denom = np.abs(peak_vec).sum() + 1e-8
    feats['peak_axis_x_frac'] = abs(peak_vec[0]) / denom
    feats['peak_axis_y_frac'] = abs(peak_vec[1]) / denom
    feats['peak_axis_z_frac'] = abs(peak_vec[2]) / denom
    feats['time_to_peak_frac'] = peak_idx / len(acc_mag)

    peaks, _ = find_peaks(acc_mag, height=acc_mag.mean() + acc_mag.std(), distance=5)
    feats['n_subpeaks'] = len(peaks)

    feats['acc_skew'] = skew(acc_mag)
    feats['acc_kurtosis'] = kurtosis(acc_mag)

    baseline = 9.8
    settled = np.where(np.abs(acc_mag[peak_idx:] - baseline) < 1.0)[0]
    feats['settle_time_frac'] = (settled[0] / len(acc_mag)) if len(settled) > 0 else 1.0

    gyro_peak_idx = int(np.argmax(gyro_mag))
    feats['gyro_acc_peak_offset'] = (gyro_peak_idx - peak_idx) / len(acc_mag)

    jerk = np.gradient(acc_mag)
    feats['max_jerk'] = np.max(np.abs(jerk))

    return feats


# ============================================================
# Shared gyro / orientation feature block
# (used by both Stage 2a and Stage 2b — this is the single
# most consistently useful feature family found in EDA)
# ============================================================

def gyro_orientation_features(seg, max_samples: int | None = None) -> dict | None:
    """
    Gyroscope axis-fraction/moment features + orientation-delta
    features. Validated as the strongest discriminator across every
    hard confusion pair tested (fall subtypes, stairs up/down,
    car-seat/chair-seat, sit/stand).
    """
    if max_samples is not None:
        seg = seg.iloc[:max_samples]
    if len(seg) < 20:
        return None

    acc = seg[['acc_x', 'acc_y', 'acc_z']].values
    gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values
    acc_mag = np.sqrt((acc ** 2).sum(axis=1))
    gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))

    feats = {}

    gyro_energy = gyro ** 2
    total = gyro_energy.sum() + 1e-8
    feats['gyro_x_frac'] = gyro_energy[:, 0].sum() / total
    feats['gyro_y_frac'] = gyro_energy[:, 1].sum() / total
    feats['gyro_z_frac'] = gyro_energy[:, 2].sum() / total
    feats['gyro_mean_mag'] = gyro_mag.mean()
    feats['gyro_skew'] = skew(gyro_mag)
    feats['gyro_kurtosis'] = kurtosis(gyro_mag)

    if all(c in seg.columns for c in ['azimuth', 'pitch', 'roll']):
        pitch = seg['pitch'].values
        roll = seg['roll'].values
        feats['pitch_net_delta'] = pitch[-1] - pitch[0]
        feats['roll_net_delta'] = roll[-1] - roll[0]
        feats['pitch_range'] = pitch.max() - pitch.min()
        feats['roll_range'] = roll.max() - roll.min()
        feats['pitch_std'] = pitch.std()
        feats['roll_std'] = roll.std()

    if len(acc) > 10:
        cx = np.corrcoef(acc[:, 0], acc[:, 2])[0, 1]
        feats['xz_corr'] = cx if not np.isnan(cx) else 0.0

    ac_signal = acc_mag - acc_mag.mean()
    autocorr = correlate(ac_signal, ac_signal, mode='full')[len(ac_signal) - 1:]
    autocorr = autocorr / (autocorr[0] + 1e-8)
    ac_peaks, _ = find_peaks(autocorr[5:], height=0.2)
    feats['autocorr_peak_strength'] = autocorr[ac_peaks[0] + 5] if len(ac_peaks) > 0 else 0.0

    return feats


# ============================================================
# Stage 2b: ADL spectral/statistical features
# ============================================================

def dominant_frequency_features(seg, fs: float = 87.0, max_samples: int = 2000) -> dict | None:
    """FFT-based dominant frequency + spectral power concentration.
    Effective for periodic ADLs (walking, jogging, stairs); weak for
    short transitional activities (CSI/SCH/CHU) — retained regardless
    since it contributes to the overall feature vector."""
    seg = seg.iloc[:max_samples]
    if len(seg) < 50:
        return None
    acc_mag = _acc_magnitude(seg).values
    acc_mag = acc_mag - acc_mag.mean()

    freqs, psd = welch(acc_mag, fs=fs, nperseg=min(256, len(acc_mag)))
    valid = freqs > 0.3
    if not valid.any():
        return None
    dom_freq = freqs[valid][np.argmax(psd[valid])]
    power_concentration = psd[valid].max() / (psd[valid].sum() + 1e-8)
    return {'dom_freq': dom_freq, 'power_concentration': power_concentration}


# ============================================================
# Assembled feature vectors — the FINAL, validated pipelines
# ============================================================

def extract_stage1_features(seg, n_bins: int = 5, max_samples: int | None = None):
    """Fall gate (binary): 5-dim magnitude energy profile."""
    return energy_profile(seg, n_bins=n_bins, axis=None, max_samples=max_samples)


def extract_stage2a_features(seg, n_bins: int = 5):
    """
    Fall subtype (4-class): 34-dim.
    15 (per-axis profile) + 10 (scalar) + 9 (gyro/orientation subset)
    Validated LOSO accuracy: 0.816 (KNN-3, scaled)
    """
    profile = energy_profile(seg, n_bins=n_bins, axis='per_axis')
    if profile is None:
        return None
    scalars = fall_subtype_scalar_features(seg)
    if scalars is None:
        return None
    scalar_vec = np.array([
        scalars['peak_axis_x_frac'], scalars['peak_axis_y_frac'], scalars['peak_axis_z_frac'],
        scalars['n_subpeaks'], scalars['max_jerk'],
        scalars['time_to_peak_frac'], scalars['settle_time_frac'],
        scalars['acc_skew'], scalars['acc_kurtosis'],
        scalars['gyro_acc_peak_offset'],
    ])
    gyro = gyro_orientation_features(seg)
    if gyro is None:
        return None
    gyro_vec = np.array([
        gyro['gyro_x_frac'], gyro['gyro_y_frac'], gyro['gyro_z_frac'],
        gyro['gyro_skew'], gyro['gyro_kurtosis'], gyro['gyro_mean_mag'],
        gyro.get('roll_std', 0.0), gyro.get('pitch_net_delta', 0.0), gyro.get('xz_corr', 0.0),
    ])
    return np.concatenate([profile, scalar_vec, gyro_vec])  # 34-dim


def extract_stage2b_features(seg, n_bins: int = 5, max_samples: int = 1000):
    """
    ADL classification (9- or 11-class): 35-dim.
    15 (profile) + 6 (spectral/stat) + 8 (gyro) + 6 (orientation)
    Validated LOSO: 9-class 0.925 acc (LDA); 11-class 0.884 acc /
    0.865 balanced acc (LDA, uniform priors)
    """
    profile = energy_profile(seg, n_bins=n_bins, axis='per_axis', max_samples=max_samples)
    if profile is None:
        return None
    freq = dominant_frequency_features(seg)
    if freq is None:
        return None

    acc_mag = _acc_magnitude(seg)
    base_extra = np.array([
        freq['dom_freq'], freq['power_concentration'],
        acc_mag.var(), acc_mag.mean(), skew(acc_mag), kurtosis(acc_mag),
    ])

    gyro = gyro_orientation_features(seg)
    if gyro is None:
        return None
    gyro_vec = np.array([
        gyro['gyro_x_frac'], gyro['gyro_y_frac'], gyro['gyro_z_frac'],
        gyro['gyro_skew'], gyro['gyro_kurtosis'], gyro['gyro_mean_mag'],
        gyro.get('roll_range', 0.0), gyro.get('autocorr_peak_strength', 0.0),
    ])
    orient_vec = np.array([
        gyro.get('pitch_net_delta', 0.0), gyro.get('roll_net_delta', 0.0),
        gyro.get('xz_corr', 0.0), gyro.get('pitch_std', 0.0),
        gyro.get('pitch_range', 0.0), gyro.get('roll_std', 0.0),
    ])
    return np.concatenate([profile, base_extra, gyro_vec, orient_vec])  # 35-dim
