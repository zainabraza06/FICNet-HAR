import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, correlate, welch


RPE_NAMES = [
    'rpe_lag1', 'rpe_lag5', 'rpe_lag10', 'rpe_lag20',
    'rpe_rel_change1', 'rpe_rel_change5', 'rpe_rel_change10',
    'rpe_weighted_mean', 'rpe_weighted_std', 'rpe_pos_entropy',
]


# ── Binned tensor builder ─────────────────────────────────────────────────────

def build_binned_features(acc_data, gyro_data=None, n_bins=5, include_gyro=False):
    """Builds the (n_bins, per_bin_dim) tensor fed into BiLSTM / Fusion."""
    if len(acc_data) < n_bins:
        return None
    bin_edges = np.linspace(0, len(acc_data), n_bins + 1).astype(int)
    bins = []
    for i in range(n_bins):
        a = acc_data[bin_edges[i]:bin_edges[i + 1]]
        if len(a) == 0:
            a = acc_data[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
        ae = (a ** 2).sum(axis=0)
        ae = ae / (ae.sum() + 1e-8)
        am = np.sqrt((a ** 2).sum(axis=1))
        feat = list(ae) + [am.mean(), am.std()]
        if include_gyro and gyro_data is not None:
            g = gyro_data[bin_edges[i]:bin_edges[i + 1]]
            if len(g) == 0:
                g = gyro_data[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
            ge = (g ** 2).sum(axis=0)
            ge = ge / (ge.sum() + 1e-8)
            feat += list(ge)
        bins.append(feat)
    return np.array(bins, dtype=np.float32)


# ── Individual feature functions ──────────────────────────────────────────────

def feat_profile(acc, gyro=None, roll=None):
    """Per-axis energy profile across 5 time bins. (15-d)"""
    bin_edges = np.linspace(0, len(acc), 6).astype(int)
    profile = []
    for axis in range(3):
        e  = acc[:, axis] ** 2
        ap = np.array([e[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(5)])
        profile.append(ap / (ap.sum() + 1e-8))
    return np.concatenate(profile)


def feat_stats5(acc, gyro=None, roll=None):
    """Skew, kurtosis, variance, mean, max-jerk of acceleration magnitude. (5-d)"""
    acc_mag  = np.sqrt((acc ** 2).sum(axis=1))
    max_jerk = np.max(np.abs(np.gradient(acc_mag)))
    return np.array([skew(acc_mag), kurtosis(acc_mag),
                     acc_mag.var(), acc_mag.mean(), max_jerk])


def feat_fall_specific(acc, gyro=None, roll=None):
    """Fall-discriminative features: peak axis, jerk, timing, settle-time. (10-d)"""
    acc_mag  = np.sqrt((acc ** 2).sum(axis=1))
    max_jerk = np.max(np.abs(np.gradient(acc_mag)))
    peak_idx = int(np.argmax(acc_mag))
    peak_vec = acc[peak_idx] if peak_idx < len(acc) else np.zeros(3)
    denom    = np.abs(peak_vec).sum() + 1e-8
    pax      = np.abs(peak_vec) / denom
    time_to_peak = peak_idx / len(acc_mag)
    peaks, _     = find_peaks(acc_mag, height=acc_mag.mean() + acc_mag.std(), distance=5)
    n_secondary  = len(peaks)
    settled      = np.where(np.abs(acc_mag[peak_idx:] - 9.8) < 1.0)[0]
    settle_time  = (settled[0] / len(acc_mag)) if len(settled) > 0 else 1.0
    onset_len    = min(20, len(acc_mag))
    onset_window = acc_mag[:onset_len]
    onset_slope  = np.polyfit(np.arange(onset_len), onset_window, 1)[0] \
                   if len(onset_window) > 1 else 0.0
    return np.array([pax[0], pax[1], pax[2], n_secondary, max_jerk,
                     time_to_peak, settle_time,
                     skew(acc_mag), kurtosis(acc_mag), onset_slope])


def feat_attention(acc, gyro=None, roll=None, window_size=50):
    """Entropy, concentration, and uniformity of energy attention. (3-d)"""
    acc_mag    = np.sqrt((acc ** 2).sum(axis=1))
    if len(acc_mag) < window_size:
        return np.zeros(3)
    n_segments = max(1, len(acc_mag) // window_size)
    segments   = np.array_split(acc_mag, n_segments)
    se         = np.array([np.sum(s ** 2) for s in segments])
    se         = se / (se.sum() + 1e-8)
    entropy     = -np.sum(se * np.log(se + 1e-8))
    concentration = np.max(se)
    uniformity    = 1 - (np.max(se) - np.min(se))
    return np.array([entropy, concentration, uniformity])


def feat_rpe(acc, gyro=None, roll=None):
    """
    Relative Position Encoding features — lag-autocorrelations, relative change
    ratios (clipped to [0, 100]), weighted statistics, positional entropy. (10-d)

    The denominator floor (1e-3) prevents blow-up when the signal is very flat.
    """
    acc_mag  = np.sqrt((acc ** 2).sum(axis=1))
    n        = len(acc_mag)
    features = np.zeros(10, dtype=np.float32)
    if n < 20:
        return features
    for i, lag in enumerate([1, 5, 10, 20]):
        if lag < n:
            corr = np.corrcoef(acc_mag[:-lag], acc_mag[lag:])[0, 1]
            features[i] = corr if not np.isnan(corr) else 0.0
    diffs       = np.diff(acc_mag)
    DENOM_FLOOR = 1e-3
    for i, step in enumerate([1, 5, 10]):
        if len(diffs) > step:
            numerator   = np.mean(np.abs(diffs[step:]))
            denominator = max(np.mean(np.abs(diffs[:step])), DENOM_FLOOR)
            rel_change  = np.clip(numerator / denominator, 0.0, 100.0)
            features[4 + i] = rel_change if not np.isnan(rel_change) else 0.0
    weights       = np.arange(1, n + 1) / n
    weighted_mean = np.sum(weights * acc_mag) / (np.sum(weights) + 1e-8)
    weighted_std  = np.sqrt(
        np.sum(weights * (acc_mag - weighted_mean) ** 2) / (np.sum(weights) + 1e-8)
    )
    features[7], features[8] = weighted_mean, weighted_std
    threshold = np.percentile(acc_mag, 80)
    hi        = np.where(acc_mag > threshold)[0]
    if len(hi) > 0:
        features[9] = -np.sum((hi / n) * np.log(hi / n + 1e-8))
    return features


def feat_gyro(acc, gyro, roll):
    """Gyroscope energy fractions, statistics, autocorrelation, roll range. (8-d)"""
    acc_mag  = np.sqrt((acc  ** 2).sum(axis=1))
    gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))
    ge  = gyro ** 2
    tge = ge.sum() + 1e-8
    gx, gy, gz = ge[:, 0].sum() / tge, ge[:, 1].sum() / tge, ge[:, 2].sum() / tge
    ac    = acc_mag - acc_mag.mean()
    acorr = correlate(ac, ac, mode='full')[len(ac) - 1:]
    acorr = acorr / (acorr[0] + 1e-8)
    pk, _ = find_peaks(acorr[5:], height=0.2)
    autocorr   = acorr[pk[0] + 5] if len(pk) > 0 else 0.0
    roll_range = roll.max() - roll.min() if len(roll) > 0 else 0.0
    return np.array([gx, gy, gz,
                     skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean(),
                     roll_range, autocorr])


def feat_spectral(acc, gyro=None, roll=None, fs=87.0):
    """Dominant frequency, power concentration, and moment stats. (6-d)"""
    acc_mag = np.sqrt((acc ** 2).sum(axis=1))
    acmc    = acc_mag - acc_mag.mean()
    freqs, psd = welch(acmc, fs=fs, nperseg=min(256, len(acmc)))
    valid  = freqs > 0.3
    if valid.any():
        dom_freq    = freqs[valid][np.argmax(psd[valid])]
        power_conc  = psd[valid].max() / (psd[valid].sum() + 1e-8)
    else:
        dom_freq, power_conc = 0.0, 0.0
    return np.array([dom_freq, power_conc,
                     acc_mag.var(), acc_mag.mean(),
                     skew(acc_mag), kurtosis(acc_mag)])


# ── Single unified registry ───────────────────────────────────────────────────

REGISTRY = {
    'profile':       {'fn': feat_profile,       'core': True},
    'stats':         {'fn': feat_stats5,         'core': True},
    'fall_specific': {'fn': feat_fall_specific,  'core': False},
    'gyro':          {'fn': feat_gyro,           'core': False},
    'spectral':      {'fn': feat_spectral,       'core': False},
    'attention':     {'fn': feat_attention,      'core': False},
    'rpe':           {'fn': feat_rpe,            'core': False},
}

CORE_GROUPS      = [g for g, v in REGISTRY.items() if     v['core']]
CANDIDATE_GROUPS = [g for g, v in REGISTRY.items() if not v['core']]


# ── Flat feature builder ──────────────────────────────────────────────────────

def build_flat_features(acc, gyro, roll, groups):
    """Concatenates the outputs of every requested feature group. Returns None if
    the segment is too short."""
    if len(acc) < 5:
        return None
    return np.concatenate(
        [REGISTRY[g]['fn'](acc, gyro, roll) for g in groups]
    ).astype(np.float32)
