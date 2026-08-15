import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, correlate, welch

def build_binned_features(acc_data, gyro_data=None, n_bins=5, include_gyro=False):
    if len(acc_data) < n_bins: return None
    bin_edges = np.linspace(0, len(acc_data), n_bins+1).astype(int)
    bins = []
    for i in range(n_bins):
        a = acc_data[bin_edges[i]:bin_edges[i+1]]
        if len(a) == 0: a = acc_data[max(0, bin_edges[i]-1):bin_edges[i]+1]
        ae = (a**2).sum(axis=0); ae = ae / (ae.sum() + 1e-8)
        am = np.sqrt((a**2).sum(axis=1))
        feat = list(ae) + [am.mean(), am.std()]
        if include_gyro and gyro_data is not None:
            g = gyro_data[bin_edges[i]:bin_edges[i+1]]
            if len(g) == 0: g = gyro_data[max(0, bin_edges[i]-1):bin_edges[i]+1]
            ge = (g**2).sum(axis=0); ge = ge / (ge.sum() + 1e-8)
            feat += list(ge)
        bins.append(feat)
    return np.array(bins, dtype=np.float32)

def feat_profile(acc_data, gyro_data=None, roll_data=None):
    bin_edges = np.linspace(0, len(acc_data), 6).astype(int)
    profile = []
    for axis in range(3):
        e = acc_data[:, axis]**2
        ap = np.array([e[bin_edges[i]:bin_edges[i+1]].sum() for i in range(5)])
        profile.append(ap / (ap.sum() + 1e-8))
    return np.concatenate(profile)  # 15

def feat_stats(acc_data, gyro_data=None, roll_data=None):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    a_skew, a_kurt, a_var, a_mean = skew(acc_mag), kurtosis(acc_mag), acc_mag.var(), acc_mag.mean()
    max_jerk = np.max(np.abs(np.gradient(acc_mag)))
    return np.array([a_skew, a_kurt, a_var, a_mean, max_jerk])  # 5

def feat_stats_s2b(acc_data, gyro_data=None, roll_data=None):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    return np.array([acc_mag.var(), acc_mag.mean(), skew(acc_mag), kurtosis(acc_mag)])  # 4

def feat_attention(acc_data, gyro_data=None, roll_data=None, window_size=50):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    if len(acc_mag) < window_size: return np.zeros(3)
    n_segments = max(1, len(acc_mag) // window_size)
    segments = np.array_split(acc_mag, n_segments)
    se = np.array([np.sum(s**2) for s in segments]); se = se / (se.sum() + 1e-8)
    entropy = -np.sum(se * np.log(se + 1e-8))
    concentration = np.max(se)
    uniformity = 1 - (np.max(se) - np.min(se))
    return np.array([entropy, concentration, uniformity])  # 3

def feat_fall_specific(acc_data, gyro_data=None, roll_data=None):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    a_skew, a_kurt = skew(acc_mag), kurtosis(acc_mag)
    max_jerk = np.max(np.abs(np.gradient(acc_mag)))
    peak_idx = int(np.argmax(acc_mag))
    peak_vec = acc_data[peak_idx] if peak_idx < len(acc_data) else np.zeros(3)
    denom = np.abs(peak_vec).sum() + 1e-8
    pax = np.abs(peak_vec) / denom
    time_to_peak = peak_idx / len(acc_mag)
    peaks, _ = find_peaks(acc_mag, height=acc_mag.mean()+acc_mag.std(), distance=5)
    n_secondary_peaks = len(peaks)
    settled = np.where(np.abs(acc_mag[peak_idx:] - 9.8) < 1.0)[0]
    settle_time = (settled[0]/len(acc_mag)) if len(settled) > 0 else 1.0
    onset_len = min(20, len(acc_mag)); onset_window = acc_mag[:onset_len]
    onset_slope = np.polyfit(np.arange(onset_len), onset_window, 1)[0] if len(onset_window) > 1 else 0.0
    return np.array([pax[0], pax[1], pax[2], n_secondary_peaks, max_jerk,
                      time_to_peak, settle_time, a_skew, a_kurt, onset_slope])  # 10

def feat_spectral(acc_data, gyro_data=None, roll_data=None, fs=87.0):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    acmc = acc_mag - acc_mag.mean()
    freqs, psd = welch(acmc, fs=fs, nperseg=min(256, len(acmc)))
    valid = freqs > 0.3
    if valid.any():
        dom_freq = freqs[valid][np.argmax(psd[valid])]
        power_conc = psd[valid].max() / (psd[valid].sum() + 1e-8)
    else:
        dom_freq, power_conc = 0.0, 0.0
    return np.array([dom_freq, power_conc, acc_mag.var(), acc_mag.mean(),
                      skew(acc_mag), kurtosis(acc_mag)])  # 6

def feat_gyro(acc_data, gyro_data, roll_data):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    if gyro_data is None:
        return np.zeros(8)
    gyro_mag = np.sqrt((gyro_data**2).sum(axis=1))
    ge = gyro_data**2
    tge = ge.sum() + 1e-8
    gx, gy, gz = ge[:,0].sum()/tge, ge[:,1].sum()/tge, ge[:,2].sum()/tge
    g_skew, g_kurt, g_mean = skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean()
    ac = acc_mag - acc_mag.mean()
    acorr = correlate(ac, ac, mode='full')[len(ac)-1:]
    acorr = acorr / (acorr[0] + 1e-8)
    pk, _ = find_peaks(acorr[5:], height=0.2)
    autocorr = acorr[pk[0]+5] if len(pk) > 0 else 0.0
    roll_range = roll_data.max() - roll_data.min() if len(roll_data) > 0 else 0.0
    return np.array([gx, gy, gz, g_skew, g_kurt, g_mean, roll_range, autocorr])  # 8

def feat_rpe(acc_data, gyro_data=None, roll_data=None):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    n = len(acc_mag)
    features = np.zeros(10, dtype=np.float32)
    if n < 20: return features
    for i, lag in enumerate([1, 5, 10, 20]):
        if lag < n:
            corr = np.corrcoef(acc_mag[:-lag], acc_mag[lag:])[0, 1]
            features[i] = corr if not np.isnan(corr) else 0.0
    diffs = np.diff(acc_mag)
    for i, step in enumerate([1, 5, 10]):
        if len(diffs) > step:
            rel_change = np.mean(np.abs(diffs[step:])) / (np.mean(np.abs(diffs[:step])) + 1e-8)
            features[4+i] = rel_change if not np.isnan(rel_change) else 0.0
    weights = np.arange(1, n+1) / n
    weighted_mean = np.sum(weights * acc_mag) / (np.sum(weights) + 1e-8)
    weighted_std = np.sqrt(np.sum(weights * (acc_mag - weighted_mean)**2) / (np.sum(weights) + 1e-8))
    features[7], features[8] = weighted_mean, weighted_std
    threshold = np.percentile(acc_mag, 80)
    hi = np.where(acc_mag > threshold)[0]
    if len(hi) > 0:
        features[9] = -np.sum((hi/n) * np.log(hi/n + 1e-8))
    return features  # 10

FEATURE_REGISTRY = {
    'profile':   {'fn': feat_profile,   'dim': 15, 'core': True,
                  'names': [f'energy_{ax}_bin{i}' for ax in 'xyz' for i in range(5)]},
    'stats':     {'fn': feat_stats,     'dim': 5,  'core': True,
                  'names': ['acc_skew', 'acc_kurtosis', 'acc_var', 'acc_mean', 'max_jerk']},
    'attention': {'fn': feat_attention, 'dim': 3,  'core': False,
                  'names': ['attn_entropy', 'attn_concentration', 'attn_uniformity']},
    'rpe':       {'fn': feat_rpe,       'dim': 10, 'core': False,
                  'names': ['rpe_lag1','rpe_lag5','rpe_lag10','rpe_lag20',
                            'rpe_rel_change1','rpe_rel_change5','rpe_rel_change10',
                            'rpe_weighted_mean','rpe_weighted_std','rpe_pos_entropy']},
}

CORE_GROUPS = [g for g, v in FEATURE_REGISTRY.items() if v['core']]
CANDIDATE_GROUPS = [g for g, v in FEATURE_REGISTRY.items() if not v['core']]

FEATURE_REGISTRY_S2A = {
    'profile':       {'fn': feat_profile,       'dim': 15, 'core': True,
                       'names': [f'energy_{ax}_bin{i}' for ax in 'xyz' for i in range(5)]},
    'stats':         {'fn': feat_stats,         'dim': 5,  'core': True,
                       'names': ['acc_skew', 'acc_kurtosis', 'acc_var', 'acc_mean', 'max_jerk']},
    'fall_specific': {'fn': feat_fall_specific, 'dim': 10, 'core': True,
                       'names': ['peak_ax_x','peak_ax_y','peak_ax_z','n_secondary_peaks','max_jerk2',
                                 'time_to_peak','settle_time','skew2','kurt2','onset_slope']},
    'attention':     {'fn': feat_attention,     'dim': 3,  'core': False,
                       'names': ['attn_entropy', 'attn_concentration', 'attn_uniformity']},
    'rpe':           {'fn': feat_rpe,           'dim': 10, 'core': False,
                       'names': ['rpe_lag1','rpe_lag5','rpe_lag10','rpe_lag20',
                                 'rpe_rel_change1','rpe_rel_change5','rpe_rel_change10',
                                 'rpe_weighted_mean','rpe_weighted_std','rpe_pos_entropy']},
}

CORE_GROUPS_S2A = [g for g, v in FEATURE_REGISTRY_S2A.items() if v['core']]
CANDIDATE_GROUPS_S2A = [g for g, v in FEATURE_REGISTRY_S2A.items() if not v['core']]

FEATURE_REGISTRY_S2B = {
    'profile':   {'fn': feat_profile,   'dim': 15, 'core': True,
                  'names': [f'energy_{ax}_bin{i}' for ax in 'xyz' for i in range(5)]},
    'stats':     {'fn': feat_stats_s2b, 'dim': 4,  'core': True,
                  'names': ['acc_var', 'acc_mean', 'acc_skew', 'acc_kurt']},
    'spectral':  {'fn': feat_spectral,  'dim': 6,  'core': False,
                  'names': ['dom_freq', 'power_conc', 'spec_var', 'spec_mean', 'spec_skew', 'spec_kurt']},
    'gyro':      {'fn': feat_gyro,      'dim': 8,  'core': False,
                  'names': ['gyro_x_energy', 'gyro_y_energy', 'gyro_z_energy',
                            'gyro_skew', 'gyro_kurt', 'gyro_mean', 'roll_range', 'autocorr']},
    'attention': {'fn': feat_attention, 'dim': 3,  'core': False,
                  'names': ['attn_entropy', 'attn_concentration', 'attn_uniformity']},
    'rpe':       {'fn': feat_rpe,       'dim': 10, 'core': False,
                  'names': ['rpe_lag1','rpe_lag5','rpe_lag10','rpe_lag20',
                            'rpe_rel_change1','rpe_rel_change5','rpe_rel_change10',
                            'rpe_weighted_mean','rpe_weighted_std','rpe_pos_entropy']},
}

CORE_GROUPS_S2B = [g for g, v in FEATURE_REGISTRY_S2B.items() if v['core']]
CANDIDATE_GROUPS_S2B = [g for g, v in FEATURE_REGISTRY_S2B.items() if not v['core']]

def build_flat_features(acc_data, gyro_data=None, roll_data=None, groups=[], stage='s1'):
    if len(acc_data) < 5: return None
    if stage == 's1':
        registry = FEATURE_REGISTRY
    elif stage == 's2a':
        registry = FEATURE_REGISTRY_S2A
    else:
        registry = FEATURE_REGISTRY_S2B
        
    parts = [registry[g]['fn'](acc_data, gyro_data, roll_data) for g in groups]
    return np.concatenate(parts).astype(np.float32)

def feature_names_for(groups, stage='s1'):
    names = []
    if stage == 's1':
        registry = FEATURE_REGISTRY
    elif stage == 's2a':
        registry = FEATURE_REGISTRY_S2A
    else:
        registry = FEATURE_REGISTRY_S2B
        
    for g in groups:
        names.extend(registry[g]['names'])
    return names
