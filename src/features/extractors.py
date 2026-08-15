import numpy as np
from scipy.stats import skew, kurtosis

def build_binned_features(acc_data, n_bins=5):
    if len(acc_data) < n_bins: return None
    bin_edges = np.linspace(0, len(acc_data), n_bins+1).astype(int)
    bins = []
    for i in range(n_bins):
        a = acc_data[bin_edges[i]:bin_edges[i+1]]
        if len(a) == 0: a = acc_data[max(0, bin_edges[i]-1):bin_edges[i]+1]
        ae = (a**2).sum(axis=0); ae = ae / (ae.sum() + 1e-8)
        am = np.sqrt((a**2).sum(axis=1))
        bins.append(list(ae) + [am.mean(), am.std()])
    return np.array(bins, dtype=np.float32)

def feat_profile(acc_data):
    bin_edges = np.linspace(0, len(acc_data), 6).astype(int)
    profile = []
    for axis in range(3):
        e = acc_data[:, axis]**2
        ap = np.array([e[bin_edges[i]:bin_edges[i+1]].sum() for i in range(5)])
        profile.append(ap / (ap.sum() + 1e-8))
    return np.concatenate(profile)  # 15

def feat_stats(acc_data):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    a_skew, a_kurt, a_var, a_mean = skew(acc_mag), kurtosis(acc_mag), acc_mag.var(), acc_mag.mean()
    max_jerk = np.max(np.abs(np.gradient(acc_mag)))
    return np.array([a_skew, a_kurt, a_var, a_mean, max_jerk])  # 5

def feat_attention(acc_data, window_size=50):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    if len(acc_mag) < window_size: return np.zeros(3)
    n_segments = max(1, len(acc_mag) // window_size)
    segments = np.array_split(acc_mag, n_segments)
    se = np.array([np.sum(s**2) for s in segments]); se = se / (se.sum() + 1e-8)
    entropy = -np.sum(se * np.log(se + 1e-8))
    concentration = np.max(se)
    uniformity = 1 - (np.max(se) - np.min(se))
    return np.array([entropy, concentration, uniformity])  # 3

def feat_rpe(acc_data):
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

def build_flat_features(acc_data, groups):
    if len(acc_data) < 5: return None
    parts = [FEATURE_REGISTRY[g]['fn'](acc_data) for g in groups]
    return np.concatenate(parts).astype(np.float32)

def feature_names_for(groups):
    names = []
    for g in groups:
        names.extend(FEATURE_REGISTRY[g]['names'])
    return names
