import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, correlate, welch
from src.data.loader import get_segment

def build_streaming_binned_features(acc_w, gyro_w, n_bins=5):
    bin_edges = np.linspace(0, len(acc_w), n_bins+1).astype(int)
    bins = []
    for i in range(n_bins):
        a = acc_w[bin_edges[i]:bin_edges[i+1]]
        if len(a) == 0:
            a = acc_w[max(0, bin_edges[i]-1):bin_edges[i]+1]
        ae = (a**2).sum(axis=0)
        ae = ae / (ae.sum() + 1e-8)
        am = np.sqrt((a**2).sum(axis=1))
        feat = list(ae) + [am.mean(), am.std()]
        g = gyro_w[bin_edges[i]:bin_edges[i+1]]
        if len(g) == 0:
            g = gyro_w[max(0, bin_edges[i]-1):bin_edges[i]+1]
        ge = (g**2).sum(axis=0)
        ge = ge / (ge.sum() + 1e-8)
        feat += list(ge)
        bins.append(feat)
    return np.array(bins, dtype=np.float32)

def extract_attention_entropy(acc_w, window_size=50):
    acc_mag = np.sqrt((acc_w**2).sum(axis=1))
    if len(acc_mag) < window_size:
        return np.array([0.0, 0.0, 0.0])
    n_segments = max(1, len(acc_mag) // window_size)
    segments = np.array_split(acc_mag, n_segments)
    segment_energies = np.array([np.sum(s**2) for s in segments])
    segment_energies = segment_energies / (segment_energies.sum() + 1e-8)
    attention_entropy = -np.sum(segment_energies * np.log(segment_energies + 1e-8))
    attention_concentration = np.max(segment_energies)
    attention_uniformity = 1 - (np.max(segment_energies) - np.min(segment_energies))
    return np.array([attention_entropy, attention_concentration, attention_uniformity])

def extract_relative_position(acc_w):
    acc_mag = np.sqrt((acc_w**2).sum(axis=1))
    n = len(acc_mag)
    if n < 20:
        return np.zeros(10)
    features = []
    for lag in [1, 5, 10, 20]:
        if lag < n:
            corr = np.corrcoef(acc_mag[:-lag], acc_mag[lag:])[0,1]
            features.append(corr if not np.isnan(corr) else 0.0)
    diffs = np.diff(acc_mag)
    for step in [1, 5, 10]:
        if len(diffs) > step:
            rel_change = np.mean(np.abs(diffs[step:])) / (np.mean(np.abs(diffs[:step])) + 1e-8)
            features.append(rel_change if not np.isnan(rel_change) else 0.0)
    weights = np.arange(1, n+1) / n
    weighted_mean = np.sum(weights * acc_mag) / (np.sum(weights) + 1e-8)
    weighted_std = np.sqrt(np.sum(weights * (acc_mag - weighted_mean)**2) / (np.sum(weights) + 1e-8))
    features.extend([weighted_mean, weighted_std])
    threshold = np.percentile(acc_mag, 80)
    high_activity_positions = np.where(acc_mag > threshold)[0]
    if len(high_activity_positions) > 0:
        pos_entropy = -np.sum((high_activity_positions / n) * np.log(high_activity_positions / n + 1e-8))
        features.append(pos_entropy)
    else:
        features.append(0.0)
    return np.array(features[:10])

def build_streaming_flat_features(acc_w, gyro_w, pitch_w, roll_w, n_bins=5,
                                  include_attention=False, include_rpe=False):
    bin_edges = np.linspace(0, len(acc_w), n_bins+1).astype(int)

    profile = []
    for axis in range(3):
        e = acc_w[:, axis]**2
        ap = np.array([e[bin_edges[i]:bin_edges[i+1]].sum() for i in range(n_bins)])
        profile.append(ap/(ap.sum()+1e-8))
    profile = np.concatenate(profile)

    acc_mag = np.sqrt((acc_w**2).sum(axis=1))
    base = np.array([acc_mag.var(), acc_mag.mean(), skew(acc_mag), kurtosis(acc_mag)])

    fs = 87.0
    acmc = acc_mag - acc_mag.mean()
    freqs, psd = welch(acmc, fs=fs, nperseg=min(256, len(acmc)))
    valid = freqs > 0.3
    if valid.any():
        dom_freq = freqs[valid][np.argmax(psd[valid])]
        power_conc = psd[valid].max() / (psd[valid].sum() + 1e-8)
    else:
        dom_freq, power_conc = 0.0, 0.0
    spectral = np.array([dom_freq, power_conc, acc_mag.var(), acc_mag.mean(), skew(acc_mag), kurtosis(acc_mag)])

    gyro_mag = np.sqrt((gyro_w**2).sum(axis=1))
    ge = gyro_w**2
    tge = ge.sum() + 1e-8
    gx, gy, gz = ge[:,0].sum()/tge, ge[:,1].sum()/tge, ge[:,2].sum()/tge
    g_skew, g_kurt, g_mean = skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean()
    ac = acc_mag - acc_mag.mean()
    acorr = correlate(ac, ac, mode='full')[len(ac)-1:]
    acorr = acorr / (acorr[0] + 1e-8)
    pk, _ = find_peaks(acorr[5:], height=0.2)
    autocorr = acorr[pk[0]+5] if len(pk) > 0 else 0.0
    roll_range = roll_w.max() - roll_w.min() if len(roll_w) > 0 else 0.0
    gyro_vec = np.array([gx, gy, gz, g_skew, g_kurt, g_mean, roll_range, autocorr])

    parts = [profile, base, spectral, gyro_vec]

    if include_attention:
        parts.append(extract_attention_entropy(acc_w))
    if include_rpe:
        parts.append(extract_relative_position(acc_w))

    return np.concatenate(parts).astype(np.float32)

def build_subject_stream(codes, subject, max_per_activity=1000):
    acc_all, gyro_all, pitch_all, roll_all, labels_all = [], [], [], [], []
    for code in codes:
        seg = get_segment(code, subject)
        if seg is None: continue
        seg = seg.iloc[:max_per_activity]
        acc_all.append(seg[['acc_x','acc_y','acc_z']].values)
        if 'gyro_x' in seg.columns:
            gyro_all.append(seg[['gyro_x','gyro_y','gyro_z']].values)
        else:
            gyro_all.append(np.zeros((len(seg), 3)))
        pitch_all.append(seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg)))
        roll_all.append(seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg)))
        labels_all.extend([code]*len(seg))
    if not acc_all: return None
    return (np.vstack(acc_all), np.vstack(gyro_all), np.concatenate(pitch_all),
            np.concatenate(roll_all), np.array(labels_all))

def window_stream(acc_s, gyro_s, pitch_s, roll_s, labels_s,
                 window_samples=200, stride=100, sub_bins=5,
                 include_attention=False, include_rpe=False):
    n = len(acc_s)
    Xb, Xf, y, purities = [], [], [], []
    start = 0
    while start + window_samples <= n:
        end = start + window_samples
        acc_w, gyro_w, pitch_w, roll_w = acc_s[start:end], gyro_s[start:end], pitch_s[start:end], roll_s[start:end]
        labels_w = labels_s[start:end]
        vals, counts = np.unique(labels_w, return_counts=True)
        maj = vals[np.argmax(counts)]
        purity = counts.max() / len(labels_w)

        fb = build_streaming_binned_features(acc_w, gyro_w, sub_bins)
        fc = build_streaming_flat_features(acc_w, gyro_w, pitch_w, roll_w, sub_bins,
                                          include_attention=include_attention,
                                          include_rpe=include_rpe)
        if fb is not None and fc is not None:
            Xb.append(fb); Xf.append(fc); y.append(maj); purities.append(purity)
        start += stride
    if not Xb: return None, None, None, None
    return np.array(Xb), np.array(Xf), np.array(y), np.array(purities)

def build_streaming_dataset(codes, subjects, max_per_activity=1000,
                           include_attention=False, include_rpe=False):
    data = {}
    for subj in subjects:
        stream = build_subject_stream(codes, subj, max_per_activity)
        if stream is None: continue
        acc_s, gyro_s, pitch_s, roll_s, labels_s = stream
        Xb, Xf, yw, purity = window_stream(acc_s, gyro_s, pitch_s, roll_s, labels_s,
                                           include_attention=include_attention,
                                           include_rpe=include_rpe)
        if Xb is not None and len(Xb) > 5:
            data[subj] = (Xb, Xf, yw, purity)
    return data
