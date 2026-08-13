import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, correlate, welch

def build_binned_features(acc_data, gyro_data=None, n_bins=5, include_gyro=False):
    if len(acc_data) < n_bins:
        return None
    bin_edges = np.linspace(0, len(acc_data), n_bins+1).astype(int)
    bins = []
    for i in range(n_bins):
        a = acc_data[bin_edges[i]:bin_edges[i+1]]
        if len(a) == 0:
            a = acc_data[max(0, bin_edges[i]-1):bin_edges[i]+1]
        ae = (a**2).sum(axis=0)
        ae = ae / (ae.sum() + 1e-8)
        am = np.sqrt((a**2).sum(axis=1))
        feat = list(ae) + [am.mean(), am.std()]
        if include_gyro and gyro_data is not None:
            g = gyro_data[bin_edges[i]:bin_edges[i+1]]
            if len(g) == 0:
                g = gyro_data[max(0, bin_edges[i]-1):bin_edges[i]+1]
            ge = (g**2).sum(axis=0)
            ge = ge / (ge.sum() + 1e-8)
            feat += list(ge)
        bins.append(feat)
    return np.array(bins, dtype=np.float32)

def extract_attention_entropy(acc_data, window_size=50):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
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

def extract_multiscale_fpn(acc_data, scales=[1, 2, 4]):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    features = []
    for scale in scales:
        downsampled = acc_mag[::scale] if scale > 1 else acc_mag
        features.extend([np.mean(downsampled), np.std(downsampled), np.max(downsampled),
                         np.min(downsampled), skew(downsampled), kurtosis(downsampled),
                         np.max(np.abs(np.gradient(downsampled)))])
    for i in range(len(scales)-1):
        ratio = np.mean(acc_mag[::scales[i]]) / (np.mean(acc_mag[::scales[i+1]]) + 1e-8)
        features.append(ratio)
    return np.array(features[:10])

def extract_kalman_fusion(acc_data, gyro_data):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
    gyro_mag = np.sqrt((gyro_data**2).sum(axis=1))
    acc_noise = np.std(acc_mag)
    gyro_noise = np.std(gyro_mag)
    kalman_gain = acc_noise / (acc_noise + gyro_noise + 1e-8)
    fused_mag = kalman_gain * acc_mag + (1 - kalman_gain) * gyro_mag
    features = [np.mean(fused_mag), np.std(fused_mag), np.max(fused_mag),
                np.min(fused_mag), np.median(fused_mag),
                np.max(np.abs(np.gradient(fused_mag))), kalman_gain,
                1 - kalman_gain, acc_noise / (gyro_noise + 1e-8),
                np.corrcoef(acc_mag, gyro_mag)[0,1] if len(acc_mag) > 10 else 0.0]
    return np.array(features[:10])

def extract_relative_position(acc_data):
    acc_mag = np.sqrt((acc_data**2).sum(axis=1))
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

def build_flat_features(acc_data, gyro_data=None, pitch_data=None, roll_data=None, 
                        n_bins=5, include_gyro=False, include_orient=False,
                        include_attention=False, include_rpe=False, 
                        include_fpn=False, include_kalman=False,
                        include_spectral=False, include_fall_specific=False):
    if len(acc_data) < n_bins:
        return None
    
    bin_edges = np.linspace(0, len(acc_data), n_bins+1).astype(int)
    
    # Profile features (15-dim)
    profile = []
    for axis in range(3):
        e = acc_data[:, axis]**2
        ap = np.array([e[bin_edges[i]:bin_edges[i+1]].sum() for i in range(n_bins)])
        profile.append(ap / (ap.sum() + 1e-8))
    profile = np.concatenate(profile)
    
    acc_mag_full = np.sqrt((acc_data**2).sum(axis=1))
    a_skew, a_kurt, a_var, a_mean = skew(acc_mag_full), kurtosis(acc_mag_full), acc_mag_full.var(), acc_mag_full.mean()
    max_jerk = np.max(np.abs(np.gradient(acc_mag_full)))
    
    parts = [profile]
    
    # Base Accelerometer stats (varies slightly by stage)
    if include_spectral:
        # Stage 2b uses 4-dim base stats
        base = np.array([a_var, a_mean, a_skew, a_kurt])
        parts.append(base)
        
        # Spectral features (6-dim)
        fs = 87.0
        acmc = acc_mag_full - a_mean
        freqs, psd = welch(acmc, fs=fs, nperseg=min(256, len(acmc)))
        valid = freqs > 0.3
        if valid.any():
            dom_freq = freqs[valid][np.argmax(psd[valid])]
            power_conc = psd[valid].max() / (psd[valid].sum() + 1e-8)
        else:
            dom_freq, power_conc = 0.0, 0.0
        spectral = np.array([dom_freq, power_conc, a_var, a_mean, a_skew, a_kurt])
        parts.append(spectral)
    else:
        # Stage 1 / 2a uses 5-dim base stats
        acc_stats = [a_skew, a_kurt, a_var, a_mean, max_jerk]
        parts.append(acc_stats)
    
    # Fall-specific features (10-dim) for Stage 2a
    if include_fall_specific:
        peak_idx = np.argmax(acc_mag_full)
        peak_vec = acc_data[peak_idx] if peak_idx < len(acc_data) else np.zeros(3)
        denom = np.abs(peak_vec).sum() + 1e-8
        pax = np.abs(peak_vec) / denom
        time_to_peak = peak_idx / len(acc_mag_full)
        
        peaks, _ = find_peaks(acc_mag_full, height=a_mean + np.std(acc_mag_full), distance=5)
        n_secondary_peaks = len(peaks)
        
        settled = np.where(np.abs(acc_mag_full[peak_idx:] - 9.8) < 1.0)[0]
        settle_time = (settled[0] / len(acc_mag_full)) if len(settled) > 0 else 1.0
        
        onset_len = min(20, len(acc_mag_full))
        onset_window = acc_mag_full[:onset_len]
        onset_slope = np.polyfit(np.arange(onset_len), onset_window, 1)[0] if len(onset_window) > 1 else 0.0
        
        fall_specific = [pax[0], pax[1], pax[2], n_secondary_peaks, max_jerk, 
                         time_to_peak, settle_time, a_skew, a_kurt, onset_slope]
        parts.append(fall_specific)
    
    # Gyroscope features (8-dim)
    if include_gyro and gyro_data is not None:
        gyro_mag = np.sqrt((gyro_data**2).sum(axis=1))
        ge = gyro_data**2
        tge = ge.sum() + 1e-8
        gx, gy, gz = ge[:,0].sum()/tge, ge[:,1].sum()/tge, ge[:,2].sum()/tge
        g_skew, g_kurt, g_mean = skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean()
        ac = acc_mag_full - a_mean
        acorr = correlate(ac, ac, mode='full')[len(ac)-1:]
        acorr = acorr / (acorr[0] + 1e-8)
        pk, _ = find_peaks(acorr[5:], height=0.2)
        autocorr = acorr[pk[0]+5] if len(pk) > 0 else 0.0
        roll_range = roll_data.max() - roll_data.min() if roll_data is not None and len(roll_data) > 0 else 0.0
        gyro_vec = np.array([gx, gy, gz, g_skew, g_kurt, g_mean, roll_range, autocorr])
        parts.append(gyro_vec)
    
    # Orientation features (6-dim)
    if include_orient and pitch_data is not None and roll_data is not None:
        pitch_delta = pitch_data[-1] - pitch_data[0] if len(pitch_data) > 0 else 0.0
        roll_delta = roll_data[-1] - roll_data[0] if len(roll_data) > 0 else 0.0
        xz_corr = np.corrcoef(acc_data[:,0], acc_data[:,2])[0,1] if len(acc_data) > 10 else 0.0
        xz_corr = xz_corr if not np.isnan(xz_corr) else 0.0
        pitch_std = pitch_data.std() if len(pitch_data) > 0 else 0.0
        pitch_range = pitch_data.max() - pitch_data.min() if len(pitch_data) > 0 else 0.0
        roll_std = roll_data.std() if len(roll_data) > 0 else 0.0
        orient = np.array([pitch_delta, roll_delta, xz_corr, pitch_std, pitch_range, roll_std])
        parts.append(orient)
    
    # Cross-domain features
    if include_attention:
        parts.append(extract_attention_entropy(acc_data))
    if include_rpe:
        parts.append(extract_relative_position(acc_data))
    if include_fpn:
        parts.append(extract_multiscale_fpn(acc_data))
    if include_kalman and gyro_data is not None:
        parts.append(extract_kalman_fusion(acc_data, gyro_data))
    
    return np.concatenate([p for p in parts if len(p) > 0]).astype(np.float32)
