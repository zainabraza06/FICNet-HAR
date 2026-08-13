import os
import numpy as np
import pandas as pd

# The local relative path to the dataset
# Assuming it's run from the project root
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'MobiAct_Dataset_v2.0', 'Annotated Data')

def load_file(code, subj, trial=1):
    path = os.path.join(DATA_ROOT, code, f"{code}_{subj}_{trial}_annotated.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

def get_segment(code, subj, trial=1):
    df = load_file(code, subj, trial)
    if df is None: return None
    seg = df[df['label'] == code].reset_index(drop=True)
    return seg if len(seg) > 0 else None

# ============================================================
# STATIC DATASET BUILDERS (For Hierarchical Evaluation)
# ============================================================
def build_stage1_dataset(all_codes, fall_codes):
    from .features import build_binned_features, build_flat_features
    X_bins, X_flat, y, groups = [], [], [], []
    for code in all_codes:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None: continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x','acc_y','acc_z']].values
            gyro = seg[['gyro_x','gyro_y','gyro_z']].values
            pitch = seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg))
            roll = seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg))
            
            fb = build_binned_features(acc, gyro, include_gyro=False)
            fc = build_flat_features(acc, gyro, pitch, roll, 
                                     include_gyro=False, include_orient=False,
                                     include_spectral=False, include_attention=True,
                                     include_rpe=True, include_fpn=False, include_kalman=False,
                                     include_fall_specific=False)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append('FALL' if code in fall_codes else 'ADL')
                groups.append(subj)
    return np.array(X_bins), np.array(X_flat), np.array(y), np.array(groups)

def build_stage2a_dataset(fall_codes):
    from .features import build_binned_features, build_flat_features
    X_bins, X_flat, y, groups = [], [], [], []
    for code in fall_codes:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None: continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x','acc_y','acc_z']].values
            gyro = seg[['gyro_x','gyro_y','gyro_z']].values
            pitch = seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg))
            roll = seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg))
            
            fb = build_binned_features(acc, gyro, include_gyro=False)
            fc = build_flat_features(acc, gyro, pitch, roll,
                                     include_gyro=False, include_orient=False,
                                     include_spectral=False, include_attention=True,
                                     include_rpe=True, include_fpn=False, include_kalman=False,
                                     include_fall_specific=True)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc); y.append(code); groups.append(subj)
    return np.array(X_bins), np.array(X_flat), np.array(y), np.array(groups)

def build_stage2b_dataset(adl_codes):
    from .features import build_binned_features, build_flat_features
    X_bins, X_flat, y, groups = [], [], [], []
    for code in adl_codes:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None: continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x','acc_y','acc_z']].values
            gyro = seg[['gyro_x','gyro_y','gyro_z']].values
            pitch = seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg))
            roll = seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg))
            
            fb = build_binned_features(acc, gyro, include_gyro=True)
            fc = build_flat_features(acc, gyro, pitch, roll,
                                     include_gyro=True, include_orient=False,
                                     include_spectral=True, include_attention=True,
                                     include_rpe=True, include_fpn=False, include_kalman=False,
                                     include_fall_specific=False)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc); y.append(code); groups.append(subj)
    return np.array(X_bins), np.array(X_flat), np.array(y), np.array(groups)

# ============================================================
# STREAMING DATASET BUILDER
# ============================================================
def build_subject_stream(codes, subject, max_per_activity=1000,
                        include_attention=False, include_rpe=False,
                        include_fpn=False, include_kalman=False):
    acc_all, gyro_all, pitch_all, roll_all, labels_all = [], [], [], [], []
    for code in codes:
        seg = get_segment(code, subject)
        if seg is None: continue
        seg = seg.iloc[:max_per_activity]
        acc_all.append(seg[['acc_x','acc_y','acc_z']].values)
        gyro_all.append(seg[['gyro_x','gyro_y','gyro_z']].values)
        pitch_all.append(seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg)))
        roll_all.append(seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg)))
        labels_all.extend([code]*len(seg))
    if not acc_all: return None
    return (np.vstack(acc_all), np.vstack(gyro_all), np.concatenate(pitch_all),
            np.concatenate(roll_all), np.array(labels_all))

def window_stream(acc_s, gyro_s, pitch_s, roll_s, labels_s,
                 window_samples=200, stride=100, sub_bins=5,
                 include_attention=False, include_rpe=False,
                 include_fpn=False, include_kalman=False):
    # Import locally to avoid circular dependencies
    from .features import build_binned_features, build_flat_features
    
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
        
        fb = build_binned_features(acc_w, gyro_w, sub_bins, include_gyro=True)
        fc = build_flat_features(acc_w, gyro_w, pitch_w, roll_w, 
                                 n_bins=sub_bins,
                                 include_gyro=True,
                                 include_orient=False,
                                 include_spectral=True,
                                 include_attention=include_attention,
                                 include_rpe=include_rpe,
                                 include_fpn=include_fpn,
                                 include_kalman=include_kalman,
                                 include_fall_specific=False)
        if fb is not None and fc is not None:
            Xb.append(fb); Xf.append(fc); y.append(maj); purities.append(purity)
        start += stride
    if not Xb: return None, None, None, None
    return np.array(Xb), np.array(Xf), np.array(y), np.array(purities)

def build_streaming_dataset(codes, subjects, max_per_activity=1000,
                           include_attention=False, include_rpe=False,
                           include_fpn=False, include_kalman=False):
    data = {}
    for subj in subjects:
        stream = build_subject_stream(codes, subj, max_per_activity,
                                      include_attention, include_rpe,
                                      include_fpn, include_kalman)
        if stream is None: continue
        acc_s, gyro_s, pitch_s, roll_s, labels_s = stream
        Xb, Xf, yw, purity = window_stream(acc_s, gyro_s, pitch_s, roll_s, labels_s,
                                           include_attention=include_attention,
                                           include_rpe=include_rpe,
                                           include_fpn=include_fpn,
                                           include_kalman=include_kalman)
        if Xb is not None and len(Xb) > 5:
            data[subj] = (Xb, Xf, yw, purity)
    return data
