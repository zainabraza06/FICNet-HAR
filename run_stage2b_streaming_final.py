"""
STAGE 2b — STREAMING FINAL STANDALONE SCRIPT
(with purity stratification + complete metric suite)

Naive classical (no memory) baseline  — RandomForest on per-window flat features
Streaming BiLSTM  — inner-bin LSTM + outer-window LSTM
Streaming Fusion  — same as above + MLP branch with cross-attention

Window / stride:  200 samples / 100 samples (~2.3 s / ~1.1 s at 87 Hz)
Purity threshold: 0.90 (windows whose dominant-label fraction >= 0.90)

Full metric suite reported everywhere:
  overall:         acc, balanced_acc, macro_precision, macro_recall, macro_f1
  pure windows:    same 5 + pure_window_fraction
  boundary windows:same 5 + boundary_window_fraction
  per-class F1:    one column per ADL code (saved to CSV / JSON)
  confusion matrix: saved per model

Single LOSO run each (documented compute-cost scope decision).
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, correlate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_ROOT = 'MobiAct_Dataset_v2.0/Annotated Data'
assert os.path.exists(DATA_ROOT), f"DATA_ROOT missing at {DATA_ROOT}"

PROJECT_ROOT = '.'
CKPT_DIR = f'{PROJECT_ROOT}/models/checkpoints/stage2b_streaming'
RESULTS_DIR = f'{PROJECT_ROOT}/results/stage2b'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']

WINDOW_SAMPLES = 200
STRIDE = 100
SUB_BINS = 5
PURITY_THRESHOLD = 0.90


# ============================================================
# DATA LOADING
# ============================================================

def load_file(code, subj, trial=1):
    path = os.path.join(DATA_ROOT, code, f"{code}_{subj}_{trial}_annotated.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def get_segment(code, subj, trial=1):
    df = load_file(code, subj, trial)
    if df is None:
        return None
    seg = df[df['label'] == code].reset_index(drop=True)
    return seg if len(seg) > 0 else None


def build_subject_stream(codes, subject, max_per_activity=1000):
    acc_all, gyro_all, pitch_all, roll_all, labels_all = [], [], [], [], []
    for code in codes:
        seg = get_segment(code, subject)
        if seg is None:
            continue
        seg = seg.iloc[:max_per_activity]
        acc_all.append(seg[['acc_x', 'acc_y', 'acc_z']].values)
        gyro_all.append(seg[['gyro_x', 'gyro_y', 'gyro_z']].values)
        pitch_all.append(seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg)))
        roll_all.append(seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg)))
        labels_all.extend([code] * len(seg))
    if not acc_all:
        return None
    return (
        np.vstack(acc_all), np.vstack(gyro_all),
        np.concatenate(pitch_all), np.concatenate(roll_all),
        np.array(labels_all),
    )


# ============================================================
# PER-WINDOW FEATURE BUILDERS
# ============================================================

def window_binned(acc_w, gyro_w, n_bins=5):
    """
    Returns shape (n_bins, 8): per-bin [acc_x_frac, acc_y_frac, acc_z_frac,
    acc_mag_mean, acc_mag_std, gyro_x_frac, gyro_y_frac, gyro_z_frac].
    """
    bin_edges = np.linspace(0, len(acc_w), n_bins + 1).astype(int)
    bins = []
    for i in range(n_bins):
        a = acc_w[bin_edges[i]:bin_edges[i + 1]]
        if len(a) == 0:
            a = acc_w[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
        ae = (a ** 2).sum(axis=0)
        ae = ae / (ae.sum() + 1e-8)
        am = np.sqrt((a ** 2).sum(axis=1))
        feat = list(ae) + [am.mean(), am.std()]
        g = gyro_w[bin_edges[i]:bin_edges[i + 1]]
        if len(g) == 0:
            g = gyro_w[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
        ge = (g ** 2).sum(axis=0)
        ge = ge / (ge.sum() + 1e-8)
        feat += list(ge)
        bins.append(feat)
    return np.array(bins, dtype=np.float32)


def window_classical(acc_w, gyro_w, pitch_w, roll_w, n_bins=5):
    """
    29-dim flat feature matching Stage 2b locked final set:
      15 (per-axis profile) + 4 (acc stats) + 8 (gyro block) + 2 (roll/autocorr)
    = 29-dim flat.
    """
    bin_edges = np.linspace(0, len(acc_w), n_bins + 1).astype(int)
    # 15-dim profile
    profile = []
    for axis in range(3):
        e = acc_w[:, axis] ** 2
        ap = np.array([e[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(n_bins)])
        profile.append(ap / (ap.sum() + 1e-8))
    profile = np.concatenate(profile)

    # 4-dim acc stats
    acc_mag = np.sqrt((acc_w ** 2).sum(axis=1))
    base = np.array([acc_mag.var(), acc_mag.mean(), skew(acc_mag), kurtosis(acc_mag)])

    # 8-dim gyro block (matches Stage 2b gyro_vec)
    gyro_mag = np.sqrt((gyro_w ** 2).sum(axis=1))
    ge = gyro_w ** 2
    tge = ge.sum() + 1e-8
    gx, gy, gz = ge[:, 0].sum() / tge, ge[:, 1].sum() / tge, ge[:, 2].sum() / tge
    g_skew, g_kurt, g_mean = skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean()
    roll_range = roll_w.max() - roll_w.min() if len(roll_w) > 0 else 0.0
    ac = acc_mag - acc_mag.mean()
    acorr = correlate(ac, ac, mode='full')[len(ac) - 1:]
    acorr = acorr / (acorr[0] + 1e-8)
    pk, _ = find_peaks(acorr[5:], height=0.2)
    autocorr_strength = acorr[pk[0] + 5] if len(pk) > 0 else 0.0
    gyro_vec = np.array([gx, gy, gz, g_skew, g_kurt, g_mean, roll_range, autocorr_strength])

    return np.concatenate([profile, base, gyro_vec]).astype(np.float32)


def window_stream(acc_s, gyro_s, pitch_s, roll_s, labels_s,
                  window_samples=WINDOW_SAMPLES, stride=STRIDE, sub_bins=SUB_BINS):
    n = len(acc_s)
    Xb, Xf, y, purities = [], [], [], []
    start = 0
    while start + window_samples <= n:
        end = start + window_samples
        acc_w = acc_s[start:end]
        gyro_w = gyro_s[start:end]
        pitch_w = pitch_s[start:end]
        roll_w = roll_s[start:end]
        labels_w = labels_s[start:end]
        vals, counts = np.unique(labels_w, return_counts=True)
        maj = vals[np.argmax(counts)]
        purity = counts.max() / len(labels_w)
        fb = window_binned(acc_w, gyro_w, sub_bins)
        fc = window_classical(acc_w, gyro_w, pitch_w, roll_w, sub_bins)
        if fb is not None and fc is not None:
            Xb.append(fb)
            Xf.append(fc)
            y.append(maj)
            purities.append(purity)
        start += stride
    if not Xb:
        return None, None, None, None
    return np.array(Xb), np.array(Xf), np.array(y), np.array(purities)


def build_streaming_dataset(codes, subjects, max_per_activity=1000):
    data = {}
    for subj in subjects:
        stream = build_subject_stream(codes, subj, max_per_activity)
        if stream is None:
            continue
        acc_s, gyro_s, pitch_s, roll_s, labels_s = stream
        Xb, Xf, yw, purity = window_stream(acc_s, gyro_s, pitch_s, roll_s, labels_s)
        if Xb is not None and len(Xb) > 5:
            data[subj] = (Xb, Xf, yw, purity)
    return data


# ============================================================
# FULL METRIC SUITE — STREAMING VERSION
# Covers overall + purity-stratified subsets,
# each with: acc, bal_acc, macro_precision, macro_recall, macro_f1
# plus per-class F1 and confusion matrix.
# ============================================================

def _subset_metrics(y_true_sub, y_pred_sub, label_list):
    """Compute the full 5-metric suite for a subset of predictions."""
    if len(y_true_sub) == 0:
        return {
            'accuracy': None, 'balanced_accuracy': None,
            'macro_precision': None, 'macro_recall': None, 'macro_f1': None,
        }
    acc = accuracy_score(y_true_sub, y_pred_sub)
    bal = balanced_accuracy_score(y_true_sub, y_pred_sub)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_sub, y_pred_sub,
        labels=list(range(len(label_list))),
        average='macro',
        zero_division=0,
    )
    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bal),
        'macro_precision': float(prec),
        'macro_recall': float(rec),
        'macro_f1': float(f1),
    }


def _per_class_f1(y_true, y_pred, label_list):
    """Per-class F1 keyed by activity code."""
    _, _, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(len(label_list))),
        average=None,
        zero_division=0,
    )
    return {label_list[i]: float(f1_per[i]) for i in range(len(label_list))}


def compute_streaming_metrics(y_true, y_pred, purity, label_list, thresh=PURITY_THRESHOLD):
    """
    Returns a dict with:
      overall_*          — full suite over all windows
      pure_window_*      — full suite over high-purity windows (purity >= thresh)
      boundary_window_*  — full suite over boundary windows (purity < thresh)
      pure_window_fraction / boundary_window_fraction
      per_class_f1       — per-class F1 over all windows
      confusion_matrix   — integer matrix (all windows)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    purity = np.array(purity)

    pm = purity >= thresh   # pure-window mask
    bm = ~pm                # boundary-window mask

    overall = _subset_metrics(y_true, y_pred, label_list)
    pure = _subset_metrics(y_true[pm], y_pred[pm], label_list) if pm.any() else _subset_metrics([], [], label_list)
    boundary = _subset_metrics(y_true[bm], y_pred[bm], label_list) if bm.any() else _subset_metrics([], [], label_list)

    per_cls_f1 = _per_class_f1(y_true, y_pred, label_list)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_list))))

    return {
        # --- overall ---
        'overall_accuracy': overall['accuracy'],
        'overall_balanced_accuracy': overall['balanced_accuracy'],
        'overall_macro_precision': overall['macro_precision'],
        'overall_macro_recall': overall['macro_recall'],
        'overall_macro_f1': overall['macro_f1'],
        # --- pure windows ---
        'pure_window_fraction': float(pm.mean()),
        'pure_window_accuracy': pure['accuracy'],
        'pure_window_balanced_accuracy': pure['balanced_accuracy'],
        'pure_window_macro_precision': pure['macro_precision'],
        'pure_window_macro_recall': pure['macro_recall'],
        'pure_window_macro_f1': pure['macro_f1'],
        # --- boundary windows ---
        'boundary_window_fraction': float(bm.mean()),
        'boundary_window_accuracy': boundary['accuracy'],
        'boundary_window_balanced_accuracy': boundary['balanced_accuracy'],
        'boundary_window_macro_precision': boundary['macro_precision'],
        'boundary_window_macro_recall': boundary['macro_recall'],
        'boundary_window_macro_f1': boundary['macro_f1'],
        # --- per-class + confusion ---
        'per_class_f1': per_cls_f1,
        'confusion_matrix': cm.tolist(),
    }


def print_streaming_metrics(name, r):
    """Pretty-print the full metric suite for one model."""
    print(f"\n  {'Model':<20s}: {name}")
    print(f"  {'Overall':<22s}  acc={r['overall_accuracy']:.4f}  "
          f"bal={r['overall_balanced_accuracy']:.4f}  "
          f"f1={r['overall_macro_f1']:.4f}  "
          f"prec={r['overall_macro_precision']:.4f}  "
          f"rec={r['overall_macro_recall']:.4f}")
    pf = r['pure_window_fraction']
    if r['pure_window_accuracy'] is not None:
        print(f"  {'Pure windows':<22s}  acc={r['pure_window_accuracy']:.4f}  "
              f"bal={r['pure_window_balanced_accuracy']:.4f}  "
              f"f1={r['pure_window_macro_f1']:.4f}  "
              f"frac={pf:.1%}")
    if r['boundary_window_accuracy'] is not None:
        print(f"  {'Boundary windows':<22s}  acc={r['boundary_window_accuracy']:.4f}  "
              f"bal={r['boundary_window_balanced_accuracy']:.4f}  "
              f"f1={r['boundary_window_macro_f1']:.4f}  "
              f"frac={r['boundary_window_fraction']:.1%}")
    f1s = r['per_class_f1']
    f1_str = '  '.join(f"{k}={v:.3f}" for k, v in f1s.items())
    print(f"  Per-class F1: {f1_str}")


# ============================================================
# MODELS
# ============================================================

class StreamingBiLSTM(nn.Module):
    """
    Two-level LSTM:
      inner  — reads sub-bins of a single window  →  window embedding
      outer  — reads the window-embedding sequence →  per-window logits
    """
    def __init__(self, per_bin_dim, n_classes, inner_hidden=16, outer_hidden=16):
        super().__init__()
        self.inner_lstm = nn.LSTM(per_bin_dim, inner_hidden, batch_first=True, bidirectional=True)
        self.outer_lstm = nn.LSTM(inner_hidden * 2, outer_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(outer_hidden * 2, n_classes)

    def forward(self, x_bins):
        # x_bins: (batch=1, n_windows, sub_bins, per_bin_dim)
        b, nw, sb, pd = x_bins.shape
        x_flat = x_bins.view(b * nw, sb, pd)
        _, (h_n, _) = self.inner_lstm(x_flat)
        ws = torch.cat([h_n[-2], h_n[-1]], dim=1).view(b, nw, -1)
        out, _ = self.outer_lstm(ws)
        out = self.dropout(out)
        return self.classifier(out).squeeze(0)   # (n_windows, n_classes)


class StreamingFusionNet(nn.Module):
    """
    Two-level BiLSTM (sequence) + MLP (flat stats) with cross-attention fusion,
    operating on a full subject stream as a sequence of windows.
    """
    def __init__(self, per_bin_dim, flat_dim, n_classes,
                 inner_hidden=16, outer_hidden=16, mlp_hidden=32):
        super().__init__()
        inner_out = inner_hidden * 2
        outer_out = outer_hidden * 2
        self.inner_lstm = nn.LSTM(per_bin_dim, inner_hidden, batch_first=True, bidirectional=True)
        self.outer_lstm = nn.LSTM(inner_out, outer_hidden, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, outer_out),
        )
        # Cross-attention projections
        self.q_proj_dl = nn.Linear(outer_out, outer_out)
        self.k_proj_ml = nn.Linear(outer_out, outer_out)
        self.v_proj_ml = nn.Linear(outer_out, outer_out)
        self.q_proj_ml = nn.Linear(outer_out, outer_out)
        self.k_proj_dl = nn.Linear(outer_out, outer_out)
        self.v_proj_dl = nn.Linear(outer_out, outer_out)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(outer_out * 2, n_classes)

    def forward(self, x_bins, x_flat):
        # x_bins: (1, n_windows, sub_bins, per_bin_dim)
        # x_flat: (1, n_windows, flat_dim)
        b, nw, sb, pd = x_bins.shape
        xf_seq = x_bins.view(b * nw, sb, pd)
        _, (h_n, _) = self.inner_lstm(xf_seq)
        ws = torch.cat([h_n[-2], h_n[-1]], dim=1).view(b, nw, -1)
        dl_out, _ = self.outer_lstm(ws)
        dl_out = self.dropout(dl_out)                             # (b, nw, outer_out)
        ml_out = self.mlp(x_flat.view(b * nw, -1)).view(b, nw, -1)
        ml_out = self.dropout(ml_out)                             # (b, nw, outer_out)
        # DL attends to ML
        q_dl = self.q_proj_dl(dl_out)
        k_ml = self.k_proj_ml(ml_out)
        v_ml = self.v_proj_ml(ml_out)
        attn_dl = torch.softmax(
            (q_dl * k_ml).sum(-1, keepdim=True) / (q_dl.size(-1) ** 0.5), dim=-1
        )
        dl_att = attn_dl * v_ml + dl_out
        # ML attends to DL
        q_ml = self.q_proj_ml(ml_out)
        k_dl = self.k_proj_dl(dl_out)
        v_dl = self.v_proj_dl(dl_out)
        attn_ml = torch.softmax(
            (q_ml * k_dl).sum(-1, keepdim=True) / (q_ml.size(-1) ** 0.5), dim=-1
        )
        ml_att = attn_ml * v_dl + ml_out
        return self.classifier(torch.cat([dl_att, ml_att], dim=-1)).squeeze(0)  # (nw, n_classes)


# ============================================================
# TRAINING HELPERS
# ============================================================

def train_streaming_bilstm(train_data, l2i, n_classes, per_bin_dim, epochs=50):
    model = StreamingBiLSTM(per_bin_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-2)
    model.train()
    for _ in range(epochs):
        for subj, (Xb, Xf, yw, p) in train_data.items():
            y_idx = np.array([l2i[l] for l in yw])
            Xb_t = torch.tensor(Xb, dtype=torch.float32).unsqueeze(0).to(device)
            y_t = torch.tensor(y_idx, dtype=torch.long).to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(Xb_t), y_t)
            loss.backward()
            opt.step()
    return model


def train_streaming_fusion(train_data, l2i, n_classes, per_bin_dim, flat_dim, epochs=50):
    model = StreamingFusionNet(per_bin_dim, flat_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-2)
    model.train()
    for _ in range(epochs):
        for subj, (Xb, Xf, yw, p) in train_data.items():
            y_idx = np.array([l2i[l] for l in yw])
            Xb_t = torch.tensor(Xb, dtype=torch.float32).unsqueeze(0).to(device)
            Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(device)
            y_t = torch.tensor(y_idx, dtype=torch.long).to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(Xb_t, Xf_t), y_t)
            loss.backward()
            opt.step()
    return model


# ============================================================
# LOSO RUNNERS
# ============================================================

def run_naive_classical(data, l2i, label_list):
    """
    Per-window RandomForest with no temporal memory.
    Baseline representing the non-streaming upper-bound for classical models.
    """
    subj_ids = list(data.keys())
    n_classes = len(label_list)
    all_true, all_pred, all_purity = [], [], []
    for fold_i, test_subj in enumerate(subj_ids):
        Xf_tr, y_tr = [], []
        for s in subj_ids:
            if s == test_subj:
                continue
            _, Xf, yw, _ = data[s]
            Xf_tr.append(Xf)
            y_tr.extend([l2i[l] for l in yw])
        Xf_tr = np.vstack(Xf_tr)
        y_tr = np.array(y_tr)
        _, Xf_te, yw_te, p_te = data[test_subj]
        y_te = np.array([l2i[l] for l in yw_te])
        sc = StandardScaler().fit(Xf_tr)
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0)
        clf.fit(sc.transform(Xf_tr), y_tr)
        pred = clf.predict(sc.transform(Xf_te))
        all_true.extend(y_te)
        all_pred.extend(pred)
        all_purity.extend(p_te)
        if fold_i % 10 == 0:
            print(f"    fold {fold_i}/{len(subj_ids)}")
    return compute_streaming_metrics(all_true, all_pred, all_purity, label_list)


def run_streaming_loso(data, l2i, label_list, per_bin_dim, flat_dim,
                        model_type, epochs, save_ckpt=None):
    """
    LOSO evaluation for streaming DL models.
    Trains once per subject-left-out fold, feeding the entire subject
    stream as a single sequence (batch=1, n_windows, ...).
    """
    subj_ids = list(data.keys())
    n_classes = len(label_list)
    all_true, all_pred, all_purity = [], [], []
    best_state, best_acc = None, -1

    for fold_i, test_subj in enumerate(subj_ids):
        train_data = {s: data[s] for s in subj_ids if s != test_subj}

        if model_type == 'bilstm':
            model = train_streaming_bilstm(train_data, l2i, n_classes, per_bin_dim, epochs)
        else:
            model = train_streaming_fusion(train_data, l2i, n_classes, per_bin_dim, flat_dim, epochs)

        model.eval()
        with torch.no_grad():
            Xb, Xf, yw, p = data[test_subj]
            y_idx = np.array([l2i[l] for l in yw])
            Xb_t = torch.tensor(Xb, dtype=torch.float32).unsqueeze(0).to(device)
            if model_type == 'fusion':
                Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(Xb_t, Xf_t)
            else:
                logits = model(Xb_t)
            pred = logits.argmax(dim=1).cpu().numpy()

        fa = accuracy_score(y_idx, pred)
        if fa > best_acc:
            best_acc = fa
            best_state = model.state_dict()

        all_true.extend(y_idx)
        all_pred.extend(pred)
        all_purity.extend(p)

        if fold_i % 10 == 0:
            print(f"    fold {fold_i}/{len(subj_ids)}")

    if save_ckpt and best_state is not None:
        torch.save(best_state, save_ckpt)

    return compute_streaming_metrics(all_true, all_pred, all_purity, label_list)


# ============================================================
# SUMMARY TABLE BUILDER
# ============================================================

def build_summary_df(results, label_list):
    """
    Flattens results dict into a wide DataFrame.
    One row per model; per-class F1 columns appended at the end.
    Confusion matrices are excluded (saved separately to JSON).
    """
    rows = []
    for model_name, r in results.items():
        row = {'model': model_name}
        for k, v in r.items():
            if k in ('per_class_f1', 'confusion_matrix'):
                continue
            row[k] = v
        for code in label_list:
            row[f'f1_{code}'] = r['per_class_f1'].get(code, None)
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

print("Building streaming dataset...")
data = build_streaming_dataset(ADL_CODES_11, list(range(1, 68)))
label_list = ADL_CODES_11
l2i = {l: i for i, l in enumerate(label_list)}
n_classes = len(label_list)

sample_subj = next(iter(data))
per_bin_dim = data[sample_subj][0].shape[2]  # Xb: (n_windows, sub_bins, per_bin_dim)
flat_dim = data[sample_subj][1].shape[1]      # Xf: (n_windows, flat_dim)
print(f"Subjects with data: {len(data)}")
print(f"per_bin_dim={per_bin_dim}  flat_dim={flat_dim}")
print(f"Window={WINDOW_SAMPLES} samples  stride={STRIDE} samples  purity_thresh={PURITY_THRESHOLD}")

results = {}

# ── (1) Naive Classical ──────────────────────────────────────
print("\n" + "=" * 60)
print("(1) NAIVE CLASSICAL — RandomForest, no temporal memory")
print("=" * 60)
results['NaiveClassical'] = run_naive_classical(data, l2i, label_list)
print_streaming_metrics('NaiveClassical', results['NaiveClassical'])

# ── (2) Streaming BiLSTM ────────────────────────────────────
print("\n" + "=" * 60)
print("(2) STREAMING BiLSTM")
print("=" * 60)
results['StreamingBiLSTM'] = run_streaming_loso(
    data, l2i, label_list, per_bin_dim, flat_dim,
    model_type='bilstm', epochs=50,
    save_ckpt=f'{CKPT_DIR}/streaming_bilstm.pt',
)
print_streaming_metrics('StreamingBiLSTM', results['StreamingBiLSTM'])

# ── (3) Streaming Fusion ─────────────────────────────────────
print("\n" + "=" * 60)
print("(3) STREAMING FUSION — BiLSTM + MLP + cross-attention")
print("=" * 60)
results['StreamingFusion'] = run_streaming_loso(
    data, l2i, label_list, per_bin_dim, flat_dim,
    model_type='fusion', epochs=50,
    save_ckpt=f'{CKPT_DIR}/streaming_fusion.pt',
)
print_streaming_metrics('StreamingFusion', results['StreamingFusion'])

# ── Save ─────────────────────────────────────────────────────
summary_df = build_summary_df(results, label_list)
summary_df.to_csv(f'{RESULTS_DIR}/stage2b_streaming_FINAL.csv', index=False)

with open(f'{RESULTS_DIR}/stage2b_streaming_FINAL.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*60}")
print("STAGE 2b STREAMING COMPLETE")
print(f"  CSV  → {RESULTS_DIR}/stage2b_streaming_FINAL.csv")
print(f"  JSON → {RESULTS_DIR}/stage2b_streaming_FINAL.json")
print(f"  Checkpoints → {CKPT_DIR}/")
print(f"{'='*60}")
