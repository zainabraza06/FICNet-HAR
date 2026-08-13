"""
STAGE 2b — FINAL COMPLETE STANDALONE SCRIPT
ADL Classification (11-class)

Locked FINAL feature set: profile + spectral + gyro, NO orientation
Evidence: orientation features HURT (0.9199 without vs 0.9168 with)
Models: LDA, KNN-3, SVM-RBF, RandomForest, BiLSTM, Fusion
Ablation runs on the ACTUAL best model, bins+flat toggled together.
Full metric suite everywhere. Targeted significance tests only.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, correlate, welch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
CKPT_DIR_2B = f'{PROJECT_ROOT}/models/checkpoints/stage2b'
RESULTS_DIR_2B = f'{PROJECT_ROOT}/results/stage2b'
os.makedirs(CKPT_DIR_2B, exist_ok=True)
os.makedirs(RESULTS_DIR_2B, exist_ok=True)

ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']

SEEDS = [0, 1, 2, 3, 4]
DETERMINISTIC_MODELS = ['LDA', 'KNN-3']
STOCHASTIC_CLASSICAL = ['SVM-RBF', 'RandomForest']
MODEL_LIST = DETERMINISTIC_MODELS + STOCHASTIC_CLASSICAL + ['BiLSTM', 'Fusion']


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


# ============================================================
# PRINT / METRIC HELPERS
# ============================================================

def print_m_seeds(name, seed_metrics):
    accs = [m['accuracy'] for m in seed_metrics]
    bals = [m['balanced_accuracy'] for m in seed_metrics]
    f1s = [m['macro_f1'] for m in seed_metrics]
    precs = [m['macro_precision'] for m in seed_metrics]
    recs = [m['macro_recall'] for m in seed_metrics]
    n = len(seed_metrics)
    print(
        f"  {name:<42s} (n={n}): acc={np.mean(accs):.4f}"
        + (f"±{np.std(accs):.4f}" if n > 1 else "")
        + f" bal={np.mean(bals):.4f}" + (f"±{np.std(bals):.4f}" if n > 1 else "")
        + f" f1={np.mean(f1s):.4f}" + (f"±{np.std(f1s):.4f}" if n > 1 else "")
        + f" prec={np.mean(precs):.4f} rec={np.mean(recs):.4f}"
    )


def summarize_full(results_dict):
    rows = []
    for name, seed_metrics in results_dict.items():
        accs = [m['accuracy'] for m in seed_metrics]
        bals = [m['balanced_accuracy'] for m in seed_metrics]
        f1s = [m['macro_f1'] for m in seed_metrics]
        precs = [m['macro_precision'] for m in seed_metrics]
        recs = [m['macro_recall'] for m in seed_metrics]
        n = len(seed_metrics)
        rows.append({
            'name': name, 'n_runs': n,
            'acc_mean': np.mean(accs), 'acc_std': np.std(accs) if n > 1 else None,
            'bal_mean': np.mean(bals), 'bal_std': np.std(bals) if n > 1 else None,
            'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s) if n > 1 else None,
            'precision_mean': np.mean(precs), 'recall_mean': np.mean(recs),
        })
    return pd.DataFrame(rows)


def paired_tests(name_a, accs_a, name_b, accs_b):
    accs_a, accs_b = np.array(accs_a), np.array(accs_b)
    t_stat, t_p = stats.ttest_rel(accs_a, accs_b)
    try:
        w_stat, w_p = stats.wilcoxon(accs_a, accs_b)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    print(
        f"\n  {name_a} vs {name_b}: t_p={t_p:.4f} {'** SIG **' if t_p < 0.05 else '(n.s.)'}, "
        f"wilcoxon_p={w_p if not np.isnan(w_p) else 'degenerate'}"
    )
    return {'t_p': float(t_p), 'w_p': float(w_p) if not np.isnan(w_p) else None}


def full_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        'accuracy': acc, 'balanced_accuracy': bal_acc,
        'macro_precision': prec, 'macro_recall': rec, 'macro_f1': f1,
        'confusion_matrix': cm.tolist(),
    }


# ============================================================
# FEATURE BUILDERS — flags control which blocks are included;
# bins+flat ALWAYS built together for a given flag combo
# ============================================================

def build_binned_2b(seg, n_bins=5, max_samples=1000, include_gyro=True):
    seg = seg.iloc[:max_samples]
    if len(seg) < n_bins:
        return None
    acc = seg[['acc_x', 'acc_y', 'acc_z']].values
    bin_edges = np.linspace(0, len(acc), n_bins + 1).astype(int)
    if include_gyro:
        gyro_vals = seg[['gyro_x', 'gyro_y', 'gyro_z']].values
    bins = []
    for i in range(n_bins):
        a = acc[bin_edges[i]:bin_edges[i + 1]]
        if len(a) == 0:
            a = acc[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
        ae = (a ** 2).sum(axis=0)
        ae = ae / (ae.sum() + 1e-8)
        am = np.sqrt((a ** 2).sum(axis=1))
        feat = list(ae) + [am.mean(), am.std()]
        if include_gyro:
            g = gyro_vals[bin_edges[i]:bin_edges[i + 1]]
            if len(g) == 0:
                g = gyro_vals[max(0, bin_edges[i] - 1):bin_edges[i] + 1]
            ge = (g ** 2).sum(axis=0)
            ge = ge / (ge.sum() + 1e-8)
            feat += list(ge)
        bins.append(feat)
    return np.array(bins, dtype=np.float32)


def build_flat_2b(seg, n_bins=5, max_samples=1000,
                   include_spectral=True, include_gyro=True, include_orient=True):
    seg_full = seg
    seg_c = seg.iloc[:max_samples]
    acc = seg_c[['acc_x', 'acc_y', 'acc_z']].values
    if len(acc) < n_bins:
        return None
    bin_edges = np.linspace(0, len(acc), n_bins + 1).astype(int)

    # 15-dim per-axis energy profile
    profile = []
    for axis in range(3):
        e = acc[:, axis] ** 2
        ap = np.array([e[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(n_bins)])
        profile.append(ap / (ap.sum() + 1e-8))
    profile = np.concatenate(profile)
    parts = [profile]

    # 6-dim spectral + acc stat block
    if include_spectral:
        acc_mag_full = np.sqrt(
            seg_full['acc_x'] ** 2 + seg_full['acc_y'] ** 2 + seg_full['acc_z'] ** 2
        )
        fs = 87.0
        acmc = acc_mag_full.values - acc_mag_full.values.mean()
        freqs, psd = welch(acmc, fs=fs, nperseg=min(256, len(acmc)))
        valid = freqs > 0.3
        if valid.any():
            dom_freq = freqs[valid][np.argmax(psd[valid])]
            power_conc = psd[valid].max() / (psd[valid].sum() + 1e-8)
        else:
            dom_freq, power_conc = 0.0, 0.0
        base = np.array([
            dom_freq, power_conc, acc_mag_full.var(), acc_mag_full.mean(),
            skew(acc_mag_full), kurtosis(acc_mag_full),
        ])  # 6
        parts.append(base)

    # 8-dim gyro block
    if include_gyro:
        acc_mag = np.sqrt((acc ** 2).sum(axis=1))
        gyro = seg_c[['gyro_x', 'gyro_y', 'gyro_z']].values
        gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))
        ge = gyro ** 2
        tge = ge.sum() + 1e-8
        gx, gy, gz = ge[:, 0].sum() / tge, ge[:, 1].sum() / tge, ge[:, 2].sum() / tge
        g_skew, g_kurt, g_mean = skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean()
        roll_range = seg_c['roll'].max() - seg_c['roll'].min() if 'roll' in seg_c.columns else 0.0
        ac = acc_mag - acc_mag.mean()
        acorr = correlate(ac, ac, mode='full')[len(ac) - 1:]
        acorr = acorr / (acorr[0] + 1e-8)
        pk, _ = find_peaks(acorr[5:], height=0.2)
        autocorr_strength = acorr[pk[0] + 5] if len(pk) > 0 else 0.0
        gyro_vec = np.array([gx, gy, gz, g_skew, g_kurt, g_mean, roll_range, autocorr_strength])  # 8
        parts.append(gyro_vec)

    # 6-dim orientation block (excluded in locked FINAL set)
    if include_orient:
        pitch_delta = (seg_c['pitch'].values[-1] - seg_c['pitch'].values[0]) if 'pitch' in seg_c.columns else 0.0
        roll_delta = (seg_c['roll'].values[-1] - seg_c['roll'].values[0]) if 'roll' in seg_c.columns else 0.0
        xz_corr = np.corrcoef(acc[:, 0], acc[:, 2])[0, 1] if len(acc) > 10 else 0.0
        xz_corr = xz_corr if not np.isnan(xz_corr) else 0.0
        pitch_std = seg_c['pitch'].std() if 'pitch' in seg_c.columns else 0.0
        pitch_range = seg_c['pitch'].max() - seg_c['pitch'].min() if 'pitch' in seg_c.columns else 0.0
        roll_std = seg_c['roll'].std() if 'roll' in seg_c.columns else 0.0
        orient = np.array([pitch_delta, roll_delta, xz_corr, pitch_std, pitch_range, roll_std])  # 6
        parts.append(orient)

    return np.concatenate(parts).astype(np.float32)


def build_dataset_2b(codes, subjects, include_spectral, include_gyro, include_orient, n_bins=5):
    Xb, Xf, y, g = [], [], [], []
    for code in codes:
        for subj in subjects:
            seg = get_segment(code, subj)
            if seg is None:
                continue
            fb = build_binned_2b(seg, n_bins, include_gyro=include_gyro)
            fc = build_flat_2b(
                seg, n_bins,
                include_spectral=include_spectral,
                include_gyro=include_gyro,
                include_orient=include_orient,
            )
            if fb is not None and fc is not None:
                Xb.append(fb); Xf.append(fc); y.append(code); g.append(subj)
    return np.array(Xb), np.array(Xf), np.array(y), np.array(g)


# ============================================================
# MODELS
# ============================================================

def get_classical_models(seed):
    return {
        'LDA': LinearDiscriminantAnalysis(priors=np.ones(11) / 11),
        'KNN-3': KNeighborsClassifier(n_neighbors=3, weights='distance'),
        'SVM-RBF': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=seed),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed),
    }


class SimpleBiLSTM(nn.Module):
    def __init__(self, per_bin_dim, n_classes, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(per_bin_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(torch.cat([h_n[-2], h_n[-1]], dim=1))
        return self.classifier(h)


class DualBranchFusionNet(nn.Module):
    def __init__(self, per_bin_dim, flat_dim, n_classes, lstm_hidden=16, mlp_hidden=32):
        super().__init__()
        out_dim = lstm_hidden * 2
        self.lstm = nn.LSTM(per_bin_dim, lstm_hidden, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, mlp_hidden), nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(), nn.Dropout(0.3), nn.Linear(mlp_hidden, out_dim)
        )
        self.q_proj_dl = nn.Linear(out_dim, out_dim)
        self.k_proj_ml = nn.Linear(out_dim, out_dim)
        self.v_proj_ml = nn.Linear(out_dim, out_dim)
        self.q_proj_ml = nn.Linear(out_dim, out_dim)
        self.k_proj_dl = nn.Linear(out_dim, out_dim)
        self.v_proj_dl = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(out_dim * 2, n_classes)

    def forward(self, x_bins, x_flat):
        _, (h_n, _) = self.lstm(x_bins)
        dl_repr = self.dropout(torch.cat([h_n[-2], h_n[-1]], dim=1))
        ml_repr = self.dropout(self.mlp(x_flat))
        q_dl = self.q_proj_dl(dl_repr); k_ml = self.k_proj_ml(ml_repr); v_ml = self.v_proj_ml(ml_repr)
        attn_dl = torch.softmax((q_dl * k_ml).sum(1, keepdim=True) / (q_dl.size(1) ** 0.5), dim=1)
        dl_att = attn_dl * v_ml + dl_repr
        q_ml = self.q_proj_ml(ml_repr); k_dl = self.k_proj_dl(dl_repr); v_dl = self.v_proj_dl(dl_repr)
        attn_ml = torch.softmax((q_ml * k_dl).sum(1, keepdim=True) / (q_ml.size(1) ** 0.5), dim=1)
        ml_att = attn_ml * v_dl + ml_repr
        return self.classifier(torch.cat([dl_att, ml_att], dim=1))


def train_bilstm(X_train, y_train_idx, n_classes, per_bin_dim, epochs, hidden, seed):
    torch.manual_seed(seed)
    model = SimpleBiLSTM(per_bin_dim, n_classes, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-2)
    Xt = torch.tensor(X_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train_idx, dtype=torch.long).to(device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt), yt)
        loss.backward()
        opt.step()
    return model


def train_fusion(Xb_train, Xf_train, y_train_idx, n_classes, per_bin_dim, flat_dim, epochs, seed):
    torch.manual_seed(seed)
    model = DualBranchFusionNet(per_bin_dim, flat_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-2)
    Xb_t = torch.tensor(Xb_train, dtype=torch.float32).to(device)
    Xf_t = torch.tensor(Xf_train, dtype=torch.float32).to(device)
    yt = torch.tensor(y_train_idx, dtype=torch.long).to(device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xb_t, Xf_t), yt)
        loss.backward()
        opt.step()
    return model


# ============================================================
# LOSO RUNNERS
# ============================================================

def run_loso_classical(X, y, groups, model_builder, labels):
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr, te in logo.split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = model_builder()
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        all_true.extend(y[te])
        all_pred.extend(pred)
    return full_metrics(all_true, all_pred, labels)


def run_loso_dl(X, y, groups, epochs, hidden, seed, labels, save_ckpt=None):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(labels)
    per_bin_dim = X.shape[2]
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    best_state, best_acc = None, -1
    for tr, te in logo.split(X, y_idx, groups):
        if len(set(y_idx[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr].reshape(-1, per_bin_dim))
        Xtr = sc.transform(X[tr].reshape(-1, per_bin_dim)).reshape(X[tr].shape)
        Xte = sc.transform(X[te].reshape(-1, per_bin_dim)).reshape(X[te].shape)
        model = train_bilstm(Xtr, y_idx[tr], n_classes, per_bin_dim, epochs, hidden, seed)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
        fa = accuracy_score(y_idx[te], pred)
        if fa > best_acc:
            best_acc = fa
            best_state = model.state_dict()
        all_true.extend(y_idx[te])
        all_pred.extend(pred)
    if save_ckpt and best_state is not None:
        torch.save(best_state, save_ckpt)
    return full_metrics([labels[i] for i in all_true], [labels[i] for i in all_pred], labels)


def run_loso_fusion(X_bins, X_flat, y, groups, epochs, seed, labels, save_ckpt=None):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(labels)
    per_bin_dim = X_bins.shape[2]
    flat_dim = X_flat.shape[1]
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    best_state, best_acc = None, -1
    for tr, te in logo.split(X_bins, y_idx, groups):
        if len(set(y_idx[tr])) < 2:
            continue
        scb = StandardScaler().fit(X_bins[tr].reshape(-1, per_bin_dim))
        scf = StandardScaler().fit(X_flat[tr])
        Xb_tr = scb.transform(X_bins[tr].reshape(-1, per_bin_dim)).reshape(X_bins[tr].shape)
        Xb_te = scb.transform(X_bins[te].reshape(-1, per_bin_dim)).reshape(X_bins[te].shape)
        Xf_tr = scf.transform(X_flat[tr])
        Xf_te = scf.transform(X_flat[te])
        model = train_fusion(Xb_tr, Xf_tr, y_idx[tr], n_classes, per_bin_dim, flat_dim, epochs, seed)
        model.eval()
        with torch.no_grad():
            pred = model(
                torch.tensor(Xb_te, dtype=torch.float32).to(device),
                torch.tensor(Xf_te, dtype=torch.float32).to(device),
            ).argmax(1).cpu().numpy()
        fa = accuracy_score(y_idx[te], pred)
        if fa > best_acc:
            best_acc = fa
            best_state = model.state_dict()
        all_true.extend(y_idx[te])
        all_pred.extend(pred)
    if save_ckpt and best_state is not None:
        torch.save(best_state, save_ckpt)
    return full_metrics([labels[i] for i in all_true], [labels[i] for i in all_pred], labels)


def run_named_model(model_name, Xb, Xf, y, groups, labels, seed, dl_epochs, dl_hidden, save_ckpt=None):
    if model_name in DETERMINISTIC_MODELS + STOCHASTIC_CLASSICAL:
        return run_loso_classical(
            Xf, y, groups,
            lambda n=model_name, s=seed: get_classical_models(s)[n],
            labels,
        )
    elif model_name == 'BiLSTM':
        return run_loso_dl(Xb, y, groups, dl_epochs, dl_hidden, seed, labels, save_ckpt)
    elif model_name == 'Fusion':
        return run_loso_fusion(Xb, Xf, y, groups, dl_epochs, seed, labels, save_ckpt)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def seeds_for(model_name):
    return [0] if model_name in DETERMINISTIC_MODELS else SEEDS


# ============================================================
# BUILD DATASETS
# FINAL (locked) = profile + spectral + gyro, NO orientation
# ============================================================

print("Building Stage 2b datasets...")
print("  FINAL (profile+spectral+gyro, NO orientation)...")
Xb_2b_final, Xf_2b_final, y_2b, g_2b = build_dataset_2b(
    ADL_CODES_11, list(range(1, 68)),
    include_spectral=True, include_gyro=True, include_orient=False,
)
print("  Comparison: profile+spectral (acc-only, no gyro)...")
Xb_2b_accOnly, Xf_2b_accOnly, y_2b_a, g_2b_a = build_dataset_2b(
    ADL_CODES_11, list(range(1, 68)),
    include_spectral=True, include_gyro=False, include_orient=False,
)
print("  Comparison: profile_only (no spectral/gyro/orient)...")
Xb_2b_profOnly, Xf_2b_profOnly, y_2b_p, g_2b_p = build_dataset_2b(
    ADL_CODES_11, list(range(1, 68)),
    include_spectral=False, include_gyro=False, include_orient=False,
)
print("  Comparison: full (WITH orientation)...")
Xb_2b_withOrient, Xf_2b_withOrient, y_2b_o, g_2b_o = build_dataset_2b(
    ADL_CODES_11, list(range(1, 68)),
    include_spectral=True, include_gyro=True, include_orient=True,
)

LABELS_2B = sorted(set(y_2b))
print(f"\nFINAL    : bins{Xb_2b_final.shape}       flat{Xf_2b_final.shape}")
print(f"acc-only : bins{Xb_2b_accOnly.shape}    flat{Xf_2b_accOnly.shape}")
print(f"prof-only: bins{Xb_2b_profOnly.shape}   flat{Xf_2b_profOnly.shape}")
print(f"w/orient : bins{Xb_2b_withOrient.shape} flat{Xf_2b_withOrient.shape}")


# ============================================================
# MODEL COMPARISON — on the LOCKED FINAL feature set
# ============================================================

print("\n" + "=" * 60)
print("STAGE 2b — MODEL COMPARISON (FINAL: profile+spectral+gyro, no orient)")
print("=" * 60)
all_results_2b = {}
for name in MODEL_LIST:
    seed_metrics = [
        run_named_model(
            name, Xb_2b_final, Xf_2b_final, y_2b, g_2b, LABELS_2B, s,
            dl_epochs=150, dl_hidden=16,
            save_ckpt=(f'{CKPT_DIR_2B}/{name}_seed{s}.pt' if name in ['BiLSTM', 'Fusion'] else None),
        )
        for s in seeds_for(name)
    ]
    all_results_2b[name] = seed_metrics
    print_m_seeds(name, seed_metrics)

best_model_2b = max(MODEL_LIST, key=lambda n: np.mean([m['accuracy'] for m in all_results_2b[n]]))
print(f"\n>>> STAGE 2b BEST MODEL: {best_model_2b}")


# ============================================================
# FEATURE ABLATION on the ACTUAL BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print(f"STAGE 2b — FEATURE ABLATION on {best_model_2b}")
print("=" * 60)

variants_2b = {
    'profile_only': (Xb_2b_profOnly, Xf_2b_profOnly, y_2b_p, g_2b_p),
    'profile+spectral (acc-only, no gyro)': (Xb_2b_accOnly, Xf_2b_accOnly, y_2b_a, g_2b_a),
    'profile+spectral+gyro (FINAL, no orient)': (Xb_2b_final, Xf_2b_final, y_2b, g_2b),
    'full (with orientation)': (Xb_2b_withOrient, Xf_2b_withOrient, y_2b_o, g_2b_o),
}
ablation_2b = {}
for name, (Xb, Xf, yv, gv) in variants_2b.items():
    seed_metrics = [
        run_named_model(best_model_2b, Xb, Xf, yv, gv, LABELS_2B, s, 150, 16)
        for s in seeds_for(best_model_2b)
    ]
    ablation_2b[name] = seed_metrics
    print_m_seeds(f"{name} ({Xf.shape[1]}-dim flat, {Xb.shape[2]}-ch bins)", seed_metrics)


# ============================================================
# SIGNIFICANCE TESTS
# ============================================================

print("\n" + "=" * 60)
print("STAGE 2b — SIGNIFICANCE")
print("=" * 60)
sig_2b = {}
if best_model_2b not in DETERMINISTIC_MODELS:
    accs_final = [m['accuracy'] for m in ablation_2b['profile+spectral+gyro (FINAL, no orient)']]
    accs_withOrient = [m['accuracy'] for m in ablation_2b['full (with orientation)']]
    sig_2b['final_vs_withOrientation'] = paired_tests(
        "FINAL (no orient)", accs_final, "with orientation", accs_withOrient
    )
    accs_accOnly = [m['accuracy'] for m in ablation_2b['profile+spectral (acc-only, no gyro)']]
    sig_2b['final_vs_accOnly'] = paired_tests(
        "FINAL (with gyro)", accs_final, "acc-only (no gyro)", accs_accOnly
    )
else:
    print(f"  {best_model_2b} is deterministic — no significance test")


# ============================================================
# SAVE RESULTS
# ============================================================

summarize_full(all_results_2b).to_csv(f'{RESULTS_DIR_2B}/stage2b_models_FINAL.csv', index=False)
summarize_full(ablation_2b).to_csv(f'{RESULTS_DIR_2B}/stage2b_ablation_FINAL.csv', index=False)
with open(f'{RESULTS_DIR_2B}/stage2b_significance_FINAL.json', 'w') as f:
    json.dump(sig_2b, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"STAGE 2b COMPLETE — best model: {best_model_2b}")
print(f"Locked feature set: profile+spectral+gyro (no orientation)")
print(f"{'='*60}")
