"""
STAGE 2a — FINAL COMPLETE STANDALONE SCRIPT
Fall Subtype Classification (4-class: BSC, FKL, FOL, SDL)

Locked FINAL feature set: acc-only (15-profile + 10 scalar + jerk)
Evidence: gyro adds NO significant gain on Stage 2a 4-class fall subtype
Models: LDA, KNN-3, SVM-RBF, RandomForest, BiLSTM, Fusion
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
from scipy.signal import find_peaks
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
    roc_auc_score,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

DATA_ROOT = 'MobiAct_Dataset_v2.0/Annotated Data'
assert os.path.exists(DATA_ROOT), f"DATA_ROOT missing at {DATA_ROOT}"

PROJECT_ROOT = '.'
CKPT_DIR_2A = f'{PROJECT_ROOT}/models/checkpoints/stage2a'
RESULTS_DIR_2A = f'{PROJECT_ROOT}/results/stage2a'
os.makedirs(CKPT_DIR_2A, exist_ok=True)
os.makedirs(RESULTS_DIR_2A, exist_ok=True)

FALL_CODES = ['BSC', 'FKL', 'FOL', 'SDL']

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
            'precision_mean': np.mean(precs), 'precision_std': np.std(precs) if n > 1 else None,
            'recall_mean': np.mean(recs), 'recall_std': np.std(recs) if n > 1 else None,
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
# FEATURE BUILDERS — bins+flat ALWAYS toggled together
# include_gyro flag controls acc-only vs with-gyro variants
# ============================================================

def build_binned_2a(seg, n_bins=5, include_gyro=True):
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


def build_flat_2a(seg, n_bins=5, include_gyro=True):
    acc = seg[['acc_x', 'acc_y', 'acc_z']].values
    if len(acc) < n_bins:
        return None
    bin_edges = np.linspace(0, len(acc), n_bins + 1).astype(int)
    profile = []
    for axis in range(3):
        e = acc[:, axis] ** 2
        ap = np.array([e[bin_edges[i]:bin_edges[i + 1]].sum() for i in range(n_bins)])
        profile.append(ap / (ap.sum() + 1e-8))
    profile = np.concatenate(profile)  # 15
    acc_mag = np.sqrt((acc ** 2).sum(axis=1))
    peak_idx = int(np.argmax(acc_mag))
    peak_vec = acc[peak_idx]
    denom = np.abs(peak_vec).sum() + 1e-8
    pax = np.abs(peak_vec) / denom
    t2p = peak_idx / len(acc_mag)
    peaks, _ = find_peaks(acc_mag, height=acc_mag.mean() + acc_mag.std(), distance=5)
    nsp = len(peaks)
    a_skew, a_kurt = skew(acc_mag), kurtosis(acc_mag)
    settled = np.where(np.abs(acc_mag[peak_idx:] - 9.8) < 1.0)[0]
    settle = (settled[0] / len(acc_mag)) if len(settled) > 0 else 1.0
    jerk = np.gradient(acc_mag)
    max_jerk = np.max(np.abs(jerk))
    onset_len = min(20, len(acc_mag))
    onset_window = acc_mag[:onset_len]
    try:
        onset_slope = np.polyfit(np.arange(onset_len), onset_window, 1)[0]
    except Exception:
        onset_slope = 0.0
    scalar = [pax[0], pax[1], pax[2], nsp, max_jerk, t2p, settle, a_skew, a_kurt, onset_slope]  # 10
    if include_gyro:
        gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values
        gyro_mag = np.sqrt((gyro ** 2).sum(axis=1))
        ge = gyro ** 2
        tge = ge.sum() + 1e-8
        gx, gy, gz = ge[:, 0].sum() / tge, ge[:, 1].sum() / tge, ge[:, 2].sum() / tge
        g_skew, g_kurt, g_mean = skew(gyro_mag), kurtosis(gyro_mag), gyro_mag.mean()
        roll_std = seg['roll'].std() if 'roll' in seg.columns else 0.0
        pitch_delta = (seg['pitch'].values[-1] - seg['pitch'].values[0]) if 'pitch' in seg.columns else 0.0
        xz_corr = np.corrcoef(acc[:, 0], acc[:, 2])[0, 1] if len(acc) > 10 else 0.0
        xz_corr = xz_corr if not np.isnan(xz_corr) else 0.0
        gyro_vec = [gx, gy, gz, g_skew, g_kurt, g_mean, roll_std, pitch_delta, xz_corr]  # 9
        return np.concatenate([profile, scalar, gyro_vec]).astype(np.float32)
    return np.concatenate([profile, scalar]).astype(np.float32)


def build_dataset_2a(codes, subjects, include_gyro):
    Xb, Xf, y, g = [], [], [], []
    for code in codes:
        for subj in subjects:
            seg = get_segment(code, subj)
            if seg is None:
                continue
            fb = build_binned_2a(seg, include_gyro=include_gyro)
            fc = build_flat_2a(seg, include_gyro=include_gyro)
            if fb is not None and fc is not None:
                Xb.append(fb); Xf.append(fc); y.append(code); g.append(subj)
    return np.array(Xb), np.array(Xf), np.array(y), np.array(g)


# ============================================================
# MODELS
# ============================================================

def get_classical_models(seed):
    return {
        'LDA': LinearDiscriminantAnalysis(),
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
# FINAL (locked) = acc-only (profile + scalar + jerk)
# Ablation comparison = with-gyro variant
# ============================================================

print("Building Stage 2a datasets...")
print("  FINAL (acc-only, no gyro)...")
Xb_2a_acc, Xf_2a_acc, y_2a, g_2a = build_dataset_2a(FALL_CODES, list(range(1, 68)), include_gyro=False)
print("  Ablation comparison: with-gyro...")
Xb_2a_gyro, Xf_2a_gyro, y_2a_g, g_2a_g = build_dataset_2a(FALL_CODES, list(range(1, 68)), include_gyro=True)

LABELS_2A = sorted(set(y_2a))
print(f"acc-only : bins{Xb_2a_acc.shape}  flat{Xf_2a_acc.shape}")
print(f"with-gyro: bins{Xb_2a_gyro.shape} flat{Xf_2a_gyro.shape}")


# ============================================================
# MODEL COMPARISON — on the LOCKED FINAL feature set (acc-only)
# ============================================================

print("\n" + "=" * 60)
print("STAGE 2a — MODEL COMPARISON (acc-only, FINAL feature set)")
print("=" * 60)
all_results_2a = {}
for name in MODEL_LIST:
    seed_metrics = [
        run_named_model(
            name, Xb_2a_acc, Xf_2a_acc, y_2a, g_2a, LABELS_2A, s,
            dl_epochs=300, dl_hidden=8,
            save_ckpt=(f'{CKPT_DIR_2A}/{name}_seed{s}.pt' if name in ['BiLSTM', 'Fusion'] else None),
        )
        for s in seeds_for(name)
    ]
    all_results_2a[name] = seed_metrics
    print_m_seeds(name, seed_metrics)

best_model_2a = max(MODEL_LIST, key=lambda n: np.mean([m['accuracy'] for m in all_results_2a[n]]))
print(f"\n>>> STAGE 2a BEST MODEL: {best_model_2a}")


# ============================================================
# FEATURE ABLATION on the ACTUAL BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print(f"STAGE 2a — FEATURE ABLATION on {best_model_2a} (acc-only vs with-gyro)")
print("=" * 60)
ablation_2a = {}
for variant_name, (Xb, Xf, yv, gv) in {
    'acc_only (FINAL)': (Xb_2a_acc, Xf_2a_acc, y_2a, g_2a),
    'with_gyro (full)': (Xb_2a_gyro, Xf_2a_gyro, y_2a_g, g_2a_g),
}.items():
    seed_metrics = [
        run_named_model(best_model_2a, Xb, Xf, yv, gv, LABELS_2A, s, 300, 8)
        for s in seeds_for(best_model_2a)
    ]
    ablation_2a[variant_name] = seed_metrics
    print_m_seeds(f"{variant_name} ({Xf.shape[1]}-dim flat, {Xb.shape[2]}-ch bins)", seed_metrics)


# ============================================================
# SIGNIFICANCE TESTS
# ============================================================

print("\n" + "=" * 60)
print("STAGE 2a — SIGNIFICANCE")
print("=" * 60)
sig_2a = {}
if best_model_2a not in DETERMINISTIC_MODELS:
    sig_2a['accOnly_vs_withGyro'] = paired_tests(
        "acc_only",
        [m['accuracy'] for m in ablation_2a['acc_only (FINAL)']],
        "with_gyro",
        [m['accuracy'] for m in ablation_2a['with_gyro (full)']],
    )
else:
    print(f"  {best_model_2a} is deterministic — no significance test (single run each)")


# ============================================================
# SAVE RESULTS
# ============================================================

summarize_full(all_results_2a).to_csv(f'{RESULTS_DIR_2A}/stage2a_models_FINAL.csv', index=False)
summarize_full(ablation_2a).to_csv(f'{RESULTS_DIR_2A}/stage2a_ablation_FINAL.csv', index=False)
with open(f'{RESULTS_DIR_2A}/stage2a_significance_FINAL.json', 'w') as f:
    json.dump(sig_2a, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"STAGE 2a COMPLETE — best model: {best_model_2a}")
print(f"Locked feature set: acc-only (profile + fall-scalar + jerk)")
print(f"{'='*60}")
