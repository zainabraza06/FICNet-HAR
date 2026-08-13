"""
Stage 1 Model Evaluation Script.
Evaluates 5 classical ML models, SimpleBiLSTM, Dual-Branch Fusion, and Residual Tree Correction.
Saves checkpoints and results, conducts statistical tests, performs ablation and ensembling.
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
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Import local pipeline components
from src.data_loader import MobiActLoader, FALL_CODES, ADL_CODES_11
from src.models import SimpleBiLSTM, DualBranchFusionNet
from src.features import build_binned_features, build_stage1_flat_features
from src.pipeline import build_stage1_neural_dataset

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Local paths config
DATA_ROOT = 'MobiAct_Dataset_v2.0/Annotated Data'
assert os.path.exists(DATA_ROOT), f"DATA_ROOT missing at {DATA_ROOT}"

PROJECT_ROOT = '.'
CKPT_DIR = f'{PROJECT_ROOT}/models/checkpoints/stage1'
RESULTS_DIR = f'{PROJECT_ROOT}/results/stage1'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]


# ============================================================
# SECTION 1: MODELS & RUNNERS
# ============================================================

def get_classical_models(seed):
    return {
        'LDA': LinearDiscriminantAnalysis(),
        'KNN-3': KNeighborsClassifier(n_neighbors=3, weights='distance'),
        'SVM-RBF': SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=seed),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed),
        'GradBoost': GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=seed),
    }


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


def full_metrics(y_true, y_pred, labels, y_score=None):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'macro_precision': prec,
        'macro_recall': rec,
        'macro_f1': f1,
        'confusion_matrix': cm.tolist(),
    }
    if y_score is not None:
        try:
            y_true_bin = np.array([1 if v == labels[1] else 0 for v in y_true])
            metrics['roc_auc'] = roc_auc_score(y_true_bin, y_score)
        except Exception:
            metrics['roc_auc'] = None
    return metrics


def run_loso_classical_full(X, y, groups, model_builder, labels):
    logo = LeaveOneGroupOut()
    all_true, all_pred, all_score = [], [], []
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
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(Xte)
            fall_idx = list(clf.classes_).index(labels[1]) if labels[1] in clf.classes_ else -1
            if fall_idx >= 0:
                all_score.extend(proba[:, fall_idx])
    y_score = all_score if len(all_score) == len(all_true) else None
    return full_metrics(all_true, all_pred, labels, y_score), all_true, all_pred


def run_loso_dl_full(X, y, groups, epochs, hidden, seed, labels, save_best_ckpt=None):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(labels)
    per_bin_dim = X.shape[2]
    logo = LeaveOneGroupOut()
    all_true, all_pred, all_score = [], [], []
    best_model_state, best_fold_acc = None, -1
    for tr, te in logo.split(X, y_idx, groups):
        if len(set(y_idx[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr].reshape(-1, per_bin_dim))
        Xtr = sc.transform(X[tr].reshape(-1, per_bin_dim)).reshape(X[tr].shape)
        Xte = sc.transform(X[te].reshape(-1, per_bin_dim)).reshape(X[te].shape)
        model = train_bilstm(Xtr, y_idx[tr], n_classes, per_bin_dim, epochs, hidden, seed)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(Xte, dtype=torch.float32).to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred = logits.argmax(1).cpu().numpy()
        fold_acc = accuracy_score(y_idx[te], pred)
        if fold_acc > best_fold_acc:
            best_fold_acc = fold_acc
            best_model_state = model.state_dict()
        all_true.extend(y_idx[te])
        all_pred.extend(pred)
        all_score.extend(probs[:, 1])  # prob of class index 1
    if save_best_ckpt and best_model_state is not None:
        torch.save(best_model_state, save_best_ckpt)
    true_lbls = [labels[i] for i in all_true]
    pred_lbls = [labels[i] for i in all_pred]
    return full_metrics(true_lbls, pred_lbls, labels, all_score), true_lbls, pred_lbls


def run_loso_fusion_full(X_bins, X_flat, y, groups, epochs, seed, labels, save_best_ckpt=None):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(labels)
    per_bin_dim = X_bins.shape[2]
    flat_dim = X_flat.shape[1]
    logo = LeaveOneGroupOut()
    all_true, all_pred, all_score = [], [], []
    best_model_state, best_fold_acc = None, -1
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
            logits = model(
                torch.tensor(Xb_te, dtype=torch.float32).to(device),
                torch.tensor(Xf_te, dtype=torch.float32).to(device),
            )
            probs = F.softmax(logits, dim=1).cpu().numpy()
            pred = logits.argmax(1).cpu().numpy()
        fold_acc = accuracy_score(y_idx[te], pred)
        if fold_acc > best_fold_acc:
            best_fold_acc = fold_acc
            best_model_state = model.state_dict()
        all_true.extend(y_idx[te])
        all_pred.extend(pred)
        all_score.extend(probs[:, 1])
    if save_best_ckpt and best_model_state is not None:
        torch.save(best_model_state, save_best_ckpt)
    true_lbls = [labels[i] for i in all_true]
    pred_lbls = [labels[i] for i in all_pred]
    return full_metrics(true_lbls, pred_lbls, labels, all_score), true_lbls, pred_lbls


def apply_residual_correction_full(X_flat, y, groups, labels, seed):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(labels)
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr, te in logo.split(X_flat, y_idx, groups):
        if len(set(y_idx[tr])) < 2:
            continue
        sc = StandardScaler().fit(X_flat[tr])
        Xtr, Xte = sc.transform(X_flat[tr]), sc.transform(X_flat[te])
        base_clf = LinearDiscriminantAnalysis()
        base_clf.fit(Xtr, y_idx[tr])
        base_pred_tr = base_clf.predict(Xtr)
        base_pred_te = base_clf.predict(Xte)
        onehot_tr = np.eye(n_classes)[base_pred_tr]
        onehot_te = np.eye(n_classes)[base_pred_te]
        tree_X_tr = np.hstack([Xtr, onehot_tr])
        tree_X_te = np.hstack([Xte, onehot_te])
        tree = DecisionTreeClassifier(max_depth=4, random_state=seed)
        tree.fit(tree_X_tr, y_idx[tr])
        corrected = tree.predict(tree_X_te)
        all_true.extend(y_idx[te])
        all_pred.extend(corrected)
    true_lbls = [labels[i] for i in all_true]
    pred_lbls = [labels[i] for i in all_pred]
    return full_metrics(true_lbls, pred_lbls, labels), true_lbls, pred_lbls


# ============================================================
# SECTION 2: FEATURE-CHANNEL ABLATION RUNNER
# ============================================================

CHANNEL_GROUPS_S1 = {
    'acc_energy_only': list(range(0, 3)),
    'acc_only (energy+mag)': list(range(0, 5)),
    'acc+gyro_energy (no acc_mag)': list(range(0, 3)) + list(range(5, 8)),
    'full (acc+mag+gyro)': list(range(0, 8)),
    'gyro_only': list(range(5, 8)),
}


def run_loso_dl_channel_ablation(X_bins, y, groups, channel_idx, epochs, hidden, seed, labels):
    X_sub = X_bins[:, :, channel_idx]
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(labels)
    per_bin_dim = X_sub.shape[2]
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr, te in logo.split(X_sub, y_idx, groups):
        if len(set(y_idx[tr])) < 2:
            continue
        sc = StandardScaler().fit(X_sub[tr].reshape(-1, per_bin_dim))
        Xtr = sc.transform(X_sub[tr].reshape(-1, per_bin_dim)).reshape(X_sub[tr].shape)
        Xte = sc.transform(X_sub[te].reshape(-1, per_bin_dim)).reshape(X_sub[te].shape)
        model = train_bilstm(Xtr, y_idx[tr], n_classes, per_bin_dim, epochs, hidden, seed)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
        all_true.extend(y_idx[te])
        all_pred.extend(pred)
    return accuracy_score(all_true, all_pred), balanced_accuracy_score(all_true, all_pred)


# ============================================================
# SECTION 3: STATISTICAL SIGNIFICANCE TESTS
# ============================================================

def paired_tests(name_a, accs_a, name_b, accs_b):
    accs_a, accs_b = np.array(accs_a), np.array(accs_b)
    t_stat, t_p = stats.ttest_rel(accs_a, accs_b)
    try:
        w_stat, w_p = stats.wilcoxon(accs_a, accs_b)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    sig_t = t_p < 0.05 if not np.isnan(t_p) else False
    sig_w = (not np.isnan(w_p)) and w_p < 0.05
    print(f"\n{name_a} vs {name_b}:")
    print(f"  {name_a}: {np.round(accs_a,4)}")
    print(f"  {name_b}: {np.round(accs_b,4)}")
    print(f"  Paired t-test: t={t_stat:.3f}, p={t_p:.4f} {'** SIG **' if sig_t else '(n.s., n=5 caveat)'}")
    print(f"  Wilcoxon:      p={w_p if not np.isnan(w_p) else 'degenerate'} {'** SIG **' if sig_w else ''}")
    return {'t_stat': t_stat, 't_p': t_p, 'w_p': w_p if not np.isnan(w_p) else None}


# ============================================================
# SECTION 4: ENSEMBLING
# ============================================================

def run_loso_ensemble(X_bins, X_flat, y, groups, labels, epochs_dl=200, hidden=8,
                       epochs_fusion=200, seed=0):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(labels)
    per_bin_dim = X_bins.shape[2]
    flat_dim = X_flat.shape[1]
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr, te in logo.split(X_bins, y_idx, groups):
        if len(set(y_idx[tr])) < 2:
            continue
        # classical
        sc_f = StandardScaler().fit(X_flat[tr])
        Xf_tr, Xf_te = sc_f.transform(X_flat[tr]), sc_f.transform(X_flat[te])
        svm = SVC(kernel='rbf', probability=True, random_state=seed).fit(Xf_tr, y_idx[tr])
        svm_proba = svm.predict_proba(Xf_te)

        # BiLSTM
        sc_b = StandardScaler().fit(X_bins[tr].reshape(-1, per_bin_dim))
        Xb_tr = sc_b.transform(X_bins[tr].reshape(-1, per_bin_dim)).reshape(X_bins[tr].shape)
        Xb_te = sc_b.transform(X_bins[te].reshape(-1, per_bin_dim)).reshape(X_bins[te].shape)
        bilstm = train_bilstm(Xb_tr, y_idx[tr], n_classes, per_bin_dim, epochs_dl, hidden, seed)
        bilstm.eval()
        with torch.no_grad():
            bilstm_proba = F.softmax(bilstm(torch.tensor(Xb_te, dtype=torch.float32).to(device)), dim=1).cpu().numpy()

        # Fusion
        fusion = train_fusion(Xb_tr, Xf_tr, y_idx[tr], n_classes, per_bin_dim, flat_dim, epochs_fusion, seed)
        fusion.eval()
        with torch.no_grad():
            fusion_proba = F.softmax(fusion(
                torch.tensor(Xb_te, dtype=torch.float32).to(device),
                torch.tensor(Xf_te, dtype=torch.float32).to(device)
            ), dim=1).cpu().numpy()

        avg_proba = (svm_proba + bilstm_proba + fusion_proba) / 3.0
        pred = avg_proba.argmax(axis=1)
        all_true.extend(y_idx[te])
        all_pred.extend(pred)

    true_lbls = [labels[i] for i in all_true]
    pred_lbls = [labels[i] for i in all_pred]
    return full_metrics(true_lbls, pred_lbls, labels)


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Stage 1 Model Evaluation.")
    parser.add_argument('--fast', action='store_true', help="Run a quick dry-run with a subset of subjects and epochs.")
    args = parser.parse_args()

    loader = MobiActLoader(DATA_ROOT)
    
    # List subjects available
    subjects = loader.list_subjects(FALL_CODES[0])
    if not subjects:
        subjects = list(range(1, 68))
        
    epochs = 200
    if args.fast:
        print("Running in FAST mode (subset of subjects and epochs)")
        subjects = subjects[:3]
        epochs = 5
    
    print("Building Stage 1 dataset...")
    X_bins_s1, X_flat_s1, y_s1, groups_s1 = build_stage1_neural_dataset(
        loader, FALL_CODES, ADL_CODES_11, subjects
    )
    print(f"X_bins: {X_bins_s1.shape}, X_flat: {X_flat_s1.shape}")
    print(f"Class balance: {dict(zip(*np.unique(y_s1, return_counts=True)))}")

    LABELS_S1 = ['ADL', 'FALL']  # index 0=ADL, 1=FALL (for AUC/positive class)
    all_results = {}

    print("\n" + "=" * 60)
    print("CLASSICAL MODELS (5 seeds each)")
    print("=" * 60)
    for name in ['LDA', 'KNN-3', 'SVM-RBF', 'RandomForest', 'GradBoost']:
        seed_metrics = []
        for seed in SEEDS:
            m, _, _ = run_loso_classical_full(
                X_flat_s1, y_s1, groups_s1,
                lambda n=name, s=seed: get_classical_models(s)[n], LABELS_S1
            )
            seed_metrics.append(m)
        all_results[name] = seed_metrics
        accs = [m['accuracy'] for m in seed_metrics]
        bals = [m['balanced_accuracy'] for m in seed_metrics]
        print(f"  {name}: acc={np.mean(accs):.4f}±{np.std(accs):.4f}  bal={np.mean(bals):.4f}±{np.std(bals):.4f}")

    print("\n" + "=" * 60)
    print("BiLSTM (DL-only), 5 seeds, checkpointing best fold")
    print("=" * 60)
    seed_metrics = []
    for seed in SEEDS:
        ckpt_path = f'{CKPT_DIR}/bilstm_seed{seed}_best_fold.pt'
        m, _, _ = run_loso_dl_full(
            X_bins_s1, y_s1, groups_s1, epochs=epochs, hidden=8, seed=seed,
            labels=LABELS_S1, save_best_ckpt=ckpt_path
        )
        seed_metrics.append(m)
        print(f"  seed {seed}: acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} f1={m['macro_f1']:.4f}")
    all_results['BiLSTM'] = seed_metrics

    print("\n" + "=" * 60)
    print("Dual-Branch Fusion, 5 seeds, checkpointing best fold")
    print("=" * 60)
    seed_metrics = []
    for seed in SEEDS:
        ckpt_path = f'{CKPT_DIR}/fusion_seed{seed}_best_fold.pt'
        m, _, _ = run_loso_fusion_full(
            X_bins_s1, X_flat_s1, y_s1, groups_s1, epochs=epochs, seed=seed,
            labels=LABELS_S1, save_best_ckpt=ckpt_path
        )
        seed_metrics.append(m)
        print(f"  seed {seed}: acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} f1={m['macro_f1']:.4f}")
    all_results['Fusion'] = seed_metrics

    print("\n" + "=" * 60)
    print("Residual Tree Correction (on LDA), 5 seeds")
    print("=" * 60)
    seed_metrics = []
    for seed in SEEDS:
        m, _, _ = apply_residual_correction_full(X_flat_s1, y_s1, groups_s1, LABELS_S1, seed)
        seed_metrics.append(m)
    all_results['Residual'] = seed_metrics
    accs = [m['accuracy'] for m in seed_metrics]
    print(f"  Residual: acc={np.mean(accs):.4f}±{np.std(accs):.4f}")

    # Feature ablation
    print("\n" + "=" * 60)
    print("FEATURE-CHANNEL ABLATION on FINAL MODEL (BiLSTM, 5 seeds)")
    print("=" * 60)
    ablation_results = {}
    for group_name, idx in CHANNEL_GROUPS_S1.items():
        accs, bals = [], []
        for seed in SEEDS:
            acc, bal = run_loso_dl_channel_ablation(X_bins_s1, y_s1, groups_s1, idx, epochs, 8, seed, LABELS_S1)
            accs.append(acc)
            bals.append(bal)
        ablation_results[group_name] = {'accs': accs, 'bals': bals}
        print(f"  {group_name:<32s} ({len(idx)} ch): acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
              f"bal={np.mean(bals):.4f}±{np.std(bals):.4f}")

    # Significance testing
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)
    sig_results = {}
    accs_bilstm = [m['accuracy'] for m in all_results['BiLSTM']]
    accs_svm = [m['accuracy'] for m in all_results['SVM-RBF']]
    accs_fusion = [m['accuracy'] for m in all_results['Fusion']]
    accs_lda = [m['accuracy'] for m in all_results['LDA']]

    sig_results['bilstm_vs_svm'] = paired_tests("BiLSTM", accs_bilstm, "SVM-RBF", accs_svm)
    sig_results['bilstm_vs_fusion'] = paired_tests("BiLSTM", accs_bilstm, "Fusion", accs_fusion)
    sig_results['fusion_vs_lda'] = paired_tests("Fusion", accs_fusion, "LDA", accs_lda)
    sig_results['full_vs_accOnly_ablation'] = paired_tests(
        "Full (gyro incl.)", ablation_results['full (acc+mag+gyro)']['accs'],
        "Acc-only", ablation_results['acc_only (energy+mag)']['accs']
    )

    # Ensemble
    print("\n" + "=" * 60)
    print("ENSEMBLE (BiLSTM + Fusion + SVM-RBF, probability averaging)")
    print("=" * 60)
    ensemble_metrics = run_loso_ensemble(
        X_bins_s1, X_flat_s1, y_s1, groups_s1, LABELS_S1,
        epochs_dl=epochs, epochs_fusion=epochs, seed=0
    )
    print(f"Ensemble: acc={ensemble_metrics['accuracy']:.4f} bal={ensemble_metrics['balanced_accuracy']:.4f} "
          f"f1={ensemble_metrics['macro_f1']:.4f}")


    # Save results
    def summarize(results_dict):
        rows = []
        for name, seed_metrics in results_dict.items():
            accs = [m['accuracy'] for m in seed_metrics]
            bals = [m['balanced_accuracy'] for m in seed_metrics]
            f1s = [m['macro_f1'] for m in seed_metrics]
            precs = [m['macro_precision'] for m in seed_metrics]
            recs = [m['macro_recall'] for m in seed_metrics]
            rows.append({
                'method': name,
                'acc_mean': np.mean(accs), 'acc_std': np.std(accs),
                'bal_mean': np.mean(bals), 'bal_std': np.std(bals),
                'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s),
                'precision_mean': np.mean(precs), 'recall_mean': np.mean(recs)
            })
        return pd.DataFrame(rows)

    summary_df = summarize(all_results)
    summary_df.to_csv(f'{RESULTS_DIR}/stage1_model_comparison.csv', index=False)
    print(f"\nSaved model comparison to {RESULTS_DIR}/stage1_model_comparison.csv")
    print(summary_df.to_string(index=False))

    ablation_df = pd.DataFrame([
        {'feature_group': k, 'n_channels': len(CHANNEL_GROUPS_S1[k]),
         'acc_mean': np.mean(v['accs']), 'acc_std': np.std(v['accs']),
         'bal_mean': np.mean(v['bals']), 'bal_std': np.std(v['bals'])}
        for k, v in ablation_results.items()
    ])
    ablation_df.to_csv(f'{RESULTS_DIR}/stage1_feature_ablation.csv', index=False)
    print(f"\nSaved feature ablation to {RESULTS_DIR}/stage1_feature_ablation.csv")
    print(ablation_df.to_string(index=False))

    with open(f'{RESULTS_DIR}/stage1_significance_tests.json', 'w') as f:
        json.dump(sig_results, f, indent=2, default=str)
    print(f"\nSaved significance tests to {RESULTS_DIR}/stage1_significance_tests.json")

    with open(f'{RESULTS_DIR}/stage1_ensemble_metrics.json', 'w') as f:
        json.dump({k: v for k, v in ensemble_metrics.items()}, f, indent=2, default=str)
    print(f"Saved ensemble metrics to {RESULTS_DIR}/stage1_ensemble_metrics.json")

    print("\n" + "=" * 60)
    print("STAGE 1 — FULLY COMPLETE. All results and checkpoints saved.")
    print("=" * 60)


if __name__ == '__main__':
    main()
