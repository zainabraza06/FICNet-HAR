"""
LOSO evaluation wrappers for the cross-domain ablation study.

Supports two model families used across the three MobiAct stages:

  Classical (SVM-RBF, RandomForest, LDA, KNN-3)
      run_loso_classical()   — works on flat X_flat feature matrix.

  DualBranchFusionNet  (best model for Stage 2a)
      run_loso_fusion()      — works on (X_bins, X_flat) pair.

Both functions return a metrics dict with:
    accuracy, balanced_accuracy, macro_precision,
    macro_recall, macro_f1
averaged (mean ± std) over the supplied SEEDS list.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)

from .models import DualBranchFusionNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Shared metric helper
# ============================================================

def _metrics(y_true, y_pred, labels):
    """Return a flat dict of scalar metrics for one LOSO run."""
    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    return {
        'accuracy':          acc,
        'balanced_accuracy': bal_acc,
        'macro_precision':   prec,
        'macro_recall':      rec,
        'macro_f1':          f1,
    }


# ============================================================
# Classical model helpers
# ============================================================

def get_classical_models(seed: int) -> dict:
    """Return a fresh dict of named classical classifiers for a given seed."""
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.neighbors import KNeighborsClassifier
    return {
        'LDA':          LinearDiscriminantAnalysis(),
        'KNN-3':        KNeighborsClassifier(n_neighbors=3, weights='distance'),
        'SVM-RBF':      SVC(kernel='rbf', C=1.0, gamma='scale',
                            probability=True, random_state=seed),
        'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=8,
                                               random_state=seed),
    }


def run_loso_classical(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       model_name: str, labels: list[str],
                       seeds: list[int]) -> dict:
    """
    LOSO cross-validation for a classical classifier averaged over seeds.

    Parameters
    ----------
    X          : flat feature matrix (N, flat_dim)
    y          : string label array  (N,)
    groups     : subject-ID array    (N,)
    model_name : key into get_classical_models()  e.g. 'SVM-RBF'
    labels     : ordered list of class labels
    seeds      : list of random seeds

    Returns
    -------
    dict  { 'accuracy':   (mean, std),
            'bal':        (mean, std),
            'f1':         (mean, std),
            'raw':        list[dict]  }
    """
    logo = LeaveOneGroupOut()
    seed_metrics = []

    for seed in seeds:
        all_true, all_pred = [], []
        clf_factory = lambda s=seed: get_classical_models(s)[model_name]

        for tr, te in logo.split(X, y, groups):
            if len(set(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = clf_factory()
            clf.fit(sc.transform(X[tr]), y[tr])
            pred = clf.predict(sc.transform(X[te]))
            all_true.extend(y[te])
            all_pred.extend(pred)

        seed_metrics.append(_metrics(all_true, all_pred, labels))

    return _aggregate(seed_metrics)


# ============================================================
# Fusion model training + LOSO
# ============================================================

def _train_fusion(Xb_tr: np.ndarray, Xf_tr: np.ndarray,
                  y_idx_tr: np.ndarray, n_classes: int,
                  per_bin_dim: int, flat_dim: int,
                  epochs: int, seed: int) -> DualBranchFusionNet:
    torch.manual_seed(seed)
    model = DualBranchFusionNet(per_bin_dim, flat_dim, n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-2)

    Xb_t = torch.tensor(Xb_tr, dtype=torch.float32).to(DEVICE)
    Xf_t = torch.tensor(Xf_tr, dtype=torch.float32).to(DEVICE)
    yt   = torch.tensor(y_idx_tr, dtype=torch.long).to(DEVICE)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(Xb_t, Xf_t), yt).backward()
        opt.step()

    return model


def run_loso_fusion(X_bins: np.ndarray, X_flat: np.ndarray,
                    y: np.ndarray, groups: np.ndarray,
                    labels: list[str], epochs: int,
                    seeds: list[int]) -> dict:
    """
    LOSO cross-validation for DualBranchFusionNet averaged over seeds.

    Parameters
    ----------
    X_bins  : (N, n_bins, per_bin_dim)
    X_flat  : (N, flat_dim)
    y       : string label array  (N,)
    groups  : subject-ID array    (N,)
    labels  : ordered list of class labels
    epochs  : training epochs per fold
    seeds   : list of random seeds

    Returns
    -------
    Same structure as run_loso_classical.
    """
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx        = np.array([label_to_idx[l] for l in y])
    n_classes    = len(labels)
    per_bin_dim  = X_bins.shape[2]
    flat_dim     = X_flat.shape[1]
    logo         = LeaveOneGroupOut()
    seed_metrics = []

    for seed in seeds:
        all_true, all_pred = [], []

        for tr, te in logo.split(X_bins, y_idx, groups):
            if len(set(y_idx[tr])) < 2:
                continue

            # Scale bins (flatten → scale → reshape)
            scb = StandardScaler().fit(X_bins[tr].reshape(-1, per_bin_dim))
            Xb_tr = scb.transform(X_bins[tr].reshape(-1, per_bin_dim)).reshape(X_bins[tr].shape)
            Xb_te = scb.transform(X_bins[te].reshape(-1, per_bin_dim)).reshape(X_bins[te].shape)

            # Scale flat
            scf   = StandardScaler().fit(X_flat[tr])
            Xf_tr = scf.transform(X_flat[tr])
            Xf_te = scf.transform(X_flat[te])

            model = _train_fusion(Xb_tr, Xf_tr, y_idx[tr], n_classes,
                                  per_bin_dim, flat_dim, epochs, seed)
            model.eval()
            with torch.no_grad():
                logits = model(
                    torch.tensor(Xb_te, dtype=torch.float32).to(DEVICE),
                    torch.tensor(Xf_te, dtype=torch.float32).to(DEVICE),
                )
                pred = logits.argmax(1).cpu().numpy()

            all_true.extend(y_idx[te])
            all_pred.extend(pred)

        m = _metrics(
            [labels[i] for i in all_true],
            [labels[i] for i in all_pred],
            labels,
        )
        seed_metrics.append(m)

    return _aggregate(seed_metrics)


# ============================================================
# Aggregation helper
# ============================================================

def _aggregate(seed_metrics: list[dict]) -> dict:
    """Compute mean ± std across seeds for each metric key."""
    keys = seed_metrics[0].keys()
    result = {}
    for k in keys:
        vals = np.array([m[k] for m in seed_metrics])
        result[k]           = float(vals.mean())
        result[k + '_std']  = float(vals.std())
    result['raw'] = seed_metrics
    return result
