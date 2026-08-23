import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support)

from src.config import EPOCHS, N_CV_FOLDS
from src.data.features import select_features, FEATURE_NAMES, N_HANDCRAFTED
from src.training.train import train_model, predict


# ── Cross-Validation (StratifiedKFold, random split) ─────────────────────────

def run_cv(mode, X, y, subj, feat_raw, labels, n_folds=N_CV_FOLDS,
           epochs=EPOCHS, seed=0):
    """
    Stratified k-fold cross-validation. Splits are NOT subject-grouped —
    same subject may appear in both train and test folds. Primary value is
    rapid feedback; use LOSO for the paper's main generalization claim.

    StandardScaler and feature selection are fit on the training fold only.

    Returns
    -------
    all_true : list[str]
    all_pred : list[str]
    """
    uses_fic = mode in ('fic', 'sam_fic')
    l2i  = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([l2i[l] for l in y])
    skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_true, all_pred = [], []

    for fold_i, (tr, te) in enumerate(skf.split(X, y_idx)):
        print(f"  -- CV fold {fold_i + 1}/{n_folds} (seed {seed}) --", flush=True)

        # Scale raw windows (flatten → fit → reshape)
        sc   = StandardScaler()
        Xtr  = sc.fit_transform(X[tr].reshape(len(tr), -1)).reshape(X[tr].shape)
        Xte  = sc.transform(X[te].reshape(len(te), -1)).reshape(X[te].shape)

        feat_tr_n, feat_te_n = None, None
        if uses_fic:
            keep = select_features(feat_raw[tr])
            fsc  = StandardScaler()
            feat_tr_n = fsc.fit_transform(feat_raw[tr][:, keep]).astype('float32')
            feat_te_n = fsc.transform(feat_raw[te][:, keep]).astype('float32')
            if fold_i == 0:
                kept = [n for n, k in zip(FEATURE_NAMES, keep) if k]
                print(f"    [{mode}] features kept "
                      f"({keep.sum()}/{N_HANDCRAFTED}): {kept}", flush=True)

        model = train_model(
            mode, Xtr.astype('float32'), y_idx[tr], subj[tr],
            len(labels), X.shape[1], X.shape[2], epochs, seed,
            feat_train=feat_tr_n,
            verbose_tag=f"CV fold {fold_i + 1}/{n_folds}",
        )
        pred = predict(model, Xte.astype('float32'), labels, mode, feat=feat_te_n)
        all_true.extend(y[te])
        all_pred.extend(pred)

    return all_true, all_pred


# ── Leave-One-Subject-Out ─────────────────────────────────────────────────────

def run_loso(mode, X, y, subj, feat_raw, labels, epochs=EPOCHS, seed=0):
    """
    Leave-One-Subject-Out evaluation with POOLED confusion matrix.

    Per-fold macro-F1 averaging is avoided because rare classes (CHU, SIT,
    fall types) are absent from most individual folds' test sets, making
    per-fold metrics non-comparable across folds.

    Returns
    -------
    all_true    : list[str]
    all_pred    : list[str]
    per_fold_n  : list[dict] — {'held_out_subject', 'n_test_windows'} per fold
    """
    uses_fic = mode in ('fic', 'sam_fic')
    l2i  = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([l2i[l] for l in y])
    unique_subjects = np.unique(subj)

    all_true, all_pred, per_fold_n = [], [], []

    for fold_i, held_out in enumerate(unique_subjects):
        tr_mask = subj != held_out
        te_mask = subj == held_out
        if te_mask.sum() == 0:
            continue

        print(f"  -- LOSO fold {fold_i + 1}/{len(unique_subjects)} "
              f"(held-out subj {held_out}, seed {seed}) --", flush=True)

        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[tr_mask].reshape(tr_mask.sum(), -1)).reshape(X[tr_mask].shape)
        Xte = sc.transform(X[te_mask].reshape(te_mask.sum(), -1)).reshape(X[te_mask].shape)

        feat_tr_n, feat_te_n = None, None
        if uses_fic:
            keep = select_features(feat_raw[tr_mask])
            fsc  = StandardScaler()
            feat_tr_n = fsc.fit_transform(feat_raw[tr_mask][:, keep]).astype('float32')
            feat_te_n = fsc.transform(feat_raw[te_mask][:, keep]).astype('float32')

        model = train_model(
            mode, Xtr.astype('float32'), y_idx[tr_mask], subj[tr_mask],
            len(labels), X.shape[1], X.shape[2], epochs, seed,
            feat_train=feat_tr_n,
            verbose_tag=f"LOSO subj={held_out} ({fold_i + 1}/{len(unique_subjects)})",
        )
        pred = predict(model, Xte.astype('float32'), labels, mode, feat=feat_te_n)
        all_true.extend(y[te_mask])
        all_pred.extend(pred)
        per_fold_n.append({
            'held_out_subject': int(held_out),
            'n_test_windows':   int(te_mask.sum()),
        })

    return all_true, all_pred, per_fold_n


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, labels):
    """Return accuracy, balanced accuracy, and macro F1."""
    acc     = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    return float(acc), float(bal_acc), float(f1)
