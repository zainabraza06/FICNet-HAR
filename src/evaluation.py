import numpy as np
import torch
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, 
                             precision_recall_fscore_support)
from .models import get_classical_models, train_bilstm, train_fusion

def full_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )
    return {'accuracy': acc, 'balanced_accuracy': bal_acc,
            'macro_precision': prec, 'macro_recall': rec, 'macro_f1': f1}

def print_metrics(name, seed_metrics):
    accs = [m['accuracy'] for m in seed_metrics]
    bals = [m['balanced_accuracy'] for m in seed_metrics]
    f1s = [m['macro_f1'] for m in seed_metrics]
    n = len(seed_metrics)
    print(f"  {name:<40s}: acc={np.mean(accs):.4f}" + 
          (f"±{np.std(accs):.4f}" if n > 1 else "") +
          f" bal={np.mean(bals):.4f}" +
          (f"±{np.std(bals):.4f}" if n > 1 else "") +
          f" f1={np.mean(f1s):.4f}" +
          (f"±{np.std(f1s):.4f}" if n > 1 else ""))

def run_loso_classical(X, y, groups, model_builder, labels):
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr, te in logo.split(X, y, groups):
        if len(set(y[tr])) < 2: continue
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = model_builder()
        clf.fit(Xtr, y[tr])
        pred = clf.predict(Xte)
        all_true.extend(y[te]); all_pred.extend(pred)
    return full_metrics(all_true, all_pred, labels)

def run_loso_dl(X, y, groups, epochs, hidden, seed, labels, device):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(labels); per_bin_dim = X.shape[2]
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr, te in logo.split(X, y_idx, groups):
        if len(set(y_idx[tr])) < 2: continue
        sc = StandardScaler().fit(X[tr].reshape(-1, per_bin_dim))
        Xtr = sc.transform(X[tr].reshape(-1, per_bin_dim)).reshape(X[tr].shape)
        Xte = sc.transform(X[te].reshape(-1, per_bin_dim)).reshape(X[te].shape)
        model = train_bilstm(Xtr, y_idx[tr], n_classes, per_bin_dim, epochs, hidden, seed, device)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
        all_true.extend(y_idx[te]); all_pred.extend(pred)
    return full_metrics([labels[i] for i in all_true], [labels[i] for i in all_pred], labels)

def run_loso_fusion(X_bins, X_flat, y, groups, epochs, seed, labels, device):
    label_to_idx = {l: i for i, l in enumerate(labels)}
    y_idx = np.array([label_to_idx[l] for l in y])
    n_classes = len(labels); per_bin_dim = X_bins.shape[2]; flat_dim = X_flat.shape[1]
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr, te in logo.split(X_bins, y_idx, groups):
        if len(set(y_idx[tr])) < 2: continue
        scb = StandardScaler().fit(X_bins[tr].reshape(-1, per_bin_dim))
        scf = StandardScaler().fit(X_flat[tr])
        Xb_tr = scb.transform(X_bins[tr].reshape(-1, per_bin_dim)).reshape(X_bins[tr].shape)
        Xb_te = scb.transform(X_bins[te].reshape(-1, per_bin_dim)).reshape(X_bins[te].shape)
        Xf_tr = scf.transform(X_flat[tr]); Xf_te = scf.transform(X_flat[te])
        model = train_fusion(Xb_tr, Xf_tr, y_idx[tr], n_classes, per_bin_dim, flat_dim, epochs, seed, device)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xb_te, dtype=torch.float32).to(device),
                         torch.tensor(Xf_te, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
        all_true.extend(y_idx[te]); all_pred.extend(pred)
    return full_metrics([labels[i] for i in all_true], [labels[i] for i in all_pred], labels)

def run_named_model(model_name, Xb, Xf, y, groups, labels, seed, device, epochs=200, priors=None, lstm_hidden=8):
    if model_name in ['LDA', 'KNN-3', 'SVM-RBF', 'RandomForest']:
        return run_loso_classical(Xf, y, groups, lambda n=model_name, s=seed: get_classical_models(s, priors=priors)[n], labels)
    elif model_name == 'BiLSTM':
        return run_loso_dl(Xb, y, groups, epochs, lstm_hidden, seed, labels, device)
    elif model_name == 'Fusion':
        return run_loso_fusion(Xb, Xf, y, groups, epochs, seed, labels, device)
