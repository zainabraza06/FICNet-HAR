import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from src.config import DEVICE, CLASSICAL_MODELS, SEEDS
from src.models.classical import get_classical_models
from src.models.deep import train_bilstm, train_fusion

def full_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                          average='macro', zero_division=0)
    return {'accuracy': acc, 'balanced_accuracy': bal_acc,
            'macro_precision': prec, 'macro_recall': rec, 'macro_f1': f1}

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
        all_true.extend(y[te])
        all_pred.extend(pred)
    return full_metrics(all_true, all_pred, labels)

def run_loso_dl(X, y, groups, epochs, hidden, seed, labels):
    l2i = {l:i for i,l in enumerate(labels)}
    y_idx = np.array([l2i[l] for l in y])
    n_classes = len(labels)
    per_bin_dim = X.shape[2]
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr, te in logo.split(X, y_idx, groups):
        if len(set(y_idx[tr])) < 2: continue
        sc = StandardScaler().fit(X[tr].reshape(-1, per_bin_dim))
        Xtr = sc.transform(X[tr].reshape(-1, per_bin_dim)).reshape(X[tr].shape)
        Xte = sc.transform(X[te].reshape(-1, per_bin_dim)).reshape(X[te].shape)
        model = train_bilstm(Xtr, y_idx[tr], n_classes, per_bin_dim, epochs, hidden, seed)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xte, dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
        all_true.extend(y_idx[te])
        all_pred.extend(pred)
    return full_metrics([labels[i] for i in all_true], [labels[i] for i in all_pred], labels)

def run_loso_fusion(Xb, Xf, y, groups, epochs, seed, labels):
    l2i = {l:i for i,l in enumerate(labels)}
    y_idx = np.array([l2i[l] for l in y])
    n_classes = len(labels)
    per_bin_dim = Xb.shape[2]
    flat_dim = Xf.shape[1]
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr, te in logo.split(Xb, y_idx, groups):
        if len(set(y_idx[tr])) < 2: continue
        scb = StandardScaler().fit(Xb[tr].reshape(-1, per_bin_dim))
        scf = StandardScaler().fit(Xf[tr])
        Xb_tr = scb.transform(Xb[tr].reshape(-1, per_bin_dim)).reshape(Xb[tr].shape)
        Xb_te = scb.transform(Xb[te].reshape(-1, per_bin_dim)).reshape(Xb[te].shape)
        Xf_tr = scf.transform(Xf[tr])
        Xf_te = scf.transform(Xf[te])
        model = train_fusion(Xb_tr, Xf_tr, y_idx[tr], n_classes, per_bin_dim, flat_dim, epochs, seed)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(Xb_te,dtype=torch.float32).to(DEVICE),
                         torch.tensor(Xf_te,dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
        all_true.extend(y_idx[te])
        all_pred.extend(pred)
    return full_metrics([labels[i] for i in all_true], [labels[i] for i in all_pred], labels)

def run_on_model(model_name, Xb, Xf, y, g, labels, epochs=200, hidden=8):
    if model_name in CLASSICAL_MODELS:
        return [run_loso_classical(Xf, y, g, lambda n=model_name, s=seed: get_classical_models(s)[n], labels)
                for seed in SEEDS]
    elif model_name == 'BiLSTM':
        return [run_loso_dl(Xb, y, g, epochs, hidden, seed, labels) for seed in SEEDS]
    else:
        return [run_loso_fusion(Xb, Xf, y, g, epochs, seed, labels) for seed in SEEDS]

def get_loso_predictions(model_name, Xb, Xf, y, g, labels, seed=0, epochs=200, hidden=8):
    l2i = {l:i for i,l in enumerate(labels)}
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    if model_name in CLASSICAL_MODELS:
        for tr, te in logo.split(Xf, y, g):
            if len(set(y[tr])) < 2: continue
            sc = StandardScaler().fit(Xf[tr])
            Xtr, Xte = sc.transform(Xf[tr]), sc.transform(Xf[te])
            clf = get_classical_models(seed)[model_name]
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xte)
            all_true.extend(y[te])
            all_pred.extend(pred)
    else:
        y_idx = np.array([l2i[l] for l in y])
        for tr, te in logo.split(Xb, y_idx, g):
            if len(set(y_idx[tr])) < 2: continue
            if model_name == 'BiLSTM':
                per_bin_dim = Xb.shape[2]
                sc = StandardScaler().fit(Xb[tr].reshape(-1, per_bin_dim))
                Xtr = sc.transform(Xb[tr].reshape(-1, per_bin_dim)).reshape(Xb[tr].shape)
                Xte = sc.transform(Xb[te].reshape(-1, per_bin_dim)).reshape(Xb[te].shape)
                model = train_bilstm(Xtr, y_idx[tr], len(labels), per_bin_dim, epochs, hidden, seed)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.tensor(Xte,dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
            else:
                per_bin_dim = Xb.shape[2]; flat_dim = Xf.shape[1]
                scb = StandardScaler().fit(Xb[tr].reshape(-1, per_bin_dim))
                scf = StandardScaler().fit(Xf[tr])
                Xb_tr = scb.transform(Xb[tr].reshape(-1, per_bin_dim)).reshape(Xb[tr].shape)
                Xb_te = scb.transform(Xb[te].reshape(-1, per_bin_dim)).reshape(Xb[te].shape)
                Xf_tr = scf.transform(Xf[tr])
                Xf_te = scf.transform(Xf[te])
                model = train_fusion(Xb_tr, Xf_tr, y_idx[tr], len(labels), per_bin_dim, flat_dim, epochs, seed)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.tensor(Xb_te,dtype=torch.float32).to(DEVICE),
                                 torch.tensor(Xf_te,dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
            all_true.extend([labels[i] for i in y_idx[te]])
            all_pred.extend([labels[i] for i in pred])
    return all_true, all_pred
