import numpy as np
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from src.config import DEVICE, CLASSICAL_MODELS, SEEDS, ALPHA
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

def run_on_model(model_name, Xb, Xf, y, g, labels, epochs=200, n_bilstm_hidden=8):
    """n_classes derived from labels so LDA priors are always consistent."""
    n_classes = len(labels)
    if model_name in CLASSICAL_MODELS:
        return [run_loso_classical(Xf, y, g,
                                   lambda n=model_name, s=seed: get_classical_models(s, n_classes)[n],
                                   labels)
                for seed in SEEDS]
    elif model_name == 'BiLSTM':
        return [run_loso_dl(Xb, y, g, epochs, n_bilstm_hidden, seed, labels) for seed in SEEDS]
    else:
        return [run_loso_fusion(Xb, Xf, y, g, epochs, seed, labels) for seed in SEEDS]

def get_loso_predictions(model_name, Xb, Xf, y, g, labels, seed=0, epochs=200, hidden=8):
    n_classes = len(labels)
    l2i = {l:i for i,l in enumerate(labels)}
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    if model_name in CLASSICAL_MODELS:
        for tr, te in logo.split(Xf, y, g):
            if len(set(y[tr])) < 2: continue
            sc = StandardScaler().fit(Xf[tr])
            Xtr, Xte = sc.transform(Xf[tr]), sc.transform(Xf[te])
            clf = get_classical_models(seed, n_classes)[model_name]
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
                model = train_bilstm(Xtr, y_idx[tr], n_classes, per_bin_dim, epochs, hidden, seed)
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
                model = train_fusion(Xb_tr, Xf_tr, y_idx[tr], n_classes, per_bin_dim, flat_dim, epochs, seed)
                model.eval()
                with torch.no_grad():
                    pred = model(torch.tensor(Xb_te,dtype=torch.float32).to(DEVICE),
                                 torch.tensor(Xf_te,dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()
            all_true.extend([labels[i] for i in y_idx[te]])
            all_pred.extend([labels[i] for i in pred])
    return all_true, all_pred


def print_seeds(name, metrics):
    accs = [m['accuracy'] for m in metrics]
    bals = [m['balanced_accuracy'] for m in metrics]
    f1s  = [m['macro_f1'] for m in metrics]
    print(f"    {name}: acc={np.mean(accs):.4f}±{np.std(accs):.4f} "
          f"bal={np.mean(bals):.4f}±{np.std(bals):.4f} f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

def pick_best_model(model_results):
    return max(model_results, key=lambda m: np.mean([x['accuracy'] for x in model_results[m]]))

def run_stage_pipeline(stage_name, build_dataset_fn, registry, labels,
                       epochs_selection=200, bilstm_hidden=8):
    """
    Generic greedy significance-gated feature + model selection pipeline.
    Shared by Stage 1, Stage 2a, Stage 2b.
    Returns a result dict containing final model, groups, data arrays, and logs.
    """
    from src.features.extractors import core_and_candidates, feature_names_for
    from src.config import MODELS

    print("\n" + "="*70)
    print(f"{stage_name} — PART 1: MODEL SELECTION ON CORE SET")
    print("="*70)
    core, candidates = core_and_candidates(registry)

    Xb_core, Xf_core, y_core, g_core = build_dataset_fn(core)
    print(f"Core groups: {core}. Dataset: {Xb_core.shape}, {Xf_core.shape}, "
          f"classes: {sorted(set(y_core))}")

    model_results_core = {}
    for model in MODELS:
        print(f"\n  {model}:")
        metrics = run_on_model(model, Xb_core, Xf_core, y_core, g_core, labels,
                                epochs=epochs_selection, n_bilstm_hidden=bilstm_hidden)
        print_seeds(model, metrics)
        model_results_core[model] = metrics
    model_stage = pick_best_model(model_results_core)
    print(f"\n  >>> Model selected on core set: {model_stage}")

    print("\n" + "="*70)
    print(f"{stage_name} — PART 2: GREEDY FEATURE SELECTION on {model_stage}")
    print("="*70)
    current_groups  = list(core)
    remaining       = list(candidates)
    current_metrics = model_results_core[model_stage]
    print_seeds(f"round 0: {current_groups}", current_metrics)
    selection_log = [{'groups': list(current_groups),
                       'mean_acc': float(np.mean([m['accuracy'] for m in current_metrics])),
                       'accepted': True, 'reason': 'core (always included)'}]

    round_num = 1
    while remaining:
        print(f"\n  -- Round {round_num}: current = {current_groups}, candidates = {remaining} --")
        candidate_results = {}
        for cand in remaining:
            trial_groups = current_groups + [cand]
            Xb_t, Xf_t, y_t, g_t = build_dataset_fn(trial_groups)
            trial_metrics = run_on_model(model_stage, Xb_t, Xf_t, y_t, g_t, labels,
                                          epochs=epochs_selection, n_bilstm_hidden=bilstm_hidden)
            accs_trial = [m['accuracy'] for m in trial_metrics]
            accs_cur   = [m['accuracy'] for m in current_metrics]
            t_stat, p  = stats.ttest_rel(accs_trial, accs_cur)
            gain = np.mean(accs_trial) - np.mean(accs_cur)
            print_seeds(f"+{cand} -> {trial_groups}", trial_metrics)
            print(f"      gain={gain:+.4f}, p={p:.4f} {'(SIG)' if p < ALPHA else '(n.s.)'}")
            candidate_results[cand] = {'groups': trial_groups, 'metrics': trial_metrics,
                                        'gain': gain, 'p': p}

        eligible = {c: r for c, r in candidate_results.items() if r['gain'] > 0 and r['p'] < ALPHA}
        if not eligible:
            print("\n  No candidate gives significant positive improvement — stopping.")
            for c, r in candidate_results.items():
                selection_log.append({'groups': r['groups'],
                                       'mean_acc': float(np.mean([m['accuracy'] for m in r['metrics']])),
                                       'accepted': False,
                                       'reason': f"gain={r['gain']:+.4f}, p={r['p']:.4f} — rejected"})
            break
        winner = max(eligible, key=lambda c: eligible[c]['gain'])
        print(f"\n  >>> ACCEPTED: '{winner}' (gain={eligible[winner]['gain']:+.4f}, "
              f"p={eligible[winner]['p']:.4f})")
        for c, r in candidate_results.items():
            accepted = (c == winner)
            selection_log.append({'groups': r['groups'],
                                   'mean_acc': float(np.mean([m['accuracy'] for m in r['metrics']])),
                                   'accepted': accepted,
                                   'reason': f"gain={r['gain']:+.4f}, p={r['p']:.4f}" +
                                             ('' if accepted else f" — beaten by '{winner}'")})
        current_groups  = candidate_results[winner]['groups']
        current_metrics = candidate_results[winner]['metrics']
        remaining.remove(winner)
        round_num += 1

    final_groups = current_groups
    print(f"\n  >>> FINAL FEATURE SET ({stage_name}, model={model_stage}): {final_groups}")
    Xb_final, Xf_final, y_final, g_final = build_dataset_fn(final_groups)

    print("\n" + "="*70)
    print(f"{stage_name} — PART 2b: RE-VERIFY MODEL on final feature set")
    print("="*70)
    model_results_final = {model_stage: current_metrics}
    for model in MODELS:
        if model == model_stage: continue
        print(f"\n  {model}:")
        metrics = run_on_model(model, Xb_final, Xf_final, y_final, g_final, labels,
                                epochs=epochs_selection, n_bilstm_hidden=bilstm_hidden)
        print_seeds(model, metrics)
        model_results_final[model] = metrics
    best_model = pick_best_model(model_results_final)
    if best_model != model_stage:
        print(f"\n  >>> Model ranking CHANGED: {model_stage} -> {best_model}")
    else:
        print(f"\n  >>> Model choice confirmed: {best_model}")

    return {
        'model_selected_on_core': model_stage,
        'final_model': best_model,
        'final_groups': final_groups,
        'selection_log': selection_log,
        'model_comparison_core': model_results_core,
        'model_comparison_final': model_results_final,
        'Xb': Xb_final, 'Xf': Xf_final, 'y': y_final, 'g': g_final,
    }
