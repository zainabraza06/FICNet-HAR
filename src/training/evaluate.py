import numpy as np
import torch
from scipy.stats import ttest_rel
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

from src.config import DEVICE, CLASSICAL_MODELS, SEEDS, ALPHA, EPOCHS
from src.models.classical import get_classical_models
from src.models.deep import train_bilstm, train_fusion


# ── Per-subject LOSO ──────────────────────────────────────────────────────────

def run_loso_per_subject(model_name, Xb, Xf, y, groups, labels, seed=0, epochs=EPOCHS, hidden=16):
    """
    Leave-One-Subject-Out evaluation that tracks per-subject accuracy.

    Returns
    -------
    all_true     : list[str]  — pooled ground-truth labels
    all_pred     : list[str]  — pooled predicted labels
    per_subject  : dict[int, (correct, total)]
    """
    l2i  = {l: i for i, l in enumerate(labels)}
    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    per_subject: dict = {}

    if model_name in CLASSICAL_MODELS:
        for tr, te in logo.split(Xf, y, groups):
            if len(set(y[tr])) < 2:
                continue
            subj_id = int(groups[te[0]])
            sc  = StandardScaler().fit(Xf[tr])
            Xtr = sc.transform(Xf[tr])
            Xte = sc.transform(Xf[te])
            clf = get_classical_models(seed, len(labels))[model_name]
            clf.fit(Xtr, y[tr])
            pred = clf.predict(Xte)
            correct = int((pred == y[te]).sum())
            per_subject[subj_id] = (correct, len(te))
            all_true.extend(y[te])
            all_pred.extend(pred)

    else:
        y_idx = np.array([l2i[l] for l in y])
        for tr, te in logo.split(Xb, y_idx, groups):
            if len(set(y_idx[tr])) < 2:
                continue
            subj_id = int(groups[te[0]])

            if model_name == 'BiLSTM':
                per_bin_dim = Xb.shape[2]
                sc  = StandardScaler().fit(Xb[tr].reshape(-1, per_bin_dim))
                Xtr = sc.transform(Xb[tr].reshape(-1, per_bin_dim)).reshape(Xb[tr].shape)
                Xte = sc.transform(Xb[te].reshape(-1, per_bin_dim)).reshape(Xb[te].shape)
                model = train_bilstm(Xtr, y_idx[tr], len(labels), per_bin_dim, epochs, hidden, seed)
                model.eval()
                with torch.no_grad():
                    pred = model(
                        torch.tensor(Xte, dtype=torch.float32).to(DEVICE)
                    ).argmax(1).cpu().numpy()

            else:  # Fusion
                per_bin_dim = Xb.shape[2]
                flat_dim    = Xf.shape[1]
                scb = StandardScaler().fit(Xb[tr].reshape(-1, per_bin_dim))
                scf = StandardScaler().fit(Xf[tr])
                Xb_tr = scb.transform(Xb[tr].reshape(-1, per_bin_dim)).reshape(Xb[tr].shape)
                Xb_te = scb.transform(Xb[te].reshape(-1, per_bin_dim)).reshape(Xb[te].shape)
                Xf_tr = scf.transform(Xf[tr])
                Xf_te = scf.transform(Xf[te])
                model = train_fusion(Xb_tr, Xf_tr, y_idx[tr], len(labels),
                                     per_bin_dim, flat_dim, epochs, seed)
                model.eval()
                with torch.no_grad():
                    pred = model(
                        torch.tensor(Xb_te, dtype=torch.float32).to(DEVICE),
                        torch.tensor(Xf_te, dtype=torch.float32).to(DEVICE),
                    ).argmax(1).cpu().numpy()

            pred_labels = [labels[i] for i in pred]
            true_labels = [labels[i] for i in y_idx[te]]
            correct = int(sum(p == t for p, t in zip(pred_labels, true_labels)))
            per_subject[subj_id] = (correct, len(true_labels))
            all_true.extend(true_labels)
            all_pred.extend(pred_labels)

    return all_true, all_pred, per_subject


# ── Statistics helpers ────────────────────────────────────────────────────────

def mean_ci(values):
    """Mean ± SD and 95 % CI for a list of scalar values."""
    values = np.asarray(values)
    n    = len(values)
    mean = float(values.mean())
    sd   = float(values.std(ddof=1)) if n > 1 else 0.0
    se   = sd / np.sqrt(n)           if n > 1 else 0.0
    return {
        'mean': mean, 'sd': sd, 'n': int(n),
        'ci95': (float(mean - 1.96 * se), float(mean + 1.96 * se)),
    }


def paired_ci_and_test(acc_a, acc_b):
    """
    Subject-level paired t-test between two per-subject accuracy dicts.

    Parameters
    ----------
    acc_a, acc_b : dict[int, float]  — {subject_id: accuracy}

    Returns
    -------
    dict with keys: n, mean_diff, t_p
    """
    common = sorted(set(acc_a.keys()) & set(acc_b.keys()))
    a = np.array([acc_a[s] for s in common])
    b = np.array([acc_b[s] for s in common])
    diffs = a - b
    n = len(diffs)
    if n < 2:
        return {'n': n, 'mean_diff': 0.0, 't_p': float('nan')}
    _, p_t = ttest_rel(a, b)
    return {'n': n, 'mean_diff': float(diffs.mean()), 't_p': float(p_t)}


# ── Full greedy selection pipeline ────────────────────────────────────────────

def run_model_selection(model_name, Xb, Xf, y, g, labels, epochs=EPOCHS, hidden=16):
    """Run LOSO over all seeds and return list of (all_true, all_pred, per_subject)."""
    return [
        run_loso_per_subject(model_name, Xb, Xf, y, g, labels, seed=s,
                             epochs=epochs, hidden=hidden)
        for s in SEEDS
    ]


def select_best_model(model_results):
    """Pick the model with the highest mean accuracy across seeds."""
    return max(
        model_results,
        key=lambda m: np.mean([accuracy_score(t, p) for t, p, _ in model_results[m]]),
    )


def greedy_feature_selection(model_name, current_groups, remaining, current_per_seed,
                              codes_for_task, label_fn, labels, build_dataset_fn,
                              epochs=EPOCHS, hidden=16):
    """
    Greedy, subject-level significance-gated feature selection.

    Iteratively adds the candidate feature group with the highest accuracy gain
    that is also statistically significant (subject-level paired t-test, p < ALPHA).

    Returns
    -------
    final_groups     : list[str]
    final_per_seed   : list of (all_true, all_pred, per_subject) tuples
    selection_log    : list[dict]
    """
    selection_log = [{
        'groups': list(current_groups),
        'mean_acc': float(np.mean([accuracy_score(t, p) for t, p, _ in current_per_seed])),
        'accepted': True,
        'reason': 'core (always included)',
    }]

    round_num = 1
    while remaining:
        print(f"\n  -- Round {round_num}: current={current_groups} --")
        current_stat = mean_ci([accuracy_score(t, p) for t, p, _ in current_per_seed])
        # subject-level accuracy from seed 0
        subj_current = {s: c / tot for s, (c, tot) in current_per_seed[0][2].items()}

        candidate_results = {}
        for cand in remaining:
            trial_groups = current_groups + [cand]
            Xb_t, Xf_t, y_t, g_t = build_dataset_fn(trial_groups, codes_for_task, label_fn)
            per_seed = run_model_selection(model_name, Xb_t, Xf_t, y_t, g_t, labels,
                                           epochs=epochs, hidden=hidden)
            stat = mean_ci([accuracy_score(t, p) for t, p, _ in per_seed])
            subj_trial = {s: c / tot for s, (c, tot) in per_seed[0][2].items()}
            subj_test  = paired_ci_and_test(subj_trial, subj_current)
            gain       = stat['mean'] - current_stat['mean']
            print(f"    +{cand}: acc={stat['mean']:.4f}, gain={gain:+.4f}, "
                  f"subj-p={subj_test['t_p']:.4g} (n={subj_test['n']})")
            candidate_results[cand] = {
                'groups': trial_groups, 'per_seed': per_seed,
                'stat': stat, 'gain': gain, 'p_subject': subj_test['t_p'],
            }

        eligible = {
            c: r for c, r in candidate_results.items()
            if r['gain'] > 0 and r['p_subject'] < ALPHA
        }
        if not eligible:
            print("  No candidate significant — stopping.")
            for c, r in candidate_results.items():
                selection_log.append({
                    'groups': r['groups'],
                    'mean_acc': r['stat']['mean'],
                    'accepted': False,
                    'reason': f"gain={r['gain']:+.4f}, p={r['p_subject']:.4g} — rejected",
                })
            break

        winner = max(eligible, key=lambda c: eligible[c]['gain'])
        print(f"  >>> ACCEPTED: {winner}")
        for c, r in candidate_results.items():
            accepted = (c == winner)
            selection_log.append({
                'groups': r['groups'],
                'mean_acc': r['stat']['mean'],
                'accepted': accepted,
                'reason': (f"gain={r['gain']:+.4f}, p={r['p_subject']:.4g}"
                           + ('' if accepted else f" — beaten by '{winner}'")),
            })
        current_groups   = candidate_results[winner]['groups']
        current_per_seed = candidate_results[winner]['per_seed']
        remaining.remove(winner)
        round_num += 1

    return current_groups, current_per_seed, selection_log
