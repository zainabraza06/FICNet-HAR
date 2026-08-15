import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

from src.config import ADL_CODES_11, RESULTS_DIR_STREAMING, SEEDS, DEVICE
from src.data.streaming import build_streaming_dataset
from src.models.deep import train_streaming_fusion

PURITY_THRESHOLD = 0.9
ALPHA = 0.05

def compute_purity_metrics(y_true, y_pred, purity, thresh=PURITY_THRESHOLD):
    y_true, y_pred, purity = np.array(y_true), np.array(y_pred), np.array(purity)

    overall_acc = accuracy_score(y_true, y_pred)
    overall_bal = balanced_accuracy_score(y_true, y_pred)

    pm = purity >= thresh
    pure_acc = accuracy_score(y_true[pm], y_pred[pm]) if pm.sum() > 0 else None
    pure_bal = balanced_accuracy_score(y_true[pm], y_pred[pm]) if pm.sum() > 0 else None

    bm = ~pm
    b_acc = accuracy_score(y_true[bm], y_pred[bm]) if bm.sum() > 0 else None
    b_bal = balanced_accuracy_score(y_true[bm], y_pred[bm]) if bm.sum() > 0 else None

    return {
        'overall_accuracy': overall_acc,
        'overall_balanced_accuracy': overall_bal,
        'pure_window_accuracy': pure_acc,
        'pure_window_balanced_accuracy': pure_bal,
        'pure_window_fraction': pm.mean(),
        'boundary_window_accuracy': b_acc,
        'boundary_window_balanced_accuracy': b_bal,
        'boundary_window_fraction': bm.mean(),
    }

def print_streaming_results(name, metrics):
    print(f"\n  {name}:")
    print(f"    Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"    Overall Balanced Acc: {metrics['overall_balanced_accuracy']:.4f}")
    print(f"    Pure Windows ({metrics['pure_window_fraction']:.1%}): {metrics['pure_window_accuracy']:.4f}")
    print(f"    Boundary Windows ({metrics['boundary_window_fraction']:.1%}): {metrics['boundary_window_accuracy']:.4f}")
    gap = metrics['pure_window_accuracy'] - metrics['boundary_window_accuracy'] if metrics['pure_window_accuracy'] is not None and metrics['boundary_window_accuracy'] is not None else None
    if gap is not None:
        print(f"    Performance Gap: {gap:.4f} ({gap*100:.1f}%)")

def run_naive_classical(data, l2i, n_classes, seed=0):
    print("\n  Testing: Naive Classical (No Memory) [RandomForest]")
    subj_ids = list(data.keys())
    all_true, all_pred, all_purity = [], [], []
    per_subject_acc = {}

    for fold_i, test_subj in enumerate(subj_ids):
        Xf_tr, y_tr = [], []
        for s in subj_ids:
            if s == test_subj: continue
            Xb, Xf, yw, p = data[s]
            Xf_tr.append(Xf)
            y_tr.extend([l2i[l] for l in yw])

        Xf_tr = np.vstack(Xf_tr)
        y_tr = np.array(y_tr)
        Xb_te, Xf_te, yw_te, p_te = data[test_subj]
        y_te = np.array([l2i[l] for l in yw_te])

        sc = StandardScaler().fit(Xf_tr)
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed)
        clf.fit(sc.transform(Xf_tr), y_tr)
        pred = clf.predict(sc.transform(Xf_te))

        all_true.extend(y_te)
        all_pred.extend(pred)
        all_purity.extend(p_te)
        per_subject_acc[test_subj] = accuracy_score(y_te, pred)

        if fold_i % 10 == 0:
            print(f"    fold {fold_i}/{len(subj_ids)} complete")

    metrics = compute_purity_metrics(all_true, all_pred, all_purity)
    metrics['_per_subject_acc'] = per_subject_acc
    return metrics

def run_streaming_evaluation_one_seed(data, l2i, n_classes, per_bin_dim, flat_dim, variant_name, epochs=50, seed=0):
    subj_ids = list(data.keys())
    all_true, all_pred, all_purity = [], [], []
    per_subject_acc = {}

    for fold_i, test_subj in enumerate(subj_ids):
        train_data = {s: data[s] for s in subj_ids if s != test_subj}
        if len(train_data) < 2: continue

        model = train_streaming_fusion(train_data, l2i, n_classes, per_bin_dim, flat_dim, epochs, seed)
        model.eval()

        with torch.no_grad():
            Xb, Xf, yw, purity = data[test_subj]
            y_idx = np.array([l2i[l] for l in yw])
            Xb_t = torch.tensor(Xb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            logits = model(Xb_t, Xf_t)
            pred = logits.argmax(dim=1).cpu().numpy()

        all_true.extend(y_idx)
        all_pred.extend(pred)
        all_purity.extend(purity)
        per_subject_acc[test_subj] = accuracy_score(y_idx, pred)

        if fold_i % 10 == 0:
            print(f"      [seed {seed}] fold {fold_i}/{len(subj_ids)} complete")

    metrics = compute_purity_metrics(all_true, all_pred, all_purity)
    metrics['_per_subject_acc'] = per_subject_acc
    return metrics

def run_streaming_evaluation_multiseed(data, l2i, n_classes, per_bin_dim, flat_dim, variant_name, epochs=50, seeds=SEEDS):
    print(f"\n  Testing: {variant_name}")
    per_seed_metrics = []
    for seed in seeds:
        m = run_streaming_evaluation_one_seed(data, l2i, n_classes, per_bin_dim, flat_dim, variant_name, epochs, seed)
        per_seed_metrics.append(m)
        print(f"    seed {seed}: overall_acc={m['overall_accuracy']:.4f}")

    agg = {
        'overall_accuracy': float(np.mean([m['overall_accuracy'] for m in per_seed_metrics])),
        'overall_accuracy_std': float(np.std([m['overall_accuracy'] for m in per_seed_metrics])),
        'overall_balanced_accuracy': float(np.mean([m['overall_balanced_accuracy'] for m in per_seed_metrics])),
        'pure_window_accuracy': float(np.mean([m['pure_window_accuracy'] for m in per_seed_metrics if m['pure_window_accuracy'] is not None])),
        'pure_window_fraction': float(np.mean([m['pure_window_fraction'] for m in per_seed_metrics])),
        'boundary_window_accuracy': float(np.mean([m['boundary_window_accuracy'] for m in per_seed_metrics if m['boundary_window_accuracy'] is not None])),
        'boundary_window_fraction': float(np.mean([m['boundary_window_fraction'] for m in per_seed_metrics])),
        '_per_seed_metrics': per_seed_metrics,
    }
    print_streaming_results(variant_name, agg)
    return agg

def paired_subject_acc_per_seed(metrics):
    if '_per_seed_metrics' in metrics:
        return [np.mean(list(m['_per_subject_acc'].values())) for m in metrics['_per_seed_metrics']]
    elif '_per_subject_acc' in metrics:
        return [np.mean(list(metrics['_per_subject_acc'].values()))]
    return None

def main():
    print("\n" + "="*70)
    print("STREAMING EVALUATION")
    print("="*70)

    feature_variants = {
        'Naive Classical': {'attention': False, 'rpe': False, 'is_naive': True},
        'Baseline (Fusion)': {'attention': False, 'rpe': False, 'is_naive': False},
        '+ Temporal Attention': {'attention': True, 'rpe': False, 'is_naive': False},
        '+ RPE': {'attention': False, 'rpe': True, 'is_naive': False},
        '+ Attention + RPE': {'attention': True, 'rpe': True, 'is_naive': False},
    }

    print("\nBuilding baseline dataset...")
    baseline_data = build_streaming_dataset(
        ADL_CODES_11, list(range(1, 68)),
        include_attention=False, include_rpe=False
    )

    sample = next(iter(baseline_data))
    per_bin_dim = baseline_data[sample][0].shape[2]
    flat_dim = baseline_data[sample][1].shape[1]
    label_list = ADL_CODES_11
    l2i = {l: i for i, l in enumerate(label_list)}
    n_classes = len(label_list)

    print(f"  Subjects: {len(baseline_data)}")
    print(f"  per_bin_dim: {per_bin_dim}")
    print(f"  flat_dim: {flat_dim}")
    print(f"  Seeds per variant: {SEEDS}")

    all_results = {}

    for variant_name, params in feature_variants.items():
        print(f"\n{'='*70}")
        print(f"Building dataset for: {variant_name}")
        print(f"{'='*70}")

        data = build_streaming_dataset(
            ADL_CODES_11, list(range(1, 68)),
            include_attention=params['attention'],
            include_rpe=params['rpe']
        )
        print(f"  Subjects: {len(data)}")

        if params.get('is_naive', False):
            per_seed = []
            for seed in SEEDS:
                per_seed.append(run_naive_classical(data, l2i, n_classes, seed=seed))
            metrics = {
                'overall_accuracy': float(np.mean([m['overall_accuracy'] for m in per_seed])),
                'overall_accuracy_std': float(np.std([m['overall_accuracy'] for m in per_seed])),
                'overall_balanced_accuracy': float(np.mean([m['overall_balanced_accuracy'] for m in per_seed])),
                'pure_window_accuracy': float(np.mean([m['pure_window_accuracy'] for m in per_seed if m['pure_window_accuracy'] is not None])),
                'pure_window_fraction': float(np.mean([m['pure_window_fraction'] for m in per_seed])),
                'boundary_window_accuracy': float(np.mean([m['boundary_window_accuracy'] for m in per_seed if m['boundary_window_accuracy'] is not None])),
                'boundary_window_fraction': float(np.mean([m['boundary_window_fraction'] for m in per_seed])),
                '_per_seed_metrics': per_seed,
            }
            print_streaming_results(variant_name, metrics)
        else:
            sample = next(iter(data))
            new_flat_dim = data[sample][1].shape[1]
            variant_flat_dim = new_flat_dim if new_flat_dim != flat_dim else flat_dim
            if new_flat_dim != flat_dim:
                print(f"  flat_dim for this variant: {flat_dim} -> {new_flat_dim}")

            metrics = run_streaming_evaluation_multiseed(
                data, l2i, n_classes, per_bin_dim, variant_flat_dim, variant_name, epochs=50, seeds=SEEDS
            )

        all_results[variant_name] = metrics

    print("\n" + "="*70)
    print("SIGNIFICANCE TESTING vs Baseline (Fusion)")
    print("="*70)

    baseline_paired = paired_subject_acc_per_seed(all_results['Baseline (Fusion)'])
    significance_results = {}
    for variant_name, metrics in all_results.items():
        if variant_name == 'Baseline (Fusion)':
            continue
        variant_paired = paired_subject_acc_per_seed(metrics)
        if variant_paired is None or baseline_paired is None or len(variant_paired) != len(baseline_paired) or len(variant_paired) < 2:
            print(f"  {variant_name}: insufficient paired seeds for a significance test (need >=2 matching seeds)")
            significance_results[variant_name] = {'p': None, 'gain': None, 'significant': False}
            continue
        gain = np.mean(variant_paired) - np.mean(baseline_paired)
        t_stat, p = ttest_rel(variant_paired, baseline_paired)
        sig = p < ALPHA and gain > 0
        print(f"  {variant_name}: gain={gain:+.4f}, p={p:.4f} {'(SIG improvement)' if sig else '(not sig / not improvement)'}")
        significance_results[variant_name] = {'p': float(p), 'gain': float(gain), 'significant': bool(sig)}

    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    rows = []
    for variant_name, metrics in all_results.items():
        rows.append({
            'variant': variant_name,
            'overall_accuracy': metrics['overall_accuracy'],
            'overall_accuracy_std': metrics.get('overall_accuracy_std'),
            'overall_balanced_accuracy': metrics['overall_balanced_accuracy'],
            'pure_window_accuracy': metrics['pure_window_accuracy'],
            'pure_window_fraction': metrics['pure_window_fraction'],
            'boundary_window_accuracy': metrics['boundary_window_accuracy'],
            'boundary_window_fraction': metrics['boundary_window_fraction'],
            'significant_vs_baseline': significance_results.get(variant_name, {}).get('significant'),
            'p_vs_baseline': significance_results.get(variant_name, {}).get('p'),
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR_STREAMING}/streaming_results.csv', index=False)
    print(f"  Saved: {RESULTS_DIR_STREAMING}/streaming_results.csv")

    def clean_for_json(d):
        out = {}
        for k, v in d.items():
            if k.startswith('_'):
                continue
            out[k] = v
        return out

    with open(f'{RESULTS_DIR_STREAMING}/streaming_results.json', 'w') as f:
        json.dump({k: clean_for_json(v) for k, v in all_results.items()}, f, indent=2, default=str)
    with open(f'{RESULTS_DIR_STREAMING}/streaming_significance.json', 'w') as f:
        json.dump(significance_results, f, indent=2, default=str)

    print("\n" + "="*70)
    print("STREAMING EVALUATION — FINAL SUMMARY")
    print("="*70)

    print(f"\n{'Variant':<25} {'Overall Acc':<15} {'Pure Acc':<12} {'Boundary Acc':<14} {'vs Baseline':<20}")
    print(f"{'-'*90}")

    for variant_name, metrics in all_results.items():
        acc = metrics['overall_accuracy']
        acc_std = metrics.get('overall_accuracy_std', 0)
        pure = metrics['pure_window_accuracy']
        boundary = metrics['boundary_window_accuracy']
        pure_str = f"{pure:.4f}" if pure is not None else "N/A"
        boundary_str = f"{boundary:.4f}" if boundary is not None else "N/A"

        if variant_name == 'Baseline (Fusion)':
            vs_str = "(reference)"
        else:
            sig_info = significance_results.get(variant_name, {})
            if sig_info.get('p') is None:
                vs_str = "n/a"
            else:
                vs_str = f"p={sig_info['p']:.3f} {'SIG' if sig_info['significant'] else 'n.s.'}"

        print(f"{variant_name:<25} {acc:.4f}±{acc_std:.4f}  {pure_str:<12} {boundary_str:<14} {vs_str:<20}")

    sig_winners = [v for v, r in significance_results.items() if r.get('significant')]
    if sig_winners:
        best_sig = max(sig_winners, key=lambda v: significance_results[v]['gain'])
        print(f"\n  Best variant with a SIGNIFICANT improvement over Baseline: {best_sig} "
              f"(gain={significance_results[best_sig]['gain']:+.4f}, p={significance_results[best_sig]['p']:.4f})")
    else:
        print(f"\n  No variant showed a statistically significant improvement over Baseline (Fusion) "
              f"across {len(SEEDS)} seed(s) — Baseline remains the recommended choice pending more seeds/data.")

    print("\n" + "="*70)
    print("STREAMING EVALUATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
