import os, json
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.config import RESULTS_DIR_STREAMING, ADL_CODES_11, ALPHA, DEVICE
from src.features.extractors import REGISTRY_S2B, build_binned_features, build_flat_features
from src.data.loader import get_segment
from src.models.deep import train_streaming_fusion

os.makedirs(RESULTS_DIR_STREAMING, exist_ok=True)

WINDOW_SAMPLES, STRIDE, SUB_BINS, PURITY_THRESHOLD = 200, 100, 5, 0.9
STREAMING_SEEDS = [0, 1, 2]

def build_subject_stream(codes, subject, max_per_activity=1000):
    acc_all, gyro_all, roll_all, labels_all = [], [], [], []
    for code in codes:
        seg = get_segment(code, subject)
        if seg is None:
            continue
        seg = seg.iloc[:max_per_activity]
        acc_all.append(seg[['acc_x', 'acc_y', 'acc_z']].values)
        if 'gyro_x' in seg.columns:
            gyro_all.append(seg[['gyro_x', 'gyro_y', 'gyro_z']].values)
        else:
            gyro_all.append(np.zeros((len(seg), 3)))
        roll_all.append(seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg)))
        labels_all.extend([code] * len(seg))
    if not acc_all:
        return None
    return np.vstack(acc_all), np.vstack(gyro_all), np.concatenate(roll_all), np.array(labels_all)

def build_streaming_flat_features(acc_w, gyro_w, roll_w, include_attention=False, include_rpe=False):
    groups = ['profile', 'stats', 'spectral', 'gyro']
    if include_attention:
        groups.append('attention')
    if include_rpe:
        groups.append('rpe')
    return build_flat_features(acc_w, gyro_w, roll_w, groups, REGISTRY_S2B)

def window_stream(acc_s, gyro_s, roll_s, labels_s, include_attention=False, include_rpe=False):
    n = len(acc_s)
    Xb, Xf, y, purities = [], [], [], []
    start = 0
    while start + WINDOW_SAMPLES <= n:
        end = start + WINDOW_SAMPLES
        acc_w, gyro_w, roll_w = acc_s[start:end], gyro_s[start:end], roll_s[start:end]
        labels_w = labels_s[start:end]
        vals, counts = np.unique(labels_w, return_counts=True)
        maj = vals[np.argmax(counts)]
        purity = counts.max() / len(labels_w)
        fb = build_binned_features(acc_w, gyro_w, SUB_BINS, include_gyro=True)
        fc = build_streaming_flat_features(acc_w, gyro_w, roll_w, include_attention, include_rpe)
        if fb is not None and fc is not None:
            Xb.append(fb); Xf.append(fc); y.append(maj); purities.append(purity)
        start += STRIDE
    if not Xb:
        return None, None, None, None
    return np.array(Xb), np.array(Xf), np.array(y), np.array(purities)

def build_streaming_dataset(codes, subjects, include_attention=False, include_rpe=False):
    data = {}
    for subj in subjects:
        stream = build_subject_stream(codes, subj)
        if stream is None:
            continue
        acc_s, gyro_s, roll_s, labels_s = stream
        Xb, Xf, yw, purity = window_stream(acc_s, gyro_s, roll_s, labels_s, include_attention, include_rpe)
        if Xb is not None and len(Xb) > 5:
            data[subj] = (Xb, Xf, yw, purity)
    return data

def compute_purity_metrics(y_true, y_pred, purity, thresh=PURITY_THRESHOLD):
    y_true, y_pred, purity = np.array(y_true), np.array(y_pred), np.array(purity)
    overall_acc = accuracy_score(y_true, y_pred)
    overall_bal = balanced_accuracy_score(y_true, y_pred)
    pm = purity >= thresh
    pure_acc = accuracy_score(y_true[pm], y_pred[pm]) if pm.sum() > 0 else None
    bm = ~pm
    b_acc = accuracy_score(y_true[bm], y_pred[bm]) if bm.sum() > 0 else None
    return {'overall_accuracy': overall_acc, 'overall_balanced_accuracy': overall_bal,
            'pure_window_accuracy': pure_acc, 'pure_window_fraction': pm.mean(),
            'boundary_window_accuracy': b_acc, 'boundary_window_fraction': bm.mean()}

def run_streaming_one_seed(data, l2i, n_classes, per_bin_dim, flat_dim, epochs=50, seed=0):
    subj_ids = list(data.keys())
    all_true, all_pred, all_purity = [], [], []
    per_subject_acc = {}
    for fold_i, test_subj in enumerate(subj_ids):
        train_data = {s: data[s] for s in subj_ids if s != test_subj}
        if len(train_data) < 2:
            continue
        model = train_streaming_fusion(train_data, l2i, n_classes, per_bin_dim, flat_dim, epochs, seed)
        model.eval()
        with torch.no_grad():
            Xb, Xf, yw, purity = data[test_subj]
            y_idx = np.array([l2i[l] for l in yw])
            Xb_t = torch.tensor(Xb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            Xf_t = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred = model(Xb_t, Xf_t).argmax(1).cpu().numpy()
        all_true.extend(y_idx); all_pred.extend(pred); all_purity.extend(purity)
        per_subject_acc[test_subj] = accuracy_score(y_idx, pred)
        if fold_i % 10 == 0:
            print(f"      [seed {seed}] fold {fold_i}/{len(subj_ids)}")
    m = compute_purity_metrics(all_true, all_pred, all_purity)
    m['_per_subject_acc'] = per_subject_acc
    return m

def run_naive_classical(data, l2i, n_classes, seed=0):
    subj_ids = list(data.keys())
    all_true, all_pred, all_purity = [], [], []
    per_subject_acc = {}
    for fold_i, test_subj in enumerate(subj_ids):
        Xf_tr, y_tr = [], []
        for s in subj_ids:
            if s == test_subj:
                continue
            Xb, Xf, yw, p = data[s]
            Xf_tr.append(Xf)
            y_tr.extend([l2i[l] for l in yw])
        Xf_tr = np.vstack(Xf_tr); y_tr = np.array(y_tr)
        Xb_te, Xf_te, yw_te, p_te = data[test_subj]
        y_te = np.array([l2i[l] for l in yw_te])
        sc = StandardScaler().fit(Xf_tr)
        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed)
        clf.fit(sc.transform(Xf_tr), y_tr)
        pred = clf.predict(sc.transform(Xf_te))
        all_true.extend(y_te); all_pred.extend(pred); all_purity.extend(p_te)
        per_subject_acc[test_subj] = accuracy_score(y_te, pred)
        if fold_i % 10 == 0:
            print(f"      fold {fold_i}/{len(subj_ids)}")
    m = compute_purity_metrics(all_true, all_pred, all_purity)
    m['_per_subject_acc'] = per_subject_acc
    return m

def paired_subject_acc_per_seed(per_seed_list):
    return [np.mean(list(m['_per_subject_acc'].values())) for m in per_seed_list]

def main():
    print("\n" + "#" * 70)
    print("# STREAMING EVALUATION")
    print("#" * 70)

    feature_variants = {
        'Naive Classical': {'attention': False, 'rpe': False, 'is_naive': True},
        'Baseline (Fusion)': {'attention': False, 'rpe': False, 'is_naive': False},
        '+ Temporal Attention': {'attention': True, 'rpe': False, 'is_naive': False},
        '+ RPE (fixed)': {'attention': False, 'rpe': True, 'is_naive': False},
        '+ Attention + RPE (fixed)': {'attention': True, 'rpe': True, 'is_naive': False},
    }

    label_list = ADL_CODES_11
    l2i = {l: i for i, l in enumerate(label_list)}
    n_classes = len(label_list)

    print("\nBuilding baseline dataset...")
    baseline_data = build_streaming_dataset(ADL_CODES_11, list(range(1, 68)))
    sample = next(iter(baseline_data))
    per_bin_dim = baseline_data[sample][0].shape[2]
    flat_dim = baseline_data[sample][1].shape[1]
    print(f"  Subjects: {len(baseline_data)}  per_bin_dim: {per_bin_dim}  flat_dim: {flat_dim}  "
          f"seeds: {STREAMING_SEEDS}")

    all_results = {}
    for variant_name, params in feature_variants.items():
        print(f"\n{'='*70}\nBuilding dataset for: {variant_name}\n{'='*70}")
        data = build_streaming_dataset(ADL_CODES_11, list(range(1, 68)),
                                        include_attention=params['attention'], include_rpe=params['rpe'])
        sample = next(iter(data))
        vflat_dim = data[sample][1].shape[1]

        per_seed = []
        for seed in STREAMING_SEEDS:
            if params.get('is_naive'):
                m = run_naive_classical(data, l2i, n_classes, seed=seed)
            else:
                m = run_streaming_one_seed(data, l2i, n_classes, per_bin_dim, vflat_dim, epochs=50, seed=seed)
            per_seed.append(m)
            print(f"    seed {seed}: overall_acc={m['overall_accuracy']:.4f}")

        agg = {
            'overall_accuracy': float(np.mean([m['overall_accuracy'] for m in per_seed])),
            'overall_accuracy_std': float(np.std([m['overall_accuracy'] for m in per_seed])),
            'pure_window_accuracy': float(np.mean([m['pure_window_accuracy'] for m in per_seed
                                                     if m['pure_window_accuracy'] is not None])),
            'boundary_window_accuracy': float(np.mean([m['boundary_window_accuracy'] for m in per_seed
                                                         if m['boundary_window_accuracy'] is not None])),
            '_per_seed': per_seed,
        }
        print(f"  {variant_name}: acc={agg['overall_accuracy']:.4f}±{agg['overall_accuracy_std']:.4f} "
              f"pure={agg['pure_window_accuracy']:.4f} boundary={agg['boundary_window_accuracy']:.4f}")
        all_results[variant_name] = agg

    print(f"\n{'='*70}\nSIGNIFICANCE vs Baseline (Fusion)\n{'='*70}")
    baseline_paired = paired_subject_acc_per_seed(all_results['Baseline (Fusion)']['_per_seed'])
    sig_results = {}
    for variant_name, agg in all_results.items():
        if variant_name == 'Baseline (Fusion)':
            continue
        variant_paired = paired_subject_acc_per_seed(agg['_per_seed'])
        if len(variant_paired) == len(baseline_paired) and len(variant_paired) >= 2:
            gain = np.mean(variant_paired) - np.mean(baseline_paired)
            t_stat, p = ttest_rel(variant_paired, baseline_paired)
            sig = p < ALPHA and gain > 0
            print(f"  {variant_name}: gain={gain:+.4f}, p={p:.4f} {'(SIG)' if sig else '(n.s.)'}")
            sig_results[variant_name] = {'gain': float(gain), 'p': float(p), 'significant': bool(sig)}
        else:
            print(f"  {variant_name}: insufficient paired seeds")
            sig_results[variant_name] = {'gain': None, 'p': None, 'significant': False}

    rows = [{'variant': v, 'overall_acc': r['overall_accuracy'], 'overall_acc_std': r['overall_accuracy_std'],
             'pure_acc': r['pure_window_accuracy'], 'boundary_acc': r['boundary_window_accuracy'],
             'p_vs_baseline': sig_results.get(v, {}).get('p'), 'significant': sig_results.get(v, {}).get('significant')}
            for v, r in all_results.items()]
    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR_STREAMING}/streaming_results.csv', index=False)
    print(f"\nSaved: {RESULTS_DIR_STREAMING}/streaming_results.csv")

    def clean_for_json(d):
        return {k: v for k, v in d.items() if not k.startswith('_')}
    with open(f'{RESULTS_DIR_STREAMING}/streaming_results.json', 'w') as f:
        json.dump({k: clean_for_json(v) for k, v in all_results.items()}, f, indent=2, default=str)
    with open(f'{RESULTS_DIR_STREAMING}/streaming_significance.json', 'w') as f:
        json.dump(sig_results, f, indent=2, default=str)

    print("\n" + "#" * 70)
    print("STREAMING EVALUATION COMPLETE")
    print("#" * 70)

if __name__ == '__main__':
    main()
