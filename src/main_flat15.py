# ================================================================================
# CELL — FLAT 15-CLASS BASELINE (BSC/FKL/FOL/SDL + 11 ADLs, single classifier)
# Run Cell 1 (common) first. Independent of Cells 2-6 — does not require Stage
# 1/2a/2b to have run, though it reuses their feature registries' candidate
# groups merged into one pool.
#
# Purpose: direct controlled comparison against the hierarchical pipeline.
# Same LOSO protocol, same model pool, same significance-gated feature
# selection procedure — the only difference is a single flat 15-class label
# space instead of the three-stage hierarchical decomposition.
#
# Produces: flat15_result (dict) — final_model, final_groups, Xb/Xf/y/g,
# comparable in structure to stage1_result / stage2a_result / stage2b_result.
# ================================================================================

import os, json, pickle, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

assert 'run_stage_pipeline' in dir(), "Run Cell 1 (common) before this cell."

RESULTS_DIR = f'{RESULTS_ROOT}/flat15_complete'
os.makedirs(RESULTS_DIR, exist_ok=True)

LABELS_FLAT15 = LABELS_ALL_15  # sorted 15-class label space, from common.py


# ================================================================================
# MERGED FEATURE REGISTRY — one flat classifier spanning both fall and ADL
# discrimination needs a feature set covering both problem types. Core groups
# are the ones no stage was willing to drop (profile + stats); candidates pool
# together every group that helped ANY stage (fall_specific from Stage 2a;
# gyro/spectral from Stage 2b; attention/rpe from all three), so the greedy
# selector can pick whichever subset the flat task actually needs — nothing
# is assumed to transfer from the per-stage results.
# ================================================================================
REGISTRY_FLAT15 = {
    'profile': {'fn': lambda acc, gyro=None, roll=None: feat_profile(acc), 'dim': 15, 'core': True,
                'names': [f'energy_{ax}_bin{i}' for ax in 'xyz' for i in range(5)]},
    'stats': {'fn': lambda acc, gyro=None, roll=None: feat_stats5(acc), 'dim': 5, 'core': True,
              'names': ['acc_skew', 'acc_kurtosis', 'acc_var', 'acc_mean', 'max_jerk']},
    'fall_specific': {'fn': lambda acc, gyro=None, roll=None: feat_fall_specific(acc), 'dim': 10, 'core': False,
                       'names': ['peak_ax_x', 'peak_ax_y', 'peak_ax_z', 'n_secondary_peaks', 'max_jerk2',
                                 'time_to_peak', 'settle_time', 'skew2', 'kurt2', 'onset_slope']},
    'gyro': {'fn': lambda acc, gyro, roll: feat_gyro(acc, gyro, roll), 'dim': 8, 'core': False,
             'names': ['gyro_x_energy', 'gyro_y_energy', 'gyro_z_energy',
                       'gyro_skew', 'gyro_kurt', 'gyro_mean', 'roll_range', 'autocorr']},
    'spectral': {'fn': lambda acc, gyro=None, roll=None: feat_spectral(acc), 'dim': 6, 'core': False,
                 'names': ['dom_freq', 'power_conc', 'spec_var', 'spec_mean', 'spec_skew', 'spec_kurt']},
    'attention': {'fn': lambda acc, gyro=None, roll=None: feat_attention(acc), 'dim': 3, 'core': False,
                  'names': ['attn_entropy', 'attn_concentration', 'attn_uniformity']},
    'rpe': {'fn': lambda acc, gyro=None, roll=None: feat_rpe(acc), 'dim': 10, 'core': False,
            'names': RPE_NAMES},
}


def build_dataset_flat15(groups):
    """Builds the 15-class flat dataset. 'gyro' toggles the binned-tensor gyro
    channels consistently, same convention as Stage 2b."""
    include_gyro_bins = 'gyro' in groups
    Xb, Xf, y, g = [], [], [], []
    for code in ALL_CODES:  # FALL_CODES + ADL_CODES_11, from common.py
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None:
                continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values
            roll = seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg))
            fb = build_binned_features(acc, gyro, n_bins=5, include_gyro=include_gyro_bins)
            fc = build_flat_features(acc, gyro, roll, groups, REGISTRY_FLAT15)
            if fb is not None and fc is not None:
                Xb.append(fb)
                Xf.append(fc)
                y.append(code)  # 15-class label directly, no binary gate
                g.append(subj)
    return np.array(Xb), np.array(Xf), np.array(y), np.array(g)


def run_flat15():
    print("\n" + "#" * 70)
    print("# FLAT 15-CLASS BASELINE")
    print("#" * 70)

    run_rpe_diagnostic()

    result = run_stage_pipeline("FLAT 15-CLASS", build_dataset_flat15, REGISTRY_FLAT15, LABELS_FLAT15,
                                 epochs_selection=250, bilstm_hidden=16)

    best_model = result['final_model']
    Xb, Xf, y, g = result['Xb'], result['Xf'], result['y'], result['g']
    feature_names = feature_names_for(result['final_groups'], REGISTRY_FLAT15)

    print("\n" + "=" * 70)
    print("FLAT 15-CLASS — PART 3: PER-CLASS PERFORMANCE (final config)")
    print("=" * 70)
    all_true, all_pred = get_loso_predictions(best_model, Xb, Xf, y, g, LABELS_FLAT15, epochs=250, hidden=16)
    print(f"\nClassification Report ({best_model}, features: {result['final_groups']}):")
    print(classification_report(all_true, all_pred, target_names=LABELS_FLAT15))
    cm = confusion_matrix(all_true, all_pred, labels=LABELS_FLAT15)
    print(pd.DataFrame(cm, index=LABELS_FLAT15, columns=LABELS_FLAT15))

    print("\n" + "=" * 70)
    print("FLAT 15-CLASS — PART 4: VISUALIZATION")
    print("=" * 70)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_FLAT15, yticklabels=LABELS_FLAT15,
                ax=ax, annot_kws={'size': 7})
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Flat 15-Class Confusion Matrix — {best_model}')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/flat15_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {RESULTS_DIR}/flat15_confusion_matrix.png")

    if best_model in CLASSICAL_MODELS:
        sc = StandardScaler().fit(Xf)
        clf_full = get_classical_models(0, len(LABELS_FLAT15))[best_model]
        clf_full.fit(sc.transform(Xf), y)
        result_imp = permutation_importance(clf_full, sc.transform(Xf), y,
                                             n_repeats=30, random_state=42, scoring='accuracy')
        imp_mean = result_imp.importances_mean
        top_n = min(4, len(feature_names))
        top_idx = np.argsort(imp_mean)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_idx]
        print(f"  Top features (permutation importance): {top_features}")
    else:
        print(f"  Best model is {best_model} — skipping permutation-importance plot.")

    print("\n" + "=" * 70)
    print("FLAT 15-CLASS — PART 5: EDGE ANALYSIS")
    print("=" * 70)
    if best_model in CLASSICAL_MODELS:
        sc = StandardScaler().fit(Xf)
        clf = get_classical_models(0, len(LABELS_FLAT15))[best_model]
        clf.fit(sc.transform(Xf), y)
        Xf_s = sc.transform(Xf)
        for _ in range(50):
            _ = clf.predict(Xf_s[:100])
        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            _ = clf.predict(Xf_s[:100])
            times.append((time.perf_counter() - t0) * 1000)
        size_mb = len(pickle.dumps(clf)) / (1024 * 1024)
    else:
        l2i = {l: i for i, l in enumerate(LABELS_FLAT15)}
        y_idx = np.array([l2i[l] for l in y])
        per_bin_dim = Xb.shape[2]
        if best_model == 'BiLSTM':
            sc = StandardScaler().fit(Xb.reshape(-1, per_bin_dim))
            Xb_s = sc.transform(Xb.reshape(-1, per_bin_dim)).reshape(Xb.shape)
            model = train_bilstm(Xb_s, y_idx, len(LABELS_FLAT15), per_bin_dim, 250, 16, 0)
            model.eval()
            Xb_t = torch.tensor(Xb_s[:100], dtype=torch.float32).to(device)
            for _ in range(50):
                with torch.no_grad():
                    _ = model(Xb_t)
            times = []
            for _ in range(200):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(Xb_t)
                times.append((time.perf_counter() - t0) * 1000)
            size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # 4*P, not 4*2*P
        else:
            flat_dim = Xf.shape[1]
            scb = StandardScaler().fit(Xb.reshape(-1, per_bin_dim))
            scf = StandardScaler().fit(Xf)
            Xb_s = scb.transform(Xb.reshape(-1, per_bin_dim)).reshape(Xb.shape)
            Xf_s = scf.transform(Xf)
            model = train_fusion(Xb_s, Xf_s, y_idx, len(LABELS_FLAT15), per_bin_dim, flat_dim, 250, 0)
            model.eval()
            Xb_t = torch.tensor(Xb_s[:100], dtype=torch.float32).to(device)
            Xf_t = torch.tensor(Xf_s[:100], dtype=torch.float32).to(device)
            for _ in range(50):
                with torch.no_grad():
                    _ = model(Xb_t, Xf_t)
            times = []
            for _ in range(200):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(Xb_t, Xf_t)
                times.append((time.perf_counter() - t0) * 1000)
            size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)  # 4*P, not 4*2*P

    time_ms, time_std = np.mean(times), np.std(times)
    print(f"    Inference: {time_ms:.3f} ± {time_std:.3f} ms   Size: {size_mb:.3f} MB")

    # Macro F1 / balanced accuracy from the pooled classification report
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(all_true, all_pred)
    bal_acc = balanced_accuracy_score(all_true, all_pred)
    _, _, f1, _ = precision_recall_fscore_support(all_true, all_pred, labels=LABELS_FLAT15,
                                                    average='macro', zero_division=0)

    results = {
        'model_selected_on_core': result['model_selected_on_core'],
        'final_model': best_model,
        'final_feature_groups': result['final_groups'],
        'feature_selection_log': result['selection_log'],
        'accuracy': float(acc),
        'balanced_accuracy': float(bal_acc),
        'macro_f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'edge_metrics': {'inference_time_ms': float(time_ms), 'model_size_mb': float(size_mb)},
    }
    with open(f'{RESULTS_DIR}/flat15_final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("FLAT 15-CLASS BASELINE COMPLETE")
    print(f"  Final model: {best_model}")
    print(f"  Final feature groups: {result['final_groups']}")
    print(f"  Accuracy: {acc:.4f}  Balanced Acc: {bal_acc:.4f}  Macro F1: {f1:.4f}")
    print(f"{'='*70}")

    return {**result, 'accuracy': acc, 'balanced_accuracy': bal_acc, 'macro_f1': f1,
            'cm': cm, 'edge_metrics': results['edge_metrics']}


flat15_result = run_flat15()
