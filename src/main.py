import os, json, pickle, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from src.config import RESULTS_DIR, LABELS_S1, ALL_CODES, FALL_CODES, CLASSICAL_MODELS
from src.features.extractors import REGISTRY_S1, build_binned_features, build_flat_features, feature_names_for
from src.data.loader import get_segment
from src.training.evaluate import run_stage_pipeline, get_loso_predictions
from src.models.classical import get_classical_models
from src.models.deep import train_bilstm, train_fusion
from src.config import DEVICE

def build_dataset_stage1(groups):
    Xb, Xf, y, g = [], [], [], []
    for code in ALL_CODES:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None:
                continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x', 'acc_y', 'acc_z']].values
            fb = build_binned_features(acc, n_bins=5, include_gyro=False)
            fc = build_flat_features(acc, None, None, groups, REGISTRY_S1)
            if fb is not None and fc is not None:
                Xb.append(fb)
                Xf.append(fc)
                y.append('FALL' if code in FALL_CODES else 'ADL')
                g.append(subj)
    return np.array(Xb), np.array(Xf), np.array(y), np.array(g)

def run_rpe_diagnostic():
    from src.features.extractors import feat_rpe
    print("\n" + "=" * 70)
    print("RPE DIAGNOSTIC (post-fix)")
    print("=" * 70)
    codes = ALL_CODES[:3]
    sample_segs = []
    for code in codes:
        for subj in range(1, 15 + 1):
            seg = get_segment(code, subj)
            if seg is None:
                continue
            acc = seg[['acc_x', 'acc_y', 'acc_z']].values[:1000]
            if len(acc) < 20:
                continue
            sample_segs.append(feat_rpe(acc))
    if not sample_segs:
        print("  No segments available for diagnostic sample — skipping.")
        return
    rpe_block = np.array(sample_segs)
    n_nan, n_inf = np.isnan(rpe_block).sum(), np.isinf(rpe_block).sum()
    max_val = np.nanmax(np.abs(rpe_block))
    print(f"  Sample size: {len(sample_segs)} segments")
    print(f"  NaN: {n_nan}  Inf: {n_inf}  Max |value|: {max_val:.4f}  (clip ceiling = 100.0)")
    if n_nan == 0 and n_inf == 0 and max_val <= 100.0:
        print("  OK: fixed feat_rpe() is bounded as expected. Proceeding.")
    else:
        print("  WARNING: unexpected values still present — investigate before proceeding.")

def main():
    print("\n" + "#" * 70)
    print("# STAGE 1")
    print("#" * 70)

    run_rpe_diagnostic()

    result = run_stage_pipeline("STAGE 1", build_dataset_stage1, REGISTRY_S1, LABELS_S1,
                                 epochs_selection=200, bilstm_hidden=8)

    best_model = result['final_model']
    Xb, Xf, y, g = result['Xb'], result['Xf'], result['y'], result['g']
    feature_names = feature_names_for(result['final_groups'], REGISTRY_S1)

    print("\n" + "=" * 70)
    print("STAGE 1 — PART 3: PER-CLASS PERFORMANCE (final config)")
    print("=" * 70)
    all_true, all_pred = get_loso_predictions(best_model, Xb, Xf, y, g, LABELS_S1, epochs=200, hidden=8)
    print(f"\nClassification Report ({best_model}, features: {result['final_groups']}):")
    print(classification_report(all_true, all_pred, target_names=LABELS_S1))
    cm = confusion_matrix(all_true, all_pred, labels=LABELS_S1)
    print(pd.DataFrame(cm, index=LABELS_S1, columns=LABELS_S1))

    print("\n" + "=" * 70)
    print("STAGE 1 — PART 4: VISUALIZATION")
    print("=" * 70)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_S1, yticklabels=LABELS_S1,
                ax=ax, cbar=False, linewidths=0.5, linecolor='white')
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Stage 1 Confusion Matrix — {best_model}')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/stage1_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/stage1_confusion_matrix.png")

    if best_model in CLASSICAL_MODELS:
        sc = StandardScaler().fit(Xf)
        clf_full = get_classical_models(0, len(LABELS_S1))[best_model]
        clf_full.fit(sc.transform(Xf), y)
        result_imp = permutation_importance(clf_full, sc.transform(Xf), y,
                                             n_repeats=30, random_state=42, scoring='accuracy')
        imp_mean = result_imp.importances_mean
        top_n = min(4, len(feature_names))
        top_idx = np.argsort(imp_mean)[-top_n:][::-1]
        top_features = [feature_names[i] for i in top_idx]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        for i, (idx, feat) in enumerate(zip(top_idx, top_features)):
            ax = axes[i]
            for label in LABELS_S1:
                mask = y == label
                ax.hist(Xf[mask, idx], bins=30, alpha=0.5, label=label, density=True)
            ax.set_xlabel(feat); ax.set_ylabel('Density'); ax.legend()
            ax.set_title(f'Top {i+1} Feature: {feat}')
        plt.suptitle(f'Stage 1 — Top Discriminative Features ({best_model})')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/stage1_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {RESULTS_DIR}/stage1_feature_distributions.png")
    else:
        print(f"  Best model is {best_model} — skipping permutation-importance plot.")

    print("\n" + "=" * 70)
    print("STAGE 1 — PART 5: EDGE ANALYSIS")
    print("=" * 70)
    if best_model in CLASSICAL_MODELS:
        sc = StandardScaler().fit(Xf)
        clf = get_classical_models(0, len(LABELS_S1))[best_model]
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
        import torch
        l2i = {l: i for i, l in enumerate(LABELS_S1)}
        y_idx = np.array([l2i[l] for l in y])
        per_bin_dim = Xb.shape[2]
        if best_model == 'BiLSTM':
            sc = StandardScaler().fit(Xb.reshape(-1, per_bin_dim))
            Xb_s = sc.transform(Xb.reshape(-1, per_bin_dim)).reshape(Xb.shape)
            model = train_bilstm(Xb_s, y_idx, len(LABELS_S1), per_bin_dim, 200, 8, 0)
            model.eval()
            Xb_t = torch.tensor(Xb_s[:100], dtype=torch.float32).to(DEVICE)
            for _ in range(50):
                with torch.no_grad():
                    _ = model(Xb_t)
            times = []
            for _ in range(200):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(Xb_t)
                times.append((time.perf_counter() - t0) * 1000)
            size_mb = sum(p.numel() for p in model.parameters()) * 4 * 2 / (1024 * 1024)
        else:
            flat_dim = Xf.shape[1]
            scb = StandardScaler().fit(Xb.reshape(-1, per_bin_dim))
            scf = StandardScaler().fit(Xf)
            Xb_s = scb.transform(Xb.reshape(-1, per_bin_dim)).reshape(Xb.shape)
            Xf_s = scf.transform(Xf)
            model = train_fusion(Xb_s, Xf_s, y_idx, len(LABELS_S1), per_bin_dim, flat_dim, 200, 0)
            model.eval()
            Xb_t = torch.tensor(Xb_s[:100], dtype=torch.float32).to(DEVICE)
            Xf_t = torch.tensor(Xf_s[:100], dtype=torch.float32).to(DEVICE)
            for _ in range(50):
                with torch.no_grad():
                    _ = model(Xb_t, Xf_t)
            times = []
            for _ in range(200):
                t0 = time.perf_counter()
                with torch.no_grad():
                    _ = model(Xb_t, Xf_t)
                times.append((time.perf_counter() - t0) * 1000)
            size_mb = sum(p.numel() for p in model.parameters()) * 4 * 2 / (1024 * 1024)

    time_ms, time_std = np.mean(times), np.std(times)
    print(f"    Inference: {time_ms:.3f} ± {time_std:.3f} ms   Size: {size_mb:.3f} MB")

    results = {
        'model_selected_on_core': result['model_selected_on_core'],
        'final_model': best_model,
        'final_feature_groups': result['final_groups'],
        'feature_selection_log': result['selection_log'],
        'confusion_matrix': cm.tolist(),
        'edge_metrics': {'inference_time_ms': float(time_ms), 'model_size_mb': float(size_mb)},
    }
    with open(f'{RESULTS_DIR}/stage1_final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("STAGE 1 COMPLETE")
    print(f"  Final model: {best_model}")
    print(f"  Final feature groups: {result['final_groups']}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
