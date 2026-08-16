import os, json, pickle, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from src.config import RESULTS_DIR_S2B, LABELS_S2B, ADL_CODES_11, CLASSICAL_MODELS
from src.features.extractors import REGISTRY_S2B, build_binned_features, build_flat_features, feature_names_for
from src.data.loader import get_segment
from src.training.evaluate import run_stage_pipeline, get_loso_predictions
from src.models.classical import get_classical_models
from src.models.deep import train_bilstm, train_fusion
from src.config import DEVICE

def build_dataset_stage2b(groups):
    include_gyro_bins = 'gyro' in groups
    Xb, Xf, y, g = [], [], [], []
    for code in ADL_CODES_11:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None:
                continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values if 'gyro_x' in seg.columns else np.zeros_like(acc)
            roll = seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg))
            fb = build_binned_features(acc, gyro, n_bins=5, include_gyro=include_gyro_bins)
            fc = build_flat_features(acc, gyro, roll, groups, REGISTRY_S2B)
            if fb is not None and fc is not None:
                Xb.append(fb)
                Xf.append(fc)
                y.append(code)
                g.append(subj)
    return np.array(Xb), np.array(Xf), np.array(y), np.array(g)

def run_rpe_diagnostic():
    pass

def main():
    print("\n" + "#" * 70)
    print("# STAGE 2b")
    print("#" * 70)

    run_rpe_diagnostic()

    result = run_stage_pipeline("STAGE 2b", build_dataset_stage2b, REGISTRY_S2B, LABELS_S2B,
                                 epochs_selection=150, bilstm_hidden=16)

    best_model = result['final_model']
    Xb, Xf, y, g = result['Xb'], result['Xf'], result['y'], result['g']
    feature_names = feature_names_for(result['final_groups'], REGISTRY_S2B)

    print("\n" + "=" * 70)
    print("STAGE 2b — PART 3: PER-CLASS PERFORMANCE (final config)")
    print("=" * 70)
    all_true, all_pred = get_loso_predictions(best_model, Xb, Xf, y, g, LABELS_S2B, epochs=150, hidden=16)
    print(f"\nClassification Report ({best_model}, features: {result['final_groups']}):")
    print(classification_report(all_true, all_pred, target_names=LABELS_S2B))
    cm = confusion_matrix(all_true, all_pred, labels=LABELS_S2B)
    print(pd.DataFrame(cm, index=LABELS_S2B, columns=LABELS_S2B))

    per_class_metrics = {}
    for i, cls in enumerate(LABELS_S2B):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(len(LABELS_S2B)) if j != i)
        fn = sum(cm[i][j] for j in range(len(LABELS_S2B)) if j != i)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class_metrics[cls] = {'precision': precision, 'recall': recall, 'f1': f1}

    print("\n" + "=" * 70)
    print("STAGE 2b — PART 4: VISUALIZATION")
    print("=" * 70)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_S2B, yticklabels=LABELS_S2B,
                ax=ax, cbar=False, linewidths=0.5, linecolor='white', annot_kws={'size': 8})
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Stage 2b Confusion Matrix — {best_model}')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR_S2B}/stage2b_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR_S2B}/stage2b_confusion_matrix.png")

    if best_model in CLASSICAL_MODELS:
        sc = StandardScaler().fit(Xf)
        clf_full = get_classical_models(0, len(LABELS_S2B))[best_model]
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
            for label in LABELS_S2B:
                mask = y == label
                ax.hist(Xf[mask, idx], bins=20, alpha=0.5, label=label, density=True)
            ax.set_xlabel(feat); ax.set_ylabel('Density'); ax.legend()
            ax.set_title(f'Top {i+1} Feature: {feat}')
        plt.suptitle(f'Stage 2b — Top Discriminative Features ({best_model})')
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR_S2B}/stage2b_feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {RESULTS_DIR_S2B}/stage2b_feature_distributions.png")
    else:
        print(f"  Best model is {best_model} — skipping permutation-importance plot.")

    selected_adls = ['WAL', 'JOG', 'SIT', 'STD']
    Xb_by_adl = {label: np.mean(Xb[y == label, :, :3], axis=0) for label in selected_adls if (y == label).sum() > 0}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for label, ax in zip(selected_adls, axes):
        if label in Xb_by_adl:
            data = Xb_by_adl[label]
            for axis in range(3):
                ax.plot(range(5), data[:, axis], 'o-', label=f'Axis {["X","Y","Z"][axis]}',
                        color=colors[axis], linewidth=2, markersize=8)
            ax.set_xlabel('Time Bin'); ax.set_ylabel('Normalized Energy')
            ax.set_title(label); ax.legend(); ax.grid(True, alpha=0.3)
    plt.suptitle('Energy Profiles by ADL Type')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR_S2B}/stage2b_energy_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR_S2B}/stage2b_energy_profiles.png")

    print("\n" + "=" * 70)
    print("STAGE 2b — PART 5: EDGE ANALYSIS")
    print("=" * 70)
    if best_model in CLASSICAL_MODELS:
        sc = StandardScaler().fit(Xf)
        clf = get_classical_models(0, len(LABELS_S2B))[best_model]
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
        l2i = {l: i for i, l in enumerate(LABELS_S2B)}
        y_idx = np.array([l2i[l] for l in y])
        per_bin_dim = Xb.shape[2]
        if best_model == 'BiLSTM':
            sc = StandardScaler().fit(Xb.reshape(-1, per_bin_dim))
            Xb_s = sc.transform(Xb.reshape(-1, per_bin_dim)).reshape(Xb.shape)
            model = train_bilstm(Xb_s, y_idx, len(LABELS_S2B), per_bin_dim, 150, 16, 0)
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
            model = train_fusion(Xb_s, Xf_s, y_idx, len(LABELS_S2B), per_bin_dim, flat_dim, 150, 0)
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
        'per_class_metrics': per_class_metrics,
        'edge_metrics': {'inference_time_ms': float(time_ms), 'model_size_mb': float(size_mb)},
    }
    with open(f'{RESULTS_DIR_S2B}/stage2b_final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("STAGE 2b COMPLETE")
    print(f"  Final model: {best_model}")
    print(f"  Final feature groups: {result['final_groups']}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
