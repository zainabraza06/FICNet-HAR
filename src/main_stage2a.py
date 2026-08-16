import os, json, pickle, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from src.config import RESULTS_DIR_S2A, LABELS_S2A, FALL_CODES, CLASSICAL_MODELS
from src.features.extractors import REGISTRY_S2A, build_binned_features, build_flat_features, feature_names_for
from src.data.loader import get_segment
from src.training.evaluate import run_stage_pipeline, get_loso_predictions
from src.models.classical import get_classical_models
from src.models.deep import train_bilstm, train_fusion
from src.config import DEVICE

def build_dataset_stage2a(groups):
    Xb, Xf, y, g = [], [], [], []
    for code in FALL_CODES:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None:
                continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x', 'acc_y', 'acc_z']].values
            fb = build_binned_features(acc, n_bins=5, include_gyro=False)
            fc = build_flat_features(acc, None, None, groups, REGISTRY_S2A)
            if fb is not None and fc is not None:
                Xb.append(fb)
                Xf.append(fc)
                y.append(code)
                g.append(subj)
    return np.array(Xb), np.array(Xf), np.array(y), np.array(g)

def run_rpe_diagnostic():
    # Diagnostic ran in Stage 1 usually, skipping direct invocation here unless explicitly needed
    pass

def main():
    print("\n" + "#" * 70)
    print("# STAGE 2a")
    print("#" * 70)

    run_rpe_diagnostic()

    result = run_stage_pipeline("STAGE 2a", build_dataset_stage2a, REGISTRY_S2A, LABELS_S2A,
                                 epochs_selection=300, bilstm_hidden=8)

    best_model = result['final_model']
    Xb, Xf, y, g = result['Xb'], result['Xf'], result['y'], result['g']

    print("\n" + "=" * 70)
    print("STAGE 2a — PART 3: PER-CLASS PERFORMANCE (final config)")
    print("=" * 70)
    all_true, all_pred = get_loso_predictions(best_model, Xb, Xf, y, g, LABELS_S2A, epochs=300, hidden=8)
    print(f"\nClassification Report ({best_model}, features: {result['final_groups']}):")
    print(classification_report(all_true, all_pred, target_names=LABELS_S2A))
    cm = confusion_matrix(all_true, all_pred, labels=LABELS_S2A)
    print(pd.DataFrame(cm, index=LABELS_S2A, columns=LABELS_S2A))

    print("\n" + "=" * 70)
    print("STAGE 2a — PART 4: VISUALIZATION")
    print("=" * 70)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_S2A, yticklabels=LABELS_S2A, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Stage 2a Confusion Matrix — {best_model}')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR_S2A}/stage2a_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR_S2A}/stage2a_confusion_matrix.png")

    Xb_by_class = {label: np.mean(Xb[y == label, :, :3], axis=0) for label in LABELS_S2A}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for label, ax in zip(LABELS_S2A, axes):
        data = Xb_by_class[label]
        for axis in range(3):
            ax.plot(range(5), data[:, axis], 'o-', label=f'Axis {["X","Y","Z"][axis]}', linewidth=2)
        ax.set_title(label); ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle('Energy Profiles by Fall Type')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR_S2A}/stage2a_energy_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR_S2A}/stage2a_energy_profiles.png")

    print("\n" + "=" * 70)
    print("STAGE 2a — PART 5: EDGE ANALYSIS")
    print("=" * 70)
    if best_model in CLASSICAL_MODELS:
        sc = StandardScaler().fit(Xf)
        clf = get_classical_models(0, len(LABELS_S2A))[best_model]
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
        l2i = {l: i for i, l in enumerate(LABELS_S2A)}
        y_idx = np.array([l2i[l] for l in y])
        per_bin_dim = Xb.shape[2]
        if best_model == 'BiLSTM':
            sc = StandardScaler().fit(Xb.reshape(-1, per_bin_dim))
            Xb_s = sc.transform(Xb.reshape(-1, per_bin_dim)).reshape(Xb.shape)
            model = train_bilstm(Xb_s, y_idx, len(LABELS_S2A), per_bin_dim, 300, 8, 0)
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
            model = train_fusion(Xb_s, Xf_s, y_idx, len(LABELS_S2A), per_bin_dim, flat_dim, 300, 0)
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
    with open(f'{RESULTS_DIR_S2A}/stage2a_final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("STAGE 2a COMPLETE")
    print(f"  Final model: {best_model}")
    print(f"  Final feature groups: {result['final_groups']}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
