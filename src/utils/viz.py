import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from src.config import RESULTS_DIR, LABELS_S1, CLASSICAL_MODELS
from src.models.classical import get_classical_models

def plot_confusion_matrix(cm, best_model, feature_set_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_S1, yticklabels=LABELS_S1,
                ax=ax, cbar=False, linewidths=0.5, linecolor='white')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Stage 1 Confusion Matrix — {best_model} ({feature_set_name})')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/stage1_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/stage1_confusion_matrix.png")

def plot_feature_distributions(best_model, Xf, y, feature_names):
    if best_model not in CLASSICAL_MODELS:
        print(f"  Best model is {best_model} — skipping permutation-importance plot")
        return
        
    sc = StandardScaler().fit(Xf)
    clf_full = get_classical_models(0)[best_model]
    clf_full.fit(sc.transform(Xf), y)
    result = permutation_importance(clf_full, sc.transform(Xf), y,
                                    n_repeats=30, random_state=42, scoring='accuracy')
    imp_mean = result.importances_mean
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
        ax.set_xlabel(feat)
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_title(f'Top {i+1} Feature: {feat}')
    plt.suptitle(f'Stage 1 — Top Discriminative Features ({best_model})')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/stage1_feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/stage1_feature_distributions.png")

def plot_energy_profiles(Xb, y):
    Xb_fall = Xb[y == 'FALL']
    Xb_adl = Xb[y == 'ADL']
    avg_fall = np.mean(Xb_fall[:, :, :3], axis=0)
    avg_adl = np.mean(Xb_adl[:, :, :3], axis=0)
    
    for axis in range(3):
        avg_fall[:, axis] = avg_fall[:, axis] / (avg_fall[:, axis].sum() + 1e-8)
        avg_adl[:, axis] = avg_adl[:, axis] / (avg_adl[:, axis].sum() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.arange(5)
    for ax, data, title in zip(axes, [avg_fall, avg_adl], ['FALL', 'ADL']):
        for axis in range(3):
            ax.plot(bins, data[:, axis], 'o-', label=f'Axis {["X","Y","Z"][axis]}', linewidth=2, markersize=8)
        ax.set_xlabel('Time Bin')
        ax.set_ylabel('Normalized Energy')
        ax.set_title(f'Energy Profile — {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(bins)
        ax.set_xticklabels([f'Bin {i+1}' for i in range(5)])
    plt.suptitle('Average Energy Profiles: FALL vs ADL')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/stage1_energy_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/stage1_energy_profiles.png")

def plot_binned_boxplots(Xb, y):
    Xb_fall = Xb[y == 'FALL']
    Xb_adl = Xb[y == 'ADL']
    bin_axis_pairs = [(b, a) for b in range(5) for a in range(3)]
    t_stats_list = []
    for b, a in bin_axis_pairs:
        t, _ = stats.ttest_ind(Xb_fall[:, b, a], Xb_adl[:, b, a])
        t_stats_list.append((abs(t), b, a))
    top_pairs = sorted(t_stats_list, reverse=True)[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, (_, b, a) in enumerate(top_pairs):
        ax = axes[i]
        data = [Xb_adl[:, b, a], Xb_fall[:, b, a]]
        bp = ax.boxplot(data, labels=['ADL', 'FALL'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
            patch.set_facecolor(color)
        ax.set_title(f'Bin {b+1} — Axis {["X","Y","Z"][a]}')
        ax.set_ylabel('Normalized Energy')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Top 6 Most Discriminative Binned Features (FALL vs ADL)')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/stage1_binned_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR}/stage1_binned_boxplots.png")
