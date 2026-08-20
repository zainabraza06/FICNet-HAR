import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, labels, task_name, model_name, results_dir):
    """Save a confusion-matrix heatmap for a given task."""
    size = max(6, len(labels))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels, ax=ax,
        cbar=False, linewidths=0.5, linecolor='white',
        annot_kws={'size': max(6, 10 - len(labels) // 4)},
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{task_name} — Confusion Matrix ({model_name}, seed 0)')
    plt.tight_layout()
    out_path = os.path.join(results_dir, f'{task_name}_confusion_matrix.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")
