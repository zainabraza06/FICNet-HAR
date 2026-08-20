import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              confusion_matrix, classification_report,
                              precision_recall_fscore_support)

from src.config import (TASK_CONFIG, TASKS_TO_RUN, MODELS, RESULTS_ROOT, EPOCHS, DEVICE)
from src.features.extractors import CORE_GROUPS, CANDIDATE_GROUPS
from src.data.loader import build_dataset
from src.training.evaluate import (run_model_selection, select_best_model,
                                   greedy_feature_selection, mean_ci)
from src.utils.viz import plot_confusion_matrix

warnings.filterwarnings('ignore')
print(f"Using device: {DEVICE}")


def run_task(task_name):
    """
    Full pipeline for a single task:
      1. Model selection on core feature set (5 seeds × LOSO)
      2. Greedy, subject-level significance-gated feature selection
      3. Final metrics + confusion matrix + JSON output

    Returns a summary dict for the cross-task table.
    """
    print("\n" + "#" * 70)
    print(f"# TASK: {task_name}")
    print("#" * 70)

    cfg            = TASK_CONFIG[task_name]
    codes_for_task = cfg['codes']
    labels         = cfg['labels']
    label_fn       = cfg['label_fn']

    results_dir = os.path.join(RESULTS_ROOT, f'task_{task_name}')
    os.makedirs(results_dir, exist_ok=True)

    # ── Part 1: model selection on core feature set ───────────────────────────
    print(f"\n{'=' * 70}\nPART 1: MODEL SELECTION [{task_name}] — core set\n{'=' * 70}")
    Xb_core, Xf_core, y_core, g_core = build_dataset(
        CORE_GROUPS, codes_for_task, label_fn
    )
    print(f"Dataset: Xb={Xb_core.shape}, Xf={Xf_core.shape}, classes={labels}")

    model_results_core = {}
    for model in MODELS:
        per_seed = run_model_selection(model, Xb_core, Xf_core, y_core, g_core,
                                       labels, epochs=EPOCHS, hidden=16)
        accs = [accuracy_score(t, p) for t, p, _ in per_seed]
        stat = mean_ci(accs)
        print(f"  {model}: acc={stat['mean']:.4f} ± {stat['sd']:.4f}  "
              f"95% CI [{stat['ci95'][0]:.4f}, {stat['ci95'][1]:.4f}]")
        model_results_core[model] = per_seed

    model_stage = select_best_model(model_results_core)
    print(f"\n  >>> Model selected: {model_stage}")

    # ── Part 2: greedy feature selection ─────────────────────────────────────
    print(f"\n{'=' * 70}\nPART 2: FEATURE SELECTION [{task_name}] on {model_stage}\n{'=' * 70}")
    final_groups, final_per_seed, selection_log = greedy_feature_selection(
        model_name       = model_stage,
        current_groups   = list(CORE_GROUPS),
        remaining        = list(CANDIDATE_GROUPS),
        current_per_seed = model_results_core[model_stage],
        codes_for_task   = codes_for_task,
        label_fn         = label_fn,
        labels           = labels,
        build_dataset_fn = build_dataset,
        epochs           = EPOCHS,
        hidden           = 16,
    )
    print(f"\n  >>> FINAL FEATURE SET [{task_name}]: {final_groups}")

    # ── Part 3: final metrics + outputs ──────────────────────────────────────
    print(f"\n{'=' * 70}\nPART 3: FINAL RESULTS [{task_name}]\n{'=' * 70}")
    pooled_accs = [accuracy_score(t, p)            for t, p, _ in final_per_seed]
    pooled_bal  = [balanced_accuracy_score(t, p)   for t, p, _ in final_per_seed]
    pooled_f1   = [
        precision_recall_fscore_support(t, p, labels=labels, average='macro',
                                        zero_division=0)[2]
        for t, p, _ in final_per_seed
    ]
    acc_stat = mean_ci(pooled_accs)
    bal_stat = mean_ci(pooled_bal)
    f1_stat  = mean_ci(pooled_f1)

    print(f"  Accuracy:     {acc_stat['mean']:.4f} ± {acc_stat['sd']:.4f}  "
          f"95% CI {acc_stat['ci95']}")
    print(f"  Balanced Acc: {bal_stat['mean']:.4f} ± {bal_stat['sd']:.4f}")
    print(f"  Macro F1:     {f1_stat['mean']:.4f} ± {f1_stat['sd']:.4f}")

    all_true, all_pred, _ = final_per_seed[0]
    print(f"\nClassification Report (seed 0, {model_stage}, features={final_groups}):")
    print(classification_report(all_true, all_pred, target_names=labels, digits=4))

    cm = confusion_matrix(all_true, all_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        os.path.join(results_dir, f'{task_name}_confusion_matrix.csv')
    )
    plot_confusion_matrix(cm, labels, task_name, model_stage, results_dir)

    results = {
        'task':              task_name,
        'labels':            labels,
        'final_model':       model_stage,
        'final_groups':      final_groups,
        'selection_log':     selection_log,
        'accuracy':          acc_stat,
        'balanced_accuracy': bal_stat,
        'macro_f1':          f1_stat,
        'confusion_matrix':  cm.tolist(),
    }
    with open(os.path.join(results_dir, f'{task_name}_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{task_name} COMPLETE — model={model_stage}, "
          f"features={final_groups}, acc={acc_stat['mean']:.4f}")

    return {
        'task':              task_name,
        'model':             model_stage,
        'features':          final_groups,
        'accuracy':          acc_stat,
        'balanced_accuracy': bal_stat,
        'macro_f1':          f1_stat,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    all_task_results = []
    for task_name in TASKS_TO_RUN:
        result = run_task(task_name)
        all_task_results.append(result)

    print("\n" + "#" * 70)
    print("# SUMMARY — ALL TASKS")
    print("#" * 70)
    rows = []
    for r in all_task_results:
        rows.append({
            'Task':         r['task'],
            'Model':        r['model'],
            'Features':     '+'.join(r['features']),
            'Accuracy':     f"{r['accuracy']['mean']:.4f} ± {r['accuracy']['sd']:.4f}",
            'Balanced Acc': f"{r['balanced_accuracy']['mean']:.4f}",
            'Macro F1':     f"{r['macro_f1']['mean']:.4f}",
        })
    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(RESULTS_ROOT, 'all_tasks_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")


if __name__ == '__main__':
    main()
