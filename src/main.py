import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, classification_report)

from src.config import (TASK_CONFIG, MODES, SEEDS_CV, SEEDS_LOSO,
                        RESULTS_DIR, EPOCHS, FIC_LOSS_WEIGHT,
                        FIC_VAR_THRESH, FIC_CORR_THRESH, DEVICE,
                        WIN_LEN, STEP, FS_TARGET, WINDOW_SEC, OVERLAP)
from src.data.loader import build_raw_dataset
from src.data.features import extract_handcrafted_features
from src.training.evaluate import run_cv, run_loso, compute_metrics
from src.utils.results import rewrite_summary_csv

warnings.filterwarnings('ignore')
print(f"Device: {DEVICE}")
print(f"Window: {WIN_LEN} samples ({WINDOW_SEC}s @ {FS_TARGET}Hz), "
      f"step {STEP} samples ({OVERLAP*100:.0f}% overlap)")


def main():
    all_results = []

    for task_name, cfg in TASK_CONFIG.items():
        print("\n" + "=" * 70)
        print(f"TASK: {task_name}")
        print("=" * 70)

        X, y, subj = build_raw_dataset(cfg['codes'], cfg['label_fn'])
        labels = sorted(np.unique(y).tolist())
        print(f"Windows: {X.shape}, classes: {labels}, "
              f"subjects: {len(np.unique(subj))}")

        # Handcrafted features computed once, raw (unscaled).
        # Scaling + selection happen per-fold on training data only.
        feat_raw = extract_handcrafted_features(X)
        print(f"Handcrafted features: {feat_raw.shape}")

        task_dir = os.path.join(RESULTS_DIR, task_name)
        os.makedirs(task_dir, exist_ok=True)

        for mode in MODES:
            for protocol in ('cv', 'loso'):
                result_path  = os.path.join(task_dir, f'{mode}_{protocol}.json')
                partial_path = os.path.join(task_dir, f'{mode}_{protocol}_PARTIAL.json')

                # ── RESUME SUPPORT ────────────────────────────────────────────
                if os.path.exists(result_path):
                    print(f"\n--- {task_name} | {mode} | {protocol} --- "
                          f"ALREADY DONE, skipping "
                          f"(delete {result_path} to redo)", flush=True)
                    with open(result_path) as f:
                        all_results.append(json.load(f))
                    rewrite_summary_csv(all_results, RESULTS_DIR)
                    continue

                print(f"\n--- {task_name} | mode={mode} | protocol={protocol} ---",
                      flush=True)
                seeds = SEEDS_CV if protocol == 'cv' else SEEDS_LOSO
                print(f"  seeds: {seeds}", flush=True)

                seed_metrics = []
                last_true, last_pred = [], []

                for seed in seeds:
                    t0 = time.time()

                    if protocol == 'cv':
                        t_true, t_pred = run_cv(
                            mode, X, y, subj, feat_raw, labels, epochs=EPOCHS, seed=seed
                        )
                        fold_info = None
                    else:
                        t_true, t_pred, fold_info = run_loso(
                            mode, X, y, subj, feat_raw, labels, epochs=EPOCHS, seed=seed
                        )

                    acc, bal_acc, f1 = compute_metrics(t_true, t_pred, labels)
                    elapsed = time.time() - t0
                    seed_metrics.append({
                        'seed': seed, 'accuracy': acc,
                        'balanced_accuracy': bal_acc, 'macro_f1': f1,
                        'runtime_sec': elapsed,
                    })
                    last_true, last_pred = t_true, t_pred
                    print(f"  seed {seed}: acc={acc:.4f}  bal_acc={bal_acc:.4f}  "
                          f"macro_f1={f1:.4f}  ({elapsed:.0f}s)", flush=True)

                    # Per-seed checkpoint
                    with open(partial_path, 'w') as f:
                        json.dump({
                            'task': task_name, 'mode': mode, 'protocol': protocol,
                            'seeds_completed': [m['seed'] for m in seed_metrics],
                            'seeds_target': seeds,
                            'per_seed_so_far': seed_metrics,
                            'last_true': list(map(str, t_true)),
                            'last_pred': list(map(str, t_pred)),
                        }, f, indent=2, default=str)

                accs     = [m['accuracy']          for m in seed_metrics]
                bal_accs = [m['balanced_accuracy']  for m in seed_metrics]
                f1s      = [m['macro_f1']           for m in seed_metrics]
                cm       = confusion_matrix(last_true, last_pred, labels=labels)
                report   = classification_report(
                    last_true, last_pred, labels=labels,
                    output_dict=True, zero_division=0,
                )

                result = {
                    'task':     task_name,
                    'mode':     mode,
                    'protocol': protocol,
                    'labels':   labels,
                    'accuracy_mean':          float(np.mean(accs)),
                    'accuracy_sd':            float(np.std(accs)),
                    'balanced_accuracy_mean': float(np.mean(bal_accs)),
                    'balanced_accuracy_sd':   float(np.std(bal_accs)),
                    'macro_f1_mean':          float(np.mean(f1s)),
                    'macro_f1_sd':            float(np.std(f1s)),
                    'per_seed':               seed_metrics,
                    'confusion_matrix_last_seed':       cm.tolist(),
                    'classification_report_last_seed':  report,
                }
                if mode in ('fic', 'sam_fic'):
                    result['fic_config'] = {
                        'fic_loss_weight': FIC_LOSS_WEIGHT,
                        'var_thresh':      FIC_VAR_THRESH,
                        'corr_thresh':     FIC_CORR_THRESH,
                        'optimizer':       'SAM' if mode == 'sam_fic' else 'Adam',
                    }

                all_results.append(result)
                with open(result_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                if os.path.exists(partial_path):
                    os.remove(partial_path)
                print(f"  Saved: {result_path}", flush=True)

                rewrite_summary_csv(all_results, RESULTS_DIR)

    # ── Final printout ────────────────────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, 'SUMMARY_all_tasks_modes_protocols.csv')
    summary_df   = pd.read_csv(summary_path)
    print("\n" + "=" * 70)
    print("FULL SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
