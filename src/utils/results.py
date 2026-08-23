import os
import pandas as pd


def rewrite_summary_csv(all_results, results_dir):
    """
    Rebuild SUMMARY_all_tasks_modes_protocols.csv from `all_results`.

    Called after every completed (or resumed) config so the CSV is always
    up-to-date regardless of which configs were skipped due to resume logic.

    Parameters
    ----------
    all_results : list[dict] — accumulated result dicts from the main loop
    results_dir : str        — directory where the CSV is written
    """
    rows = [
        {
            'task':              r['task'],
            'mode':              r['mode'],
            'protocol':          r['protocol'],
            'accuracy':          f"{r['accuracy_mean']:.4f} +/- {r['accuracy_sd']:.4f}",
            'balanced_accuracy': f"{r['balanced_accuracy_mean']:.4f} "
                                 f"+/- {r['balanced_accuracy_sd']:.4f}",
            'macro_f1':          f"{r['macro_f1_mean']:.4f} +/- {r['macro_f1_sd']:.4f}",
        }
        for r in all_results
    ]
    out_path = os.path.join(results_dir, 'SUMMARY_all_tasks_modes_protocols.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False)
