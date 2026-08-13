"""
Cross-Domain Ablation Study — Kalman Fusion + Relative Position Encoding
=========================================================================

Second ablation round.  Baseline for this round is 'Temporal Attention'
(the best variant from the first ablation in ablation_cross_domain.py).

Ablation variants
-----------------
  Baseline (Attention)       — Temporal Attention always on; no new features
  Kalman Fusion              — + Kalman sensor-fusion features (10 dims)
  Relative Position Encoding — + RPE features (10 dims)
  Kalman + RPE               — both families combined

Stage policy
------------
  Stage 1  (binary, acc-only):   Kalman skipped (no gyro in baseline)
  Stage 2a (fall subtypes):      Kalman skipped (acc-only baseline)
  Stage 2b (ADL, 11-class):      Full grid — both families tested

Best-model mapping (carried over from prior LOSO experiments)
-------------------------------------------------------------
  Stage 1  → SVM-RBF
  Stage 2a → DualBranchFusionNet
  Stage 2b → RandomForest

Results
-------
  results/ablation_kalman_rpe/kalman_rpe_ablation_results.csv

Usage
-----
  python ablation_kalman_rpe.py [--data-root PATH] [--out-dir PATH]
                                [--seeds 0 1 2 3 4] [--stages 1 2a 2b]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from src.kalman_rpe_builders import (
    FALL_CODES, ADL_CODES_11,
    build_stage1_krpe, build_stage2a_krpe, build_stage2b_krpe,
)
from src.ablation_eval import run_loso_classical, run_loso_fusion

# ============================================================
# Defaults
# ============================================================
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    'MobiAct_Dataset_v2.0', 'Annotated Data',
)
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), 'results', 'ablation_kalman_rpe'
)
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

LABELS_S1  = ['ADL', 'FALL']
LABELS_S2A = sorted(FALL_CODES)
LABELS_S2B = sorted(ADL_CODES_11)

# Variants: note Kalman is present in the grid but suppressed inside
# the builder for stages 1 and 2a — so those rows will simply not use it.
VARIANTS = {
    'Baseline (Attention)':       {'kalman': False, 'rpe': False},
    'Kalman Fusion':              {'kalman': True,  'rpe': False},
    'Relative Position Encoding': {'kalman': False, 'rpe': True},
    'Kalman + RPE':               {'kalman': True,  'rpe': True},
}

STAGE_CFG = [
    {
        'name':    'Stage 1',
        'desc':    'Stage 1: Binary Fall vs ADL',
        'builder': build_stage1_krpe,
        'model':   'SVM-RBF',
        'labels':  LABELS_S1,
        'epochs':  200,
    },
    {
        'name':    'Stage 2a',
        'desc':    'Stage 2a: Fall Subtypes (4-class)',
        'builder': build_stage2a_krpe,
        'model':   'Fusion',
        'labels':  LABELS_S2A,
        'epochs':  300,
    },
    {
        'name':    'Stage 2b',
        'desc':    'Stage 2b: ADL Classification (11-class)',
        'builder': build_stage2b_krpe,
        'model':   'RandomForest',
        'labels':  LABELS_S2B,
        'epochs':  150,
    },
]


# ============================================================
# Per-stage ablation runner
# ============================================================

def run_stage_ablation(cfg: dict, data_root: str, seeds: list[int]) -> dict:
    stage_name = cfg['name']
    model_name = cfg['model']
    labels     = cfg['labels']
    epochs     = cfg['epochs']
    builder    = cfg['builder']

    print(f"\n{'='*70}")
    print(f"  {cfg['desc']}")
    print(f"  Best model: {model_name}  |  Seeds: {seeds}")
    print(f"{'='*70}")

    stage_results = {}

    for variant_name, params in VARIANTS.items():
        print(f"\n  [{variant_name}]  building dataset …", end=' ', flush=True)

        X_bins, X_flat, y, groups = builder(
            data_root,
            include_kalman=params['kalman'],
            include_rpe=params['rpe'],
        )

        if len(X_bins) == 0:
            print("⚠  no data — skipped")
            continue

        print(f"{len(X_bins)} samples  |  flat_dim={X_flat.shape[1]}"
              f"  bins_shape={X_bins.shape[1:]}")

        if model_name == 'Fusion':
            result = run_loso_fusion(
                X_bins, X_flat, y, groups,
                labels=labels, epochs=epochs, seeds=seeds,
            )
        else:
            result = run_loso_classical(
                X_flat, y, groups,
                model_name=model_name, labels=labels, seeds=seeds,
            )

        acc     = result['accuracy']
        bal     = result['balanced_accuracy']
        f1      = result['macro_f1']
        acc_std = result['accuracy_std']
        print(f"    acc={acc:.4f}±{acc_std:.4f}  bal={bal:.4f}  f1={f1:.4f}")

        stage_results[variant_name] = result

    return stage_results


# ============================================================
# Summary printer
# ============================================================

def print_summary(all_results: dict):
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)

    for stage_name, stage_results in all_results.items():
        print(f"\n  {stage_name}")
        print(f"    {'Variant':<32} {'Accuracy':<14} {'Bal Acc':<12} {'F1':<12} {'Δ acc'}")
        print("    " + "-" * 72)

        baseline_acc = stage_results.get(
            'Baseline (Attention)', {}
        ).get('accuracy', 0.0)

        for variant_name, result in stage_results.items():
            acc   = result['accuracy']
            bal   = result['balanced_accuracy']
            f1    = result['macro_f1']
            std   = result['accuracy_std']
            delta = (acc - baseline_acc) * 100.0
            marker = "✅" if delta > 0.05 else ("⚠️" if abs(delta) <= 0.05 else "❌")
            print(
                f"    {variant_name:<32} {acc:.4f}±{std:.4f}  "
                f"{bal:.4f}      {f1:.4f}      "
                f"{marker} {delta:+.2f}%"
            )


# ============================================================
# Results saver
# ============================================================

def save_results(all_results: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for stage_name, stage_results in all_results.items():
        for variant_name, result in stage_results.items():
            rows.append({
                'stage':                 stage_name,
                'variant':               variant_name,
                'accuracy':              result['accuracy'],
                'accuracy_std':          result['accuracy_std'],
                'balanced_accuracy':     result['balanced_accuracy'],
                'balanced_accuracy_std': result.get('balanced_accuracy_std', 0.0),
                'macro_f1':              result['macro_f1'],
                'macro_f1_std':          result.get('macro_f1_std', 0.0),
                'macro_precision':       result['macro_precision'],
                'macro_recall':          result['macro_recall'],
            })

    df   = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'kalman_rpe_ablation_results.csv')
    df.to_csv(path, index=False)
    print(f"\n  ✅  Results saved → {path}")
    return path


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Ablation: Kalman Fusion + Relative Position Encoding'
    )
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                   help='Path to MobiAct Annotated Data directory')
    p.add_argument('--out-dir',   default=DEFAULT_OUT_DIR,
                   help='Output directory for CSV results')
    p.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
                   help='Random seeds (default: 0 1 2 3 4)')
    p.add_argument('--stages', nargs='+', default=['1', '2a', '2b'],
                   help='Stages to run  e.g. --stages 1 2a')
    return p.parse_args()


def main():
    args = parse_args()

    stage_key_map = {'1': 'Stage 1', '2a': 'Stage 2a', '2b': 'Stage 2b'}
    requested     = {stage_key_map[s] for s in args.stages if s in stage_key_map}
    stages_to_run = [c for c in STAGE_CFG if c['name'] in requested]

    if not stages_to_run:
        print("No valid stages specified. Use --stages 1 2a 2b")
        sys.exit(1)

    print("\n" + "="*70)
    print("  ABLATION: Kalman Fusion + Relative Position Encoding")
    print("  (Baseline = Temporal Attention from round 1)")
    print(f"  Stages : {[c['name'] for c in stages_to_run]}")
    print(f"  Seeds  : {args.seeds}")
    print(f"  Data   : {args.data_root}")
    print("="*70)

    all_results = {}
    for cfg in stages_to_run:
        all_results[cfg['name']] = run_stage_ablation(cfg, args.data_root, args.seeds)

    print_summary(all_results)
    save_results(all_results, args.out_dir)
    print("\n✅  Kalman + RPE ablation complete.")


if __name__ == '__main__':
    main()
