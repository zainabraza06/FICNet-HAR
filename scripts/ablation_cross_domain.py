"""
Cross-Domain Ablation Study — Temporal Attention + Multi-Scale FPN
===================================================================

Tests the marginal contribution of two CV-adapted feature families
on ALL THREE stages of MobiAct (binary fall gate, fall subtypes,
ADL classification).

Ablation variants
-----------------
  Baseline         — best-performing model from prior experiments,
                     no extra features.
  Temporal Attention — + attention entropy (3 dims)
  Multi-Scale FPN  — + multi-scale temporal features (26 dims)
  Attention + FPN  — both families combined

Best-model mapping (from prior LOSO experiments)
-------------------------------------------------
  Stage 1  → SVM-RBF      on flat features
  Stage 2a → Fusion       (DualBranchFusionNet)
  Stage 2b → RandomForest on flat features

Results are written to:
  results/ablation_cross_domain/cross_domain_ablation_results.csv

Usage
-----
  python ablation_cross_domain.py [--data-root PATH] [--out-dir PATH]
                                  [--seeds 0 1 2 3 4]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Allow running from project root without installing the package
sys.path.insert(0, os.path.dirname(__file__))

from src.ablation_builders import (
    FALL_CODES, ADL_CODES_11,
    build_stage1, build_stage2a, build_stage2b,
)
from src.ablation_eval import run_loso_classical, run_loso_fusion

# ============================================================
# Defaults
# ============================================================
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    'MobiAct_Dataset_v2.0', 'Annotated Data'
)
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), 'results', 'ablation_cross_domain')
DEFAULT_SEEDS   = [0, 1, 2, 3, 4]

LABELS_S1  = ['ADL', 'FALL']
LABELS_S2A = sorted(FALL_CODES)
LABELS_S2B = sorted(ADL_CODES_11)

VARIANTS = {
    'Baseline':           {'attention': False, 'multiscale': False},
    'Temporal Attention': {'attention': True,  'multiscale': False},
    'Multi-Scale FPN':    {'attention': False, 'multiscale': True},
    'Attention + FPN':    {'attention': True,  'multiscale': True},
}

STAGE_CFG = [
    {
        'name':       'Stage 1',
        'desc':       'Stage 1: Binary Fall vs ADL',
        'builder':    build_stage1,
        'model':      'SVM-RBF',
        'labels':     LABELS_S1,
        'epochs':     200,          # unused for classical; kept for doc
    },
    {
        'name':       'Stage 2a',
        'desc':       'Stage 2a: Fall Subtypes (4-class)',
        'builder':    build_stage2a,
        'model':      'Fusion',
        'labels':     LABELS_S2A,
        'epochs':     300,
    },
    {
        'name':       'Stage 2b',
        'desc':       'Stage 2b: ADL Classification (11-class)',
        'builder':    build_stage2b,
        'model':      'RandomForest',
        'labels':     LABELS_S2B,
        'epochs':     150,          # unused for classical
    },
]


# ============================================================
# Per-stage ablation runner
# ============================================================

def run_stage_ablation(cfg: dict, data_root: str, seeds: list[int]) -> dict:
    """Run all four ablation variants for one stage configuration."""
    stage_name  = cfg['name']
    model_name  = cfg['model']
    labels      = cfg['labels']
    epochs      = cfg['epochs']
    builder     = cfg['builder']

    print(f"\n{'='*70}")
    print(f"  {cfg['desc']}")
    print(f"  Best model: {model_name}  |  Seeds: {seeds}")
    print(f"{'='*70}")

    stage_results = {}

    for variant_name, params in VARIANTS.items():
        print(f"\n  [{variant_name}]  building dataset …", end=' ', flush=True)

        X_bins, X_flat, y, groups = builder(
            data_root,
            include_attention=params['attention'],
            include_multiscale=params['multiscale'],
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

        acc = result['accuracy']
        bal = result['balanced_accuracy']
        f1  = result['macro_f1']
        acc_std = result['accuracy_std']
        print(f"    acc={acc:.4f}±{acc_std:.4f}  "
              f"bal={bal:.4f}  f1={f1:.4f}")

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
        header = f"    {'Variant':<25} {'Accuracy':<14} {'Bal Acc':<12} {'F1':<12} {'Δ acc'}"
        print(header)
        print("    " + "-" * 65)

        baseline_acc = stage_results.get('Baseline', {}).get('accuracy', 0.0)

        for variant_name, result in stage_results.items():
            acc  = result['accuracy']
            bal  = result['balanced_accuracy']
            f1   = result['macro_f1']
            std  = result['accuracy_std']
            delta = (acc - baseline_acc) * 100
            marker = "✅" if delta > 0 else ("⚠️" if delta == 0 else "❌")
            print(f"    {variant_name:<25} {acc:.4f}±{std:.4f}  "
                  f"{bal:.4f}      {f1:.4f}      "
                  f"{marker} {delta:+.2f}%")


# ============================================================
# Results saver
# ============================================================

def save_results(all_results: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for stage_name, stage_results in all_results.items():
        for variant_name, result in stage_results.items():
            rows.append({
                'stage':              stage_name,
                'variant':            variant_name,
                'accuracy':           result['accuracy'],
                'accuracy_std':       result['accuracy_std'],
                'balanced_accuracy':  result['balanced_accuracy'],
                'balanced_accuracy_std': result.get('balanced_accuracy_std', 0.0),
                'macro_f1':           result['macro_f1'],
                'macro_f1_std':       result.get('macro_f1_std', 0.0),
                'macro_precision':    result['macro_precision'],
                'macro_recall':       result['macro_recall'],
            })

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'cross_domain_ablation_results.csv')
    df.to_csv(path, index=False)
    print(f"\n  ✅  Results saved → {path}")
    return path


# ============================================================
# CLI entry point
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Cross-domain ablation: Temporal Attention + Multi-Scale FPN'
    )
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                   help='Path to MobiAct Annotated Data directory')
    p.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                   help='Directory where CSV results will be written')
    p.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS,
                   help='Random seeds to average over (default: 0 1 2 3 4)')
    p.add_argument('--stages', nargs='+', default=['1', '2a', '2b'],
                   help='Which stages to run  (e.g. --stages 1 2a)')
    return p.parse_args()


def main():
    args = parse_args()

    # Filter to requested stages
    stage_key_map = {'1': 'Stage 1', '2a': 'Stage 2a', '2b': 'Stage 2b'}
    requested     = {stage_key_map[s] for s in args.stages if s in stage_key_map}
    stages_to_run = [c for c in STAGE_CFG if c['name'] in requested]

    if not stages_to_run:
        print("No valid stages specified. Use --stages 1 2a 2b")
        sys.exit(1)

    print("\n" + "="*70)
    print("  CROSS-DOMAIN ABLATION: Temporal Attention + Multi-Scale FPN")
    print(f"  Stages : {[c['name'] for c in stages_to_run]}")
    print(f"  Seeds  : {args.seeds}")
    print(f"  Data   : {args.data_root}")
    print("="*70)

    all_results = {}
    for cfg in stages_to_run:
        all_results[cfg['name']] = run_stage_ablation(cfg, args.data_root, args.seeds)

    print_summary(all_results)
    save_results(all_results, args.out_dir)
    print("\n✅  Cross-domain ablation complete.")


if __name__ == '__main__':
    main()
