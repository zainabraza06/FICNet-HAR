"""
Streaming Ablation — Temporal Attention + Relative Position Encoding
=====================================================================

Tests the marginal contribution of Temporal Attention and RPE features
in the Stage 2b *streaming* setting using StreamingFusionNet (the best
streaming model from run_stage2b_streaming_final.py).

Ablation variants
-----------------
  Baseline (No Attention, No RPE)   — plain binned+flat features
  Temporal Attention Only           — + 3 attention entropy dims
  Relative Position Encoding Only   — + 10 RPE dims
  Temporal Attention + RPE          — both combined

Metrics reported per variant
----------------------------
  overall_accuracy / overall_balanced_accuracy
  pure_window_accuracy / pure_window_balanced_accuracy
  boundary_window_accuracy / boundary_window_balanced_accuracy
  pure_window_fraction / boundary_window_fraction

Results
-------
  results/ablation_streaming/streaming_ablation_results.csv

Usage
-----
  python ablation_streaming.py [--data-root PATH] [--out-dir PATH]
                               [--subjects 1..67] [--epochs 50]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from src.streaming_builders import (
    ADL_CODES_11, PURITY_THRESHOLD,
    build_streaming_dataset,
)
from src.streaming_eval import run_streaming_loso

# ============================================================
# Defaults
# ============================================================
DEFAULT_DATA_ROOT = os.path.join(
    os.path.dirname(__file__),
    'MobiAct_Dataset_v2.0', 'Annotated Data',
)
DEFAULT_OUT_DIR = os.path.join(
    os.path.dirname(__file__), 'results', 'ablation_streaming'
)
DEFAULT_SUBJECTS = list(range(1, 68))
DEFAULT_EPOCHS   = 50

VARIANTS = {
    'Baseline (No Attention, No RPE)': {'attention': False, 'rpe': False},
    'Temporal Attention Only':         {'attention': True,  'rpe': False},
    'Relative Position Encoding Only': {'attention': False, 'rpe': True},
    'Temporal Attention + RPE':        {'attention': True,  'rpe': True},
}


# ============================================================
# Runner
# ============================================================

def run_ablation(data_root: str, out_dir: str,
                 subjects: list[int], epochs: int) -> dict:
    all_results: dict = {}

    for variant_name, params in VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"  [{variant_name}]  building streaming dataset …")

        data = build_streaming_dataset(
            data_root=data_root,
            codes=ADL_CODES_11,
            subjects=subjects,
            include_attention=params['attention'],
            include_rpe=params['rpe'],
        )

        if not data:
            print("  ⚠  no subjects produced windows — skipped")
            continue

        sample_subj   = next(iter(data))
        flat_dim      = data[sample_subj][1].shape[1]
        total_windows = sum(len(v[0]) for v in data.values())
        print(f"  Subjects: {len(data)}  |  total windows: {total_windows}"
              f"  |  flat_dim: {flat_dim}")

        result = run_streaming_loso(
            data=data,
            label_list=ADL_CODES_11,
            epochs=epochs,
            seed=0,
            purity_thresh=PURITY_THRESHOLD,
            verbose=True,
        )

        oa  = result['overall_accuracy']
        pa  = result['pure_window_accuracy']   or 0.0
        ba  = result['boundary_window_accuracy'] or 0.0
        pf  = result['pure_window_fraction']
        gap = pa - ba
        print(f"\n  overall={oa:.4f}  "
              f"pure({pf:.0%})={pa:.4f}  "
              f"boundary({1-pf:.0%})={ba:.4f}  gap={gap:.4f}")

        all_results[variant_name] = result

    return all_results


# ============================================================
# Summary printer
# ============================================================

def print_summary(all_results: dict):
    print("\n" + "="*70)
    print("  FINAL SUMMARY — STREAMING ABLATION (StreamingFusionNet)")
    print("="*70)

    header = (f"\n  {'Variant':<38} {'Overall':^10} {'Pure':^10}"
              f" {'Boundary':^10} {'Gap':^8} {'Δ overall'}")
    print(header)
    print("  " + "-" * 80)

    baseline_acc = all_results.get(
        'Baseline (No Attention, No RPE)', {}
    ).get('overall_accuracy', 0.0)

    for variant_name, result in all_results.items():
        oa    = result['overall_accuracy']
        pa    = result['pure_window_accuracy']     or 0.0
        ba    = result['boundary_window_accuracy'] or 0.0
        pf    = result['pure_window_fraction']
        gap   = pa - ba
        delta = (oa - baseline_acc) * 100.0
        marker = "✅" if delta > 0.05 else ("⚠️" if abs(delta) <= 0.05 else "❌")
        print(
            f"  {variant_name:<38} {oa:.4f}     {pa:.4f}     "
            f"{ba:.4f}   {gap:.4f}  {marker} {delta:+.2f}%"
        )


# ============================================================
# Results saver
# ============================================================

def save_results(all_results: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    baseline_acc = all_results.get(
        'Baseline (No Attention, No RPE)', {}
    ).get('overall_accuracy', 0.0)

    for variant_name, result in all_results.items():
        oa  = result['overall_accuracy']
        rows.append({
            'variant':                           variant_name,
            'overall_accuracy':                  oa,
            'overall_balanced_accuracy':         result['overall_balanced_accuracy'],
            'pure_window_accuracy':              result['pure_window_accuracy'],
            'pure_window_balanced_accuracy':     result['pure_window_balanced_accuracy'],
            'pure_window_fraction':              result['pure_window_fraction'],
            'boundary_window_accuracy':          result['boundary_window_accuracy'],
            'boundary_window_balanced_accuracy': result['boundary_window_balanced_accuracy'],
            'boundary_window_fraction':          result['boundary_window_fraction'],
            'pure_boundary_gap': (
                (result['pure_window_accuracy'] or 0.0) -
                (result['boundary_window_accuracy'] or 0.0)
            ),
            'delta_vs_baseline_pct': (oa - baseline_acc) * 100.0,
        })

    df   = pd.DataFrame(rows)
    path = os.path.join(out_dir, 'streaming_ablation_results.csv')
    df.to_csv(path, index=False)
    print(f"\n  ✅  Results saved → {path}")
    return path


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Streaming ablation: Temporal Attention + RPE'
    )
    p.add_argument('--data-root', default=DEFAULT_DATA_ROOT,
                   help='Path to MobiAct Annotated Data directory')
    p.add_argument('--out-dir',   default=DEFAULT_OUT_DIR,
                   help='Output directory for CSV results')
    p.add_argument('--subjects', nargs='+', type=int, default=DEFAULT_SUBJECTS,
                   help='Subject IDs to include (default: 1..67)')
    p.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                   help='Training epochs per LOSO fold (default: 50)')
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "="*70)
    print("  STREAMING ABLATION: Temporal Attention + Relative Position Encoding")
    print("  Model: StreamingFusionNet (best streaming baseline)")
    print(f"  ADL codes : {ADL_CODES_11}")
    print(f"  Subjects  : {len(args.subjects)}  |  Epochs: {args.epochs}")
    print(f"  Data      : {args.data_root}")
    print("="*70)

    all_results = run_ablation(
        data_root=args.data_root,
        out_dir=args.out_dir,
        subjects=args.subjects,
        epochs=args.epochs,
    )

    print_summary(all_results)
    save_results(all_results, args.out_dir)
    print("\n✅  Streaming ablation complete.")


if __name__ == '__main__':
    main()
