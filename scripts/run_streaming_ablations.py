import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import build_streaming_dataset
from src.evaluation import run_streaming_evaluation, run_naive_classical

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'streaming_ablation')
os.makedirs(RESULTS_DIR, exist_ok=True)

ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
WINDOW_SAMPLES, STRIDE, SUB_BINS, PURITY_THRESHOLD = 200, 100, 5, 0.9

def main():
    print("\n" + "="*70)
    print("🔬 STREAMING ABLATION: Complete Feature Variants")
    print("="*70)

    # Define all feature variants to test
    feature_variants = {
        'Naive Classical': {
            'attention': False, 'rpe': False, 'fpn': False, 'kalman': False,
            'is_naive': True
        },
        'Baseline (Fusion)': {
            'attention': False, 'rpe': False, 'fpn': False, 'kalman': False,
            'is_naive': False
        },
        '+ Temporal Attention': {
            'attention': True, 'rpe': False, 'fpn': False, 'kalman': False,
            'is_naive': False
        },
        '+ RPE': {
            'attention': False, 'rpe': True, 'fpn': False, 'kalman': False,
            'is_naive': False
        },
        '+ Attention + RPE': {
            'attention': True, 'rpe': True, 'fpn': False, 'kalman': False,
            'is_naive': False
        },
        '+ FPN': {
            'attention': False, 'rpe': False, 'fpn': True, 'kalman': False,
            'is_naive': False
        },
        '+ Kalman': {
            'attention': False, 'rpe': False, 'fpn': False, 'kalman': True,
            'is_naive': False
        },
        '+ Attention + RPE + FPN': {
            'attention': True, 'rpe': True, 'fpn': True, 'kalman': False,
            'is_naive': False
        },
        '+ Attention + RPE + Kalman': {
            'attention': True, 'rpe': True, 'fpn': False, 'kalman': True,
            'is_naive': False
        },
    }

    all_results = {}

    # First, get baseline dataset dimensions
    print("\n📊 Building baseline dataset to get dimensions...")
    baseline_data = build_streaming_dataset(
        ADL_CODES_11, list(range(1, 68)),
        include_attention=False, include_rpe=False, include_fpn=False, include_kalman=False
    )
    
    if not baseline_data:
        print("❌ Error: Failed to generate baseline dataset. Check data_loader and features.")
        return

    sample = next(iter(baseline_data))
    per_bin_dim = baseline_data[sample][0].shape[2]
    flat_dim = baseline_data[sample][1].shape[1]
    l2i = {l: i for i, l in enumerate(ADL_CODES_11)}
    n_classes = len(ADL_CODES_11)
    print(f"  per_bin_dim: {per_bin_dim}, flat_dim: {flat_dim}")

    # Run each variant
    for variant_name, params in feature_variants.items():
        print(f"\n{'='*70}")
        print(f"📊 Building dataset for: {variant_name}")
        print(f"{'='*70}")
        
        data = build_streaming_dataset(
            ADL_CODES_11, list(range(1, 68)),
            include_attention=params['attention'],
            include_rpe=params['rpe'],
            include_fpn=params['fpn'],
            include_kalman=params['kalman']
        )
        
        print(f"  Subjects: {len(data)}")
        
        if params.get('is_naive', False):
            metrics = run_naive_classical(data, l2i, n_classes)
        else:
            # Get updated flat_dim if it changed
            sample = next(iter(data))
            new_flat_dim = data[sample][1].shape[1]
            if new_flat_dim != flat_dim:
                print(f"  flat_dim updated: {flat_dim} → {new_flat_dim}")
                flat_dim = new_flat_dim
            
            metrics = run_streaming_evaluation(
                data, l2i, n_classes, per_bin_dim, flat_dim,
                variant_name,
                params['attention'], params['rpe'],
                params['fpn'], params['kalman'],
                epochs=50, device=device
            )
        
        all_results[variant_name] = metrics

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)

    rows = []
    for variant_name, metrics in all_results.items():
        rows.append({
            'variant': variant_name,
            'overall_accuracy': metrics['overall_accuracy'],
            'overall_balanced_accuracy': metrics['overall_balanced_accuracy'],
            'pure_window_accuracy': metrics['pure_window_accuracy'],
            'pure_window_balanced_accuracy': metrics['pure_window_balanced_accuracy'],
            'pure_window_fraction': metrics['pure_window_fraction'],
            'boundary_window_accuracy': metrics['boundary_window_accuracy'],
            'boundary_window_balanced_accuracy': metrics['boundary_window_balanced_accuracy'],
            'boundary_window_fraction': metrics['boundary_window_fraction'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'{RESULTS_DIR}/streaming_ablation_results.csv', index=False)
    print(f"  ✅ Saved: {RESULTS_DIR}/streaming_ablation_results.csv")

    with open(f'{RESULTS_DIR}/streaming_ablation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY: STREAMING ABLATION")
    print("="*70)

    print(f"\n{'Variant':<35} {'Overall Acc':<12} {'Pure Acc':<12} {'Boundary Acc':<12} {'Gap':<10}")
    print(f"{'-'*85}")

    baseline_acc = all_results.get('Baseline (Fusion)', {}).get('overall_accuracy', 0)

    for variant_name, metrics in all_results.items():
        acc = metrics['overall_accuracy']
        pure = metrics['pure_window_accuracy']
        boundary = metrics['boundary_window_accuracy']
        
        pure_val = pure if pure is not None else 0
        boundary_val = boundary if boundary is not None else 0
        
        gap = pure_val - boundary_val
        improvement = (acc - baseline_acc) * 100
        marker = "✅" if improvement > 0 and variant_name != 'Naive Classical' else ("⚠️" if abs(improvement) < 0.1 else "❌")
        
        if variant_name == 'Naive Classical':
            marker = "📉"
        
        print(f"{variant_name:<35} {acc:.4f}      {pure_val:.4f}      {boundary_val:.4f}      {gap:.4f}  ({marker} {improvement:+.2f}%)")

    print("\n" + "="*70)
    print("✅ STREAMING ABLATION COMPLETE!")

if __name__ == "__main__":
    main()
