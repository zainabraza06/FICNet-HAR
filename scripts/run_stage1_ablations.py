import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import stats

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_segment
from src.features import build_binned_features, build_flat_features
from src.evaluation import run_named_model, print_metrics

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'stage1_all_ablations')
os.makedirs(RESULTS_DIR, exist_ok=True)

FALL_CODES = ['BSC', 'FKL', 'FOL', 'SDL']
ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
ALL_CODES = FALL_CODES + ADL_CODES_11
SEEDS = [0, 1, 2, 3, 4]
LABELS_S1 = ['ADL', 'FALL']

# ============================================================
# DATASET BUILDER
# ============================================================
def build_stage1_dataset(include_gyro=False, include_attention=False, include_rpe=False,
                         include_fpn=False, include_kalman=False):
    X_bins, X_flat, y, groups = [], [], [], []
    for code in ALL_CODES:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None: continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x','acc_y','acc_z']].values
            gyro = seg[['gyro_x','gyro_y','gyro_z']].values
            pitch = seg['pitch'].values if 'pitch' in seg.columns else np.zeros(len(seg))
            roll = seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg))
            
            fb = build_binned_features(acc, gyro, include_gyro=include_gyro)
            fc = build_flat_features(acc, gyro, pitch, roll, include_gyro=include_gyro,
                                     include_attention=include_attention,
                                     include_rpe=include_rpe,
                                     include_fpn=include_fpn,
                                     include_kalman=include_kalman)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append('FALL' if code in FALL_CODES else 'ADL')
                groups.append(subj)
    return np.array(X_bins), np.array(X_flat), np.array(y), np.array(groups)

def main():
    # ============================================================
    # ABLATION 1: MODEL COMPARISON
    # ============================================================
    print("\n" + "="*70)
    print("🔬 ABLATION 1: MODEL COMPARISON (Best Feature Set)")
    print("="*70)

    # Build best feature set (Profile+Stats+Attention+RPE)
    Xb, Xf, y, g = build_stage1_dataset(
        include_gyro=False,
        include_attention=True,
        include_rpe=True,
        include_fpn=False,
        include_kalman=False
    )

    model_results = {}
    for model in ['LDA', 'KNN-3', 'SVM-RBF', 'RandomForest', 'BiLSTM', 'Fusion']:
        seeds = [0] if model in ['LDA', 'KNN-3'] else SEEDS
        metrics = [run_named_model(model, Xb, Xf, y, g, LABELS_S1, s, device) for s in seeds]
        model_results[model] = metrics
        print_metrics(model, metrics)

    best_model = max(model_results, key=lambda m: np.mean([x['accuracy'] for x in model_results[m]]))
    print(f"\n>>> Best Model: {best_model}")

    # ============================================================
    # ABLATION 2: FEATURE SET ABLATION
    # ============================================================
    print("\n" + "="*70)
    print("🔬 ABLATION 2: FEATURE SET ABLATION")
    print("="*70)

    feature_configs = {
        'Profile Only': {
            'include_gyro': False, 'include_attention': False, 'include_rpe': False,
            'include_fpn': False, 'include_kalman': False
        },
        'Stats Only': {
            'include_gyro': False, 'include_attention': False, 'include_rpe': False,
            'include_fpn': False, 'include_kalman': False
        },
        'Full (Profile+Stats)': {
            'include_gyro': False, 'include_attention': False, 'include_rpe': False,
            'include_fpn': False, 'include_kalman': False
        },
        'Gyro Only': {
            'include_gyro': True, 'include_attention': False, 'include_rpe': False,
            'include_fpn': False, 'include_kalman': False
        },
        'Acc + Gyro (No Stats)': {
            'include_gyro': True, 'include_attention': False, 'include_rpe': False,
            'include_fpn': False, 'include_kalman': False
        },
        'Acc + Gyro (Full)': {
            'include_gyro': True, 'include_attention': False, 'include_rpe': False,
            'include_fpn': False, 'include_kalman': False
        },
        '+ Temporal Attention': {
            'include_gyro': False, 'include_attention': True, 'include_rpe': False,
            'include_fpn': False, 'include_kalman': False
        },
        '+ RPE': {
            'include_gyro': False, 'include_attention': False, 'include_rpe': True,
            'include_fpn': False, 'include_kalman': False
        },
        '+ Attention + RPE': {
            'include_gyro': False, 'include_attention': True, 'include_rpe': True,
            'include_fpn': False, 'include_kalman': False
        },
        '+ FPN': {
            'include_gyro': False, 'include_attention': False, 'include_rpe': False,
            'include_fpn': True, 'include_kalman': False
        },
        '+ Kalman': {
            'include_gyro': True, 'include_attention': False, 'include_rpe': False,
            'include_fpn': False, 'include_kalman': True
        },
        'Full + Attention + RPE + Gyro': {
            'include_gyro': True, 'include_attention': True, 'include_rpe': True,
            'include_fpn': False, 'include_kalman': False
        },
    }

    feature_results = {}
    for name, config in feature_configs.items():
        Xb, Xf, y, g = build_stage1_dataset(**config)
        if len(Xf) == 0:
            print(f"  {name:<35s}: SKIPPED (no data)")
            continue
        # Use best model from ablation 1
        m = run_named_model(best_model, Xb, Xf, y, g, LABELS_S1, 0, device)
        feature_results[name] = m
        print(f"  {name:<35s}: acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} f1={m['macro_f1']:.4f}")

    # ============================================================
    # ABLATION 3: STATISTICAL SIGNIFICANCE
    # ============================================================
    print("\n" + "="*70)
    print("🔬 ABLATION 3: STATISTICAL SIGNIFICANCE")
    print("="*70)

    # Compare best vs baseline
    baseline_accs = []
    best_accs = []
    for seed in SEEDS:
        # Baseline: Full (Profile+Stats)
        Xb_b, Xf_b, y_b, g_b = build_stage1_dataset(
            include_gyro=False, include_attention=False, include_rpe=False,
            include_fpn=False, include_kalman=False
        )
        m_b = run_named_model(best_model, Xb_b, Xf_b, y_b, g_b, LABELS_S1, seed, device)
        baseline_accs.append(m_b['accuracy'])
        
        # Best: + Attention + RPE
        Xb_best, Xf_best, y_best, g_best = build_stage1_dataset(
            include_gyro=False, include_attention=True, include_rpe=True,
            include_fpn=False, include_kalman=False
        )
        m_best = run_named_model(best_model, Xb_best, Xf_best, y_best, g_best, LABELS_S1, seed, device)
        best_accs.append(m_best['accuracy'])

    t_stat, t_p = stats.ttest_rel(best_accs, baseline_accs)
    print(f"\n  Baseline (Full): {np.mean(baseline_accs):.4f} ± {np.std(baseline_accs):.4f}")
    print(f"  Best (+Attention+RPE): {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
    print(f"  Paired t-test: t={t_stat:.4f}, p={t_p:.4f} {'** SIG **' if t_p < 0.05 else '(n.s.)'}")
    print(f"  Improvement: {(np.mean(best_accs)-np.mean(baseline_accs))*100:.2f}%")

    # ============================================================
    # ABLATION 4: GYROSCOPE CONTRIBUTION
    # ============================================================
    print("\n" + "="*70)
    print("🔬 ABLATION 4: GYROSCOPE CONTRIBUTION")
    print("="*70)

    # Acc-only (best)
    Xb_acc, Xf_acc, y_acc, g_acc = build_stage1_dataset(
        include_gyro=False, include_attention=True, include_rpe=True,
        include_fpn=False, include_kalman=False
    )
    m_acc = run_named_model(best_model, Xb_acc, Xf_acc, y_acc, g_acc, LABELS_S1, 0, device)

    # Acc + Gyro
    Xb_gyro, Xf_gyro, y_gyro, g_gyro = build_stage1_dataset(
        include_gyro=True, include_attention=True, include_rpe=True,
        include_fpn=False, include_kalman=False
    )
    m_gyro = run_named_model(best_model, Xb_gyro, Xf_gyro, y_gyro, g_gyro, LABELS_S1, 0, device)

    print(f"\n  Acc-Only: {m_acc['accuracy']:.4f}")
    print(f"  Acc + Gyro: {m_gyro['accuracy']:.4f}")
    print(f"  Gyroscope Contribution: {(m_gyro['accuracy'] - m_acc['accuracy'])*100:+.2f}%")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)

    results_output = {
        'model_comparison': {k: [{'accuracy': m['accuracy'], 'balanced': m['balanced_accuracy'], 'f1': m['macro_f1']} 
                                 for m in v] for k, v in model_results.items()},
        'feature_ablation': {k: {'accuracy': v['accuracy'], 'balanced': v['balanced_accuracy'], 'f1': v['macro_f1']} 
                             for k, v in feature_results.items()},
        'best_model': best_model,
        'statistical_significance': {'t_stat': t_stat, 'p_value': t_p, 'significant': t_p < 0.05},
        'gyroscope_contribution': {'acc_only': m_acc['accuracy'], 'acc_gyro': m_gyro['accuracy'], 
                                   'difference': m_gyro['accuracy'] - m_acc['accuracy']}
    }

    with open(f'{RESULTS_DIR}/stage1_complete_ablations.json', 'w') as f:
        json.dump(results_output, f, indent=2, default=str)

    # Summary table
    rows = []
    for name, metrics in feature_results.items():
        rows.append({
            'feature_set': name,
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'f1_score': metrics['macro_f1']
        })
    pd.DataFrame(rows).to_csv(f'{RESULTS_DIR}/stage1_feature_ablation.csv', index=False)

    # Model summary
    model_rows = []
    for model, metrics in model_results.items():
        accs = [m['accuracy'] for m in metrics]
        bals = [m['balanced_accuracy'] for m in metrics]
        f1s = [m['macro_f1'] for m in metrics]
        model_rows.append({
            'model': model,
            'accuracy': np.mean(accs),
            'accuracy_std': np.std(accs),
            'balanced_accuracy': np.mean(bals),
            'f1_score': np.mean(f1s),
            'n_seeds': len(metrics)
        })
    pd.DataFrame(model_rows).to_csv(f'{RESULTS_DIR}/stage1_model_comparison.csv', index=False)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("📊 STAGE 1 — FINAL SUMMARY")
    print("="*70)

    print(f"\n  Best Model: {best_model}")
    print(f"\n  Feature Set Rankings:")
    print(f"    {'Rank':<5} {'Feature Set':<35} {'Accuracy':<12}")
    print(f"    {'-'*55}")
    sorted_features = sorted(feature_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_features, 1):
        print(f"    {i:<5} {name:<35} {metrics['accuracy']:.4f}")

    print(f"\n  Gyroscope Contribution: {(m_gyro['accuracy'] - m_acc['accuracy'])*100:+.2f}%")
    print(f"\n  Statistical Significance (Best vs Baseline): p={t_p:.4f} {'(SIG)' if t_p < 0.05 else '(n.s.)'}")

    print(f"\n✅ Results saved to: {RESULTS_DIR}")
    print("="*70)
    print("✅ STAGE 1 COMPLETE")

if __name__ == "__main__":
    main()
