import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import get_segment
from src.features import build_binned_features, build_flat_features
from src.evaluation import run_named_model, print_metrics
from src.models import get_classical_models

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'stage2b_all_ablations')
os.makedirs(RESULTS_DIR, exist_ok=True)

ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
SEEDS = [0, 1, 2, 3, 4]
LABELS_S2B = sorted(ADL_CODES_11)
PRIORS_2B = np.ones(11) / 11

# ============================================================
# DATASET BUILDER
# ============================================================
def build_stage2b_dataset(include_gyro=True, include_orient=False, 
                          include_spectral=True,
                          include_attention=False, include_rpe=False,
                          include_fpn=False, include_kalman=False):
    X_bins, X_flat, y, groups = [], [], [], []
    for code in ADL_CODES_11:
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
                                     include_orient=include_orient,
                                     include_spectral=include_spectral,
                                     include_attention=include_attention,
                                     include_rpe=include_rpe,
                                     include_fpn=include_fpn,
                                     include_kalman=include_kalman,
                                     include_fall_specific=False)
            if fb is not None and fc is not None:
                X_bins.append(fb); X_flat.append(fc)
                y.append(code); groups.append(subj)
    return np.array(X_bins), np.array(X_flat), np.array(y), np.array(groups)

def main():
    # ============================================================
    # ABLATION 1: MODEL COMPARISON (Best Feature Set)
    # ============================================================
    print("\n" + "="*70)
    print("🔬 ABLATION 1: MODEL COMPARISON (Best Feature Set)")
    print("="*70)

    # Build best feature set (Profile+Spectral+Gyro+Attention+RPE)
    Xb, Xf, y, g = build_stage2b_dataset(
        include_gyro=True,
        include_orient=False,
        include_spectral=True,
        include_attention=True,
        include_rpe=True,
        include_fpn=False,
        include_kalman=False
    )

    print(f"  Dataset: {len(y)} samples, {len(set(y))} classes")

    model_results = {}
    for model in ['LDA', 'KNN-3', 'SVM-RBF', 'RandomForest', 'BiLSTM', 'Fusion']:
        seeds = [0] if model in ['LDA', 'KNN-3'] else SEEDS
        metrics = [run_named_model(model, Xb, Xf, y, g, LABELS_S2B, s, device, epochs=150, priors=PRIORS_2B, lstm_hidden=16) for s in seeds]
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
            'include_gyro': False, 'include_orient': False, 'include_spectral': False,
            'include_attention': False, 'include_rpe': False, 'include_fpn': False, 'include_kalman': False
        },
        'Profile + Spectral': {
            'include_gyro': False, 'include_orient': False, 'include_spectral': True,
            'include_attention': False, 'include_rpe': False, 'include_fpn': False, 'include_kalman': False
        },
        'Profile + Spectral + Gyro': {
            'include_gyro': True, 'include_orient': False, 'include_spectral': True,
            'include_attention': False, 'include_rpe': False, 'include_fpn': False, 'include_kalman': False
        },
        'Full (With Orientation)': {
            'include_gyro': True, 'include_orient': True, 'include_spectral': True,
            'include_attention': False, 'include_rpe': False, 'include_fpn': False, 'include_kalman': False
        },
        '+ Temporal Attention': {
            'include_gyro': True, 'include_orient': False, 'include_spectral': True,
            'include_attention': True, 'include_rpe': False, 'include_fpn': False, 'include_kalman': False
        },
        '+ RPE': {
            'include_gyro': True, 'include_orient': False, 'include_spectral': True,
            'include_attention': False, 'include_rpe': True, 'include_fpn': False, 'include_kalman': False
        },
        '+ Attention + RPE': {
            'include_gyro': True, 'include_orient': False, 'include_spectral': True,
            'include_attention': True, 'include_rpe': True, 'include_fpn': False, 'include_kalman': False
        },
        '+ FPN': {
            'include_gyro': True, 'include_orient': False, 'include_spectral': True,
            'include_attention': False, 'include_rpe': False, 'include_fpn': True, 'include_kalman': False
        },
        '+ Kalman': {
            'include_gyro': True, 'include_orient': False, 'include_spectral': True,
            'include_attention': False, 'include_rpe': False, 'include_fpn': False, 'include_kalman': True
        },
    }

    feature_results = {}
    for name, config in feature_configs.items():
        Xb, Xf, y, g = build_stage2b_dataset(**config)
        if len(Xf) == 0:
            print(f"  {name:<35s}: SKIPPED (no data)")
            continue
        m = run_named_model(best_model, Xb, Xf, y, g, LABELS_S2B, 0, device, epochs=150, priors=PRIORS_2B, lstm_hidden=16)
        feature_results[name] = m
        print(f"  {name:<35s}: acc={m['accuracy']:.4f} bal={m['balanced_accuracy']:.4f} f1={m['macro_f1']:.4f}")

    # ============================================================
    # ABLATION 3: STATISTICAL SIGNIFICANCE
    # ============================================================
    print("\n" + "="*70)
    print("🔬 ABLATION 3: STATISTICAL SIGNIFICANCE")
    print("="*70)

    baseline_accs = []
    best_accs = []
    for seed in SEEDS:
        Xb_b, Xf_b, y_b, g_b = build_stage2b_dataset(
            include_gyro=True, include_orient=False, include_spectral=True,
            include_attention=False, include_rpe=False, include_fpn=False, include_kalman=False
        )
        m_b = run_named_model(best_model, Xb_b, Xf_b, y_b, g_b, LABELS_S2B, seed, device, epochs=150, priors=PRIORS_2B, lstm_hidden=16)
        baseline_accs.append(m_b['accuracy'])
        
        Xb_best, Xf_best, y_best, g_best = build_stage2b_dataset(
            include_gyro=True, include_orient=False, include_spectral=True,
            include_attention=True, include_rpe=True, include_fpn=False, include_kalman=False
        )
        m_best = run_named_model(best_model, Xb_best, Xf_best, y_best, g_best, LABELS_S2B, seed, device, epochs=150, priors=PRIORS_2B, lstm_hidden=16)
        best_accs.append(m_best['accuracy'])

    t_stat, t_p = stats.ttest_rel(best_accs, baseline_accs)
    print(f"\n  Baseline (Profile+Spectral+Gyro): {np.mean(baseline_accs):.4f} ± {np.std(baseline_accs):.4f}")
    print(f"  Best (+Attention+RPE): {np.mean(best_accs):.4f} ± {np.std(best_accs):.4f}")
    print(f"  Paired t-test: t={t_stat:.4f}, p={t_p:.4f} {'** SIG **' if t_p < 0.05 else '(n.s.)'}")
    print(f"  Improvement: {(np.mean(best_accs)-np.mean(baseline_accs))*100:.2f}%")

    # ============================================================
    # ABLATION 4: GYROSCOPE CONTRIBUTION
    # ============================================================
    print("\n" + "="*70)
    print("🔬 ABLATION 4: GYROSCOPE CONTRIBUTION")
    print("="*70)

    Xb_no, Xf_no, y_no, g_no = build_stage2b_dataset(
        include_gyro=False, include_orient=False, include_spectral=True,
        include_attention=True, include_rpe=True, include_fpn=False, include_kalman=False
    )
    m_no = run_named_model(best_model, Xb_no, Xf_no, y_no, g_no, LABELS_S2B, 0, device, epochs=150, priors=PRIORS_2B, lstm_hidden=16)

    Xb_yes, Xf_yes, y_yes, g_yes = build_stage2b_dataset(
        include_gyro=True, include_orient=False, include_spectral=True,
        include_attention=True, include_rpe=True, include_fpn=False, include_kalman=False
    )
    m_yes = run_named_model(best_model, Xb_yes, Xf_yes, y_yes, g_yes, LABELS_S2B, 0, device, epochs=150, priors=PRIORS_2B, lstm_hidden=16)

    print(f"\n  Without Gyro: {m_no['accuracy']:.4f}")
    print(f"  With Gyro: {m_yes['accuracy']:.4f}")
    print(f"  Gyroscope Contribution: {(m_yes['accuracy'] - m_no['accuracy'])*100:+.2f}%")

    # ============================================================
    # ABLATION 5: ORIENTATION CONTRIBUTION
    # ============================================================
    print("\n" + "="*70)
    print("🔬 ABLATION 5: ORIENTATION CONTRIBUTION")
    print("="*70)

    Xb_no_orient, Xf_no_orient, y_no_orient, g_no_orient = build_stage2b_dataset(
        include_gyro=True, include_orient=False, include_spectral=True,
        include_attention=True, include_rpe=True, include_fpn=False, include_kalman=False
    )
    m_no_orient = run_named_model(best_model, Xb_no_orient, Xf_no_orient, y_no_orient, g_no_orient, 
                                  LABELS_S2B, 0, device, epochs=150, priors=PRIORS_2B, lstm_hidden=16)

    Xb_orient, Xf_orient, y_orient, g_orient = build_stage2b_dataset(
        include_gyro=True, include_orient=True, include_spectral=True,
        include_attention=True, include_rpe=True, include_fpn=False, include_kalman=False
    )
    m_orient = run_named_model(best_model, Xb_orient, Xf_orient, y_orient, g_orient, 
                               LABELS_S2B, 0, device, epochs=150, priors=PRIORS_2B, lstm_hidden=16)

    print(f"\n  Without Orientation: {m_no_orient['accuracy']:.4f}")
    print(f"  With Orientation: {m_orient['accuracy']:.4f}")
    print(f"  Orientation Contribution: {(m_orient['accuracy'] - m_no_orient['accuracy'])*100:+.2f}%")

    # ============================================================
    # ABLATION 6: PER-CLASS PERFORMANCE
    # ============================================================
    print("\n" + "="*70)
    print("🔬 ABLATION 6: PER-CLASS PERFORMANCE")
    print("="*70)

    Xb_best, Xf_best, y_best, g_best = build_stage2b_dataset(
        include_gyro=True, include_orient=False, include_spectral=True,
        include_attention=True, include_rpe=True, include_fpn=False, include_kalman=False
    )

    logo = LeaveOneGroupOut()
    all_true, all_pred = [], []
    for tr, te in logo.split(Xf_best, y_best, g_best):
        if len(set(y_best[tr])) < 2: continue
        sc = StandardScaler().fit(Xf_best[tr])
        Xtr, Xte = sc.transform(Xf_best[tr]), sc.transform(Xf_best[te])
        clf = get_classical_models(0, priors=PRIORS_2B)[best_model]
        clf.fit(Xtr, y_best[tr])
        pred = clf.predict(Xte)
        all_true.extend(y_best[te]); all_pred.extend(pred)

    print("\n  Classification Report:")
    print(classification_report(all_true, all_pred, target_names=LABELS_S2B))

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
        'gyroscope_contribution': {'without_gyro': m_no['accuracy'], 'with_gyro': m_yes['accuracy'], 
                                   'difference': m_yes['accuracy'] - m_no['accuracy']},
        'orientation_contribution': {'without_orient': m_no_orient['accuracy'], 'with_orient': m_orient['accuracy'],
                                     'difference': m_orient['accuracy'] - m_no_orient['accuracy']}
    }

    with open(f'{RESULTS_DIR}/stage2b_complete_ablations.json', 'w') as f:
        json.dump(results_output, f, indent=2, default=str)

    rows = []
    for name, metrics in feature_results.items():
        rows.append({
            'feature_set': name,
            'accuracy': metrics['accuracy'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'f1_score': metrics['macro_f1']
        })
    pd.DataFrame(rows).to_csv(f'{RESULTS_DIR}/stage2b_feature_ablation.csv', index=False)

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
    pd.DataFrame(model_rows).to_csv(f'{RESULTS_DIR}/stage2b_model_comparison.csv', index=False)

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print("📊 STAGE 2b — FINAL SUMMARY")
    print("="*70)

    print(f"\n  Best Model: {best_model}")
    print(f"\n  Feature Set Rankings:")
    print(f"    {'Rank':<5} {'Feature Set':<35} {'Accuracy':<12}")
    print(f"    {'-'*55}")
    sorted_features = sorted(feature_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for i, (name, metrics) in enumerate(sorted_features, 1):
        print(f"    {i:<5} {name:<35} {metrics['accuracy']:.4f}")

    print(f"\n  Gyroscope Contribution: {(m_yes['accuracy'] - m_no['accuracy'])*100:+.2f}%")
    print(f"  Orientation Contribution: {(m_orient['accuracy'] - m_no_orient['accuracy'])*100:+.2f}%")
    print(f"\n  Statistical Significance (Best vs Baseline): p={t_p:.4f} {'(SIG)' if t_p < 0.05 else '(n.s.)'}")

    print(f"\n  Per-Class Performance (Best Model):")
    unique_classes = sorted(set(all_true))
    class_accs = {}
    for cls in unique_classes:
        idx = [i for i, c in enumerate(all_true) if c == cls]
        if idx:
            cls_acc = np.mean(np.array(all_pred)[idx] == cls)
            class_accs[cls] = cls_acc
            print(f"    {cls}: {cls_acc:.4f}")

    worst_classes = sorted(class_accs.items(), key=lambda x: x[1])[:3]
    print(f"\n  Worst Performing Classes:")
    for cls, acc in worst_classes:
        print(f"    {cls}: {acc:.4f}")

    print(f"\n✅ Results saved to: {RESULTS_DIR}")
    print("="*70)
    print("✅ STAGE 2b COMPLETE")

if __name__ == "__main__":
    main()
