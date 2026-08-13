import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support

# Add project root to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import build_stage1_dataset, build_stage2a_dataset, build_stage2b_dataset
from src.evaluation import run_loso_svm_with_proba, run_loso_rf, get_stage2a_predictions

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'hierarchical')
os.makedirs(RESULTS_DIR, exist_ok=True)

FALL_CODES = ['BSC', 'FKL', 'FOL', 'SDL']
ADL_CODES_11 = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO']
ALL_CODES = FALL_CODES + ADL_CODES_11
CONFIDENCE_THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

def evaluate_hierarchical_with_confidence(s1_true, s1_pred, s1_proba, s1_acc, s1_bal, s1_f1,
                                          s2a_true, s2a_pred, s2a_acc, s2a_bal, s2a_f1,
                                          s2b_true, s2b_pred, s2b_acc, s2b_bal, s2b_f1,
                                          fall_fraction, adl_fraction, confidence_threshold=None):
    print(f"\n{'='*70}")
    if confidence_threshold is None:
        print("🏆 HIERARCHICAL SYSTEM — WITHOUT CONFIDENCE THRESHOLD")
    else:
        print(f"🏆 HIERARCHICAL SYSTEM — WITH CONFIDENCE THRESHOLD (≥ {confidence_threshold:.0%})")
    print("="*70)

    p_stage1_correct = s1_acc
    p_stage2a_correct = s2a_acc
    p_stage2b_correct = s2b_acc

    if confidence_threshold is not None:
        s1_proba = np.array(s1_proba)
        high_conf_mask = s1_proba >= confidence_threshold
        low_conf_mask = ~high_conf_mask
        
        high_conf_fraction = high_conf_mask.mean()
        low_conf_fraction = low_conf_mask.mean()
        
        print(f"\n  High confidence samples (≥ {confidence_threshold:.0%}): {high_conf_fraction:.1%}")
        print(f"  Low confidence samples (< {confidence_threshold:.0%}): {low_conf_fraction:.1%}")
        
        s1_high_acc = accuracy_score(np.array(s1_true)[high_conf_mask], np.array(s1_pred)[high_conf_mask]) if high_conf_mask.sum() > 0 else s1_acc
        s1_low_acc = accuracy_score(np.array(s1_true)[low_conf_mask], np.array(s1_pred)[low_conf_mask]) if low_conf_mask.sum() > 0 else s1_acc
        
        print(f"  Stage 1 accuracy (high confidence): {s1_high_acc:.4f}")
        print(f"  Stage 1 accuracy (low confidence): {s1_low_acc:.4f}")
        
        high_conf_hierarchical_fall = s1_high_acc * p_stage2a_correct
        high_conf_hierarchical_adl = s1_high_acc * p_stage2b_correct
        high_conf_hierarchical = (fall_fraction * high_conf_hierarchical_fall) + (adl_fraction * high_conf_hierarchical_adl)
        
        low_conf_hierarchical = s1_low_acc
        
        hierarchical_acc = (high_conf_fraction * high_conf_hierarchical) + (low_conf_fraction * low_conf_hierarchical)
        
        print(f"\n  High confidence hierarchical accuracy: {high_conf_hierarchical:.4f}")
        print(f"  Low confidence (binary only) accuracy: {low_conf_hierarchical:.4f}")
        print(f"  ★ HIERARCHICAL SYSTEM ACCURACY: {hierarchical_acc:.4f} ★")
        
    else:
        hierarchical_fall_acc = p_stage1_correct * p_stage2a_correct
        hierarchical_adl_acc = p_stage1_correct * p_stage2b_correct
        hierarchical_acc = (fall_fraction * hierarchical_fall_acc) + (adl_fraction * hierarchical_adl_acc)
        
        print(f"\n  Hierarchical Accuracy (Falls): {hierarchical_fall_acc:.4f}")
        print(f"  Hierarchical Accuracy (ADLs): {hierarchical_adl_acc:.4f}")
        print(f"  ★ HIERARCHICAL SYSTEM ACCURACY: {hierarchical_acc:.4f} ★")
    
    return {
        'stage1': {'accuracy': s1_acc, 'balanced': s1_bal, 'f1': s1_f1},
        'stage2a': {'accuracy': s2a_acc, 'balanced': s2a_bal, 'f1': s2a_f1},
        'stage2b': {'accuracy': s2b_acc, 'balanced': s2b_bal, 'f1': s2b_f1},
        'hierarchical_accuracy': hierarchical_acc,
        'fall_fraction': fall_fraction,
        'adl_fraction': adl_fraction,
        'confidence_threshold': confidence_threshold,
        'high_confidence_fraction': high_conf_fraction if confidence_threshold is not None else 1.0,
        'low_confidence_fraction': low_conf_fraction if confidence_threshold is not None else 0.0,
    }

def main():
    print("\n" + "="*70)
    print("🔬 HIERARCHICAL SYSTEM EVALUATION")
    print("Confidence Threshold Ablation")
    print("="*70)

    # 1. Build Datasets
    print("\n  Building datasets...")
    Xb_s1, Xf_s1, y_s1, g_s1 = build_stage1_dataset(ALL_CODES, FALL_CODES)
    Xb_s2a, Xf_s2a, y_s2a, g_s2a = build_stage2a_dataset(FALL_CODES)
    Xb_s2b, Xf_s2b, y_s2b, g_s2b = build_stage2b_dataset(ADL_CODES_11)
    
    print(f"    Stage 1: {len(y_s1)} samples, {len(set(y_s1))} classes")
    print(f"    Stage 2a: {len(y_s2a)} samples, {len(set(y_s2a))} classes")
    print(f"    Stage 2b: {len(y_s2b)} samples, {len(set(y_s2b))} classes")
    
    fall_fraction = len(y_s2a) / len(y_s1)
    adl_fraction = 1 - fall_fraction

    # 2. Run Base Evaluations
    print("\n" + "="*70)
    print("🔴 STAGE 1: Fall Gate (Binary Fall vs ADL)")
    print("="*70)
    print("  Running LOSO evaluation...")
    s1_true, s1_pred, s1_proba = run_loso_svm_with_proba(Xf_s1, y_s1, g_s1, seed=0)
    s1_acc = accuracy_score(s1_true, s1_pred)
    s1_bal = balanced_accuracy_score(s1_true, s1_pred)
    s1_prec, s1_rec, s1_f1, _ = precision_recall_fscore_support(s1_true, s1_pred, average='macro')
    print(f"    Accuracy: {s1_acc:.4f}  |  Balanced Acc: {s1_bal:.4f}  |  F1-Score: {s1_f1:.4f}")

    print("\n" + "="*70)
    print("🟡 STAGE 2a: Fall Subtypes (4-class: BSC, FKL, FOL, SDL)")
    print("="*70)
    print("  Running LOSO evaluation on actual falls...")
    s2a_true, s2a_pred = get_stage2a_predictions(Xf_s2a, y_s2a, g_s2a, seed=0)
    s2a_acc = accuracy_score(s2a_true, s2a_pred)
    s2a_bal = balanced_accuracy_score(s2a_true, s2a_pred)
    s2a_prec, s2a_rec, s2a_f1, _ = precision_recall_fscore_support(s2a_true, s2a_pred, average='macro')
    print(f"    Accuracy: {s2a_acc:.4f}  |  Balanced Acc: {s2a_bal:.4f}  |  F1-Score: {s2a_f1:.4f}")

    print("\n" + "="*70)
    print("🟢 STAGE 2b: ADL Classification (11-class)")
    print("="*70)
    print("  Running LOSO evaluation on ADLs...")
    s2b_true, s2b_pred = run_loso_rf(Xf_s2b, y_s2b, g_s2b, seed=0)
    s2b_acc = accuracy_score(s2b_true, s2b_pred)
    s2b_bal = balanced_accuracy_score(s2b_true, s2b_pred)
    s2b_prec, s2b_rec, s2b_f1, _ = precision_recall_fscore_support(s2b_true, s2b_pred, average='macro')
    print(f"    Accuracy: {s2b_acc:.4f}  |  Balanced Acc: {s2b_bal:.4f}  |  F1-Score: {s2b_f1:.4f}")

    # 3. Confidence Evaluation
    all_results = {}
    
    # Without threshold
    results_no_threshold = evaluate_hierarchical_with_confidence(
        s1_true, s1_pred, s1_proba, s1_acc, s1_bal, s1_f1,
        s2a_true, s2a_pred, s2a_acc, s2a_bal, s2a_f1,
        s2b_true, s2b_pred, s2b_acc, s2b_bal, s2b_f1,
        fall_fraction, adl_fraction, confidence_threshold=None
    )
    all_results['no_threshold'] = results_no_threshold

    # With thresholds
    threshold_results = {}
    for thresh in CONFIDENCE_THRESHOLDS:
        results = evaluate_hierarchical_with_confidence(
            s1_true, s1_pred, s1_proba, s1_acc, s1_bal, s1_f1,
            s2a_true, s2a_pred, s2a_acc, s2a_bal, s2a_f1,
            s2b_true, s2b_pred, s2b_acc, s2b_bal, s2b_f1,
            fall_fraction, adl_fraction, confidence_threshold=thresh
        )
        threshold_results[f'threshold_{int(thresh*100)}'] = results

    all_results['with_thresholds'] = threshold_results

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    print("\n" + "="*70)
    print("💾 SAVING RESULTS")
    print("="*70)

    with open(f'{RESULTS_DIR}/hierarchical_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    rows = []
    rows.append({
        'threshold': 'None',
        'hierarchical_accuracy': results_no_threshold['hierarchical_accuracy'],
        'stage1_accuracy': results_no_threshold['stage1']['accuracy'],
        'stage2a_accuracy': results_no_threshold['stage2a']['accuracy'],
        'stage2b_accuracy': results_no_threshold['stage2b']['accuracy'],
        'high_confidence_fraction': 1.0,
        'low_confidence_fraction': 0.0,
    })

    for thresh, results in threshold_results.items():
        threshold_value = int(thresh.split('_')[1])
        rows.append({
            'threshold': f'{threshold_value}%',
            'hierarchical_accuracy': results['hierarchical_accuracy'],
            'stage1_accuracy': results['stage1']['accuracy'],
            'stage2a_accuracy': results['stage2a']['accuracy'],
            'stage2b_accuracy': results['stage2b']['accuracy'],
            'high_confidence_fraction': results['high_confidence_fraction'],
            'low_confidence_fraction': results['low_confidence_fraction'],
        })

    pd.DataFrame(rows).to_csv(f'{RESULTS_DIR}/hierarchical_ablation_results.csv', index=False)
    
    print("\n" + "="*70)
    print("✅ HIERARCHICAL SYSTEM COMPLETE")

if __name__ == "__main__":
    main()
