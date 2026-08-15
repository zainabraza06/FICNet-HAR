import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.subplots as plt_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

from src.config import PROJECT_ROOT, RESULTS_DIR_HIERARCHICAL, ALL_CODES, FALL_CODES, ADL_CODES_11, LABELS_S1, LABELS_S2A, LABELS_S2B, LABELS_ALL_15, SEEDS
from src.features.extractors import build_binned_features, build_flat_features
from src.data.loader import get_segment
from src.models.deep import train_fusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

RESTRICT_TO_COMPLETE_SUBJECTS = False

def get_stage1_svm(seed):
    return SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=seed)

def predict_fusion(model, Xb, Xf):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xb, dtype=torch.float32).to(device),
                      torch.tensor(Xf, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
    return pred

def main():
    print("\n"+"="*70); print("PART 1: BUILDING UNIFIED DATASET"); print("="*70)

    Xf_s1_list, Xf_s2a_list, Xb_s2a_list, Xf_s2b_list, Xb_s2b_list = [], [], [], [], []
    y_leaf, y_binary, groups = [], [], []

    # Final feature sets per stage
    s1_groups = ['profile', 'stats', 'rpe']
    s2a_groups = ['profile', 'stats', 'fall_specific']
    s2b_groups = ['profile', 'stats', 'rpe', 'gyro']

    for code in ALL_CODES:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None: continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x','acc_y','acc_z']].values
            gyro = seg[['gyro_x','gyro_y','gyro_z']].values if 'gyro_x' in seg.columns else None
            roll = seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg))
            if len(acc) < 20: continue

            # Build binned features
            Xb_acc = build_binned_features(acc, gyro_data=None, n_bins=5, include_gyro=False)
            Xb_gyro = build_binned_features(acc, gyro_data=gyro, n_bins=5, include_gyro=True)
            if Xb_acc is None or Xb_gyro is None: continue

            # Build flat features using modular extractors
            fc_s1 = build_flat_features(acc, groups=s1_groups, stage='s1')
            fc_s2a = build_flat_features(acc, groups=s2a_groups, stage='s2a')
            fc_s2b = build_flat_features(acc, gyro_data=gyro, roll_data=roll, groups=s2b_groups, stage='s2b')
            
            if fc_s1 is None or fc_s2a is None or fc_s2b is None: continue

            Xf_s1_list.append(fc_s1)
            Xf_s2a_list.append(fc_s2a)
            Xb_s2a_list.append(Xb_acc)
            Xf_s2b_list.append(fc_s2b)
            Xb_s2b_list.append(Xb_gyro)

            y_leaf.append(code)
            y_binary.append('FALL' if code in FALL_CODES else 'ADL')
            groups.append(subj)

    Xf_s1 = np.array(Xf_s1_list); Xf_s2a = np.array(Xf_s2a_list); Xb_s2a = np.array(Xb_s2a_list)
    Xf_s2b = np.array(Xf_s2b_list); Xb_s2b = np.array(Xb_s2b_list)
    y_leaf = np.array(y_leaf); y_binary = np.array(y_binary); groups = np.array(groups)

    print(f"Total samples: {len(y_leaf)}, subjects: {len(set(groups))}")
    print(f"  Xf_s1: {Xf_s1.shape}  Xf_s2a: {Xf_s2a.shape} Xb_s2a: {Xb_s2a.shape}")
    print(f"  Xf_s2b: {Xf_s2b.shape} Xb_s2b: {Xb_s2b.shape}")

    print("\n"+"="*70); print("PART 1b: SUBJECT COVERAGE DIAGNOSTIC"); print("="*70)

    subj_ids = sorted(set(groups))
    coverage = []
    for s in subj_ids:
        mask = groups == s
        n_total = mask.sum()
        n_fall = ((groups == s) & (y_binary == 'FALL')).sum()
        n_adl = ((groups == s) & (y_binary == 'ADL')).sum()
        n_leaf_types = len(set(y_leaf[mask]))
        coverage.append({'subject': int(s), 'n_total': int(n_total), 'n_fall': int(n_fall),
                          'n_adl': int(n_adl), 'n_activity_types': int(n_leaf_types)})

    cov_df = pd.DataFrame(coverage)
    print(f"  Subjects with data at all: {len(cov_df)} / 67")
    print(f"  Subjects with ZERO fall samples: {(cov_df['n_fall']==0).sum()}")
    print(f"  Subjects with ZERO ADL samples: {(cov_df['n_adl']==0).sum()}")
    print(f"  Subjects with all 15 activity types present: {(cov_df['n_activity_types']==15).sum()}")
    print(f"  Min / median / max activity types per subject: {cov_df['n_activity_types'].min()} / {cov_df['n_activity_types'].median():.0f} / {cov_df['n_activity_types'].max()}")

    incomplete = cov_df[cov_df['n_activity_types'] < 15]
    if len(incomplete) > 0:
        print(f"\n  {len(incomplete)} subject(s) missing at least one activity type:")
        print(incomplete.to_string(index=False))
    else:
        print("\n  All subjects have complete coverage across all 15 activity types.")

    cov_df.to_csv(f'{RESULTS_DIR_HIERARCHICAL}/subject_coverage.csv', index=False)
    print(f"\n  Saved: {RESULTS_DIR_HIERARCHICAL}/subject_coverage.csv")

    if RESTRICT_TO_COMPLETE_SUBJECTS:
        complete_subjects = set(cov_df[cov_df['n_activity_types'] == 15]['subject'])
        keep_mask = np.array([g in complete_subjects for g in groups])
        n_before = len(groups)
        Xf_s1, Xf_s2a, Xb_s2a = Xf_s1[keep_mask], Xf_s2a[keep_mask], Xb_s2a[keep_mask]
        Xf_s2b, Xb_s2b = Xf_s2b[keep_mask], Xb_s2b[keep_mask]
        y_leaf, y_binary, groups = y_leaf[keep_mask], y_binary[keep_mask], groups[keep_mask]
        print(f"\n  RESTRICT_TO_COMPLETE_SUBJECTS=True: kept {len(groups)}/{n_before} samples across {len(complete_subjects)} complete subjects.")
    else:
        print(f"\n  RESTRICT_TO_COMPLETE_SUBJECTS=False: using all {len(set(groups))} subjects as-is.")

    print("\n"+"="*70); print("PART 2: HIERARCHICAL LOSO"); print("="*70)

    s2a_l2i = {l:i for i,l in enumerate(LABELS_S2A)}
    s2b_l2i = {l:i for i,l in enumerate(LABELS_S2B)}
    logo = LeaveOneGroupOut()

    seed_results = []

    for seed in SEEDS:
        print(f"\n  -- Seed {seed} --")
        all_true_leaf, all_pred_leaf = [], []
        all_true_binary, all_pred_binary = [], []
        correctly_routed_fall_true, correctly_routed_fall_pred = [], []
        correctly_routed_adl_true, correctly_routed_adl_pred = [], []

        fold_idx = 0
        for tr_idx, te_idx in logo.split(Xf_s1, y_binary, groups):
            fold_idx += 1
            if len(set(y_binary[tr_idx])) < 2:
                continue

            # Stage 1
            sc1 = StandardScaler().fit(Xf_s1[tr_idx])
            clf1 = get_stage1_svm(seed)
            clf1.fit(sc1.transform(Xf_s1[tr_idx]), y_binary[tr_idx])
            pred1 = clf1.predict(sc1.transform(Xf_s1[te_idx]))

            # Stage 2a
            fall_train_mask = y_binary[tr_idx] == 'FALL'
            tr_fall_idx = tr_idx[fall_train_mask]
            if len(set(y_leaf[tr_fall_idx])) < 2:
                continue
            scb_2a = StandardScaler().fit(Xb_s2a[tr_fall_idx].reshape(-1, 5))
            scf_2a = StandardScaler().fit(Xf_s2a[tr_fall_idx])
            Xb_tr_2a = scb_2a.transform(Xb_s2a[tr_fall_idx].reshape(-1, 5)).reshape(Xb_s2a[tr_fall_idx].shape)
            Xf_tr_2a = scf_2a.transform(Xf_s2a[tr_fall_idx])
            y_tr_2a = np.array([s2a_l2i[l] for l in y_leaf[tr_fall_idx]])
            model_2a = train_fusion(Xb_tr_2a, Xf_tr_2a, y_tr_2a, len(LABELS_S2A), 5, Xf_s2a.shape[1], 300, seed)

            # Stage 2b
            adl_train_mask = y_binary[tr_idx] == 'ADL'
            tr_adl_idx = tr_idx[adl_train_mask]
            if len(set(y_leaf[tr_adl_idx])) < 2:
                continue
            scb_2b = StandardScaler().fit(Xb_s2b[tr_adl_idx].reshape(-1, 8))
            scf_2b = StandardScaler().fit(Xf_s2b[tr_adl_idx])
            Xb_tr_2b = scb_2b.transform(Xb_s2b[tr_adl_idx].reshape(-1, 8)).reshape(Xb_s2b[tr_adl_idx].shape)
            Xf_tr_2b = scf_2b.transform(Xf_s2b[tr_adl_idx])
            y_tr_2b = np.array([s2b_l2i[l] for l in y_leaf[tr_adl_idx]])
            model_2b = train_fusion(Xb_tr_2b, Xf_tr_2b, y_tr_2b, len(LABELS_S2B), 8, Xf_s2b.shape[1], 150, seed)

            # Route
            for local_i, global_i in enumerate(te_idx):
                true_leaf = y_leaf[global_i]
                true_bin = y_binary[global_i]
                pred_bin = pred1[local_i]

                if pred_bin == 'FALL':
                    xb = scb_2a.transform(Xb_s2a[global_i:global_i+1].reshape(-1,5)).reshape(1,5,5)
                    xf = scf_2a.transform(Xf_s2a[global_i:global_i+1])
                    pred_idx = predict_fusion(model_2a, xb, xf)[0]
                    pred_leaf = LABELS_S2A[pred_idx]
                else:
                    xb = scb_2b.transform(Xb_s2b[global_i:global_i+1].reshape(-1,8)).reshape(1,5,8)
                    xf = scf_2b.transform(Xf_s2b[global_i:global_i+1])
                    pred_idx = predict_fusion(model_2b, xb, xf)[0]
                    pred_leaf = LABELS_S2B[pred_idx]

                all_true_leaf.append(true_leaf); all_pred_leaf.append(pred_leaf)
                all_true_binary.append(true_bin); all_pred_binary.append(pred_bin)

                if true_bin == 'FALL' and pred_bin == 'FALL':
                    correctly_routed_fall_true.append(true_leaf); correctly_routed_fall_pred.append(pred_leaf)
                if true_bin == 'ADL' and pred_bin == 'ADL':
                    correctly_routed_adl_true.append(true_leaf); correctly_routed_adl_pred.append(pred_leaf)

        overall_acc = accuracy_score(all_true_leaf, all_pred_leaf)
        overall_bal = balanced_accuracy_score(all_true_leaf, all_pred_leaf)
        stage1_acc = accuracy_score(all_true_binary, all_pred_binary)
        routed_fall_acc = accuracy_score(correctly_routed_fall_true, correctly_routed_fall_pred) if correctly_routed_fall_true else np.nan
        routed_adl_acc = accuracy_score(correctly_routed_adl_true, correctly_routed_adl_pred) if correctly_routed_adl_true else np.nan

        print(f"    Stage 1 (binary) accuracy: {stage1_acc:.4f}")
        print(f"    Stage 2a accuracy | correctly routed to FALL: {routed_fall_acc:.4f} (n={len(correctly_routed_fall_true)})")
        print(f"    Stage 2b accuracy | correctly routed to ADL:  {routed_adl_acc:.4f} (n={len(correctly_routed_adl_true)})")
        print(f"    END-TO-END hierarchical accuracy (15-class):  {overall_acc:.4f}  (balanced: {overall_bal:.4f})")

        seed_results.append({
            'overall_acc': overall_acc, 'overall_bal': overall_bal, 'stage1_acc': stage1_acc,
            'routed_fall_acc': routed_fall_acc, 'routed_adl_acc': routed_adl_acc,
            'all_true_leaf': all_true_leaf, 'all_pred_leaf': all_pred_leaf,
            'all_true_binary': all_true_binary, 'all_pred_binary': all_pred_binary,
        })

    print("\n"+"="*70); print("PART 3: AGGREGATE RESULTS ACROSS SEEDS"); print("="*70)

    overall_accs = [r['overall_acc'] for r in seed_results]
    overall_bals = [r['overall_bal'] for r in seed_results]
    stage1_accs = [r['stage1_acc'] for r in seed_results]
    routed_fall_accs = [r['routed_fall_acc'] for r in seed_results]
    routed_adl_accs = [r['routed_adl_acc'] for r in seed_results]

    print(f"\n  Stage 1 (binary FALL/ADL) accuracy: mean={np.mean(stage1_accs):.4f} ± {np.std(stage1_accs):.4f}")
    print(f"  Stage 2a accuracy | correctly routed:  mean={np.nanmean(routed_fall_accs):.4f} ± {np.nanstd(routed_fall_accs):.4f}")
    print(f"  Stage 2b accuracy | correctly routed:  mean={np.nanmean(routed_adl_accs):.4f} ± {np.nanstd(routed_adl_accs):.4f}")
    print(f"\n  END-TO-END hierarchical accuracy (15-class): mean={np.mean(overall_accs):.4f} ± {np.std(overall_accs):.4f}")
    print(f"  END-TO-END hierarchical balanced accuracy:   mean={np.mean(overall_bals):.4f} ± {np.std(overall_bals):.4f}")

    report_run = seed_results[0]
    print(f"\n  Detailed classification report (seed={SEEDS[0]}):")
    print(classification_report(report_run['all_true_leaf'], report_run['all_pred_leaf'], labels=LABELS_ALL_15, target_names=LABELS_ALL_15, zero_division=0))

    cm_leaf = confusion_matrix(report_run['all_true_leaf'], report_run['all_pred_leaf'], labels=LABELS_ALL_15)
    print("  15-class confusion matrix:")
    print(pd.DataFrame(cm_leaf, index=LABELS_ALL_15, columns=LABELS_ALL_15))

    cm_bin = confusion_matrix(report_run['all_true_binary'], report_run['all_pred_binary'], labels=LABELS_S1)
    print("\n  Stage 1 binary confusion matrix:")
    print(pd.DataFrame(cm_bin, index=LABELS_S1, columns=LABELS_S1))

    misroute_errors = sum(1 for t, p in zip(report_run['all_true_binary'], report_run['all_pred_binary']) if t != p)
    total = len(report_run['all_true_binary'])
    leaf_errors_given_correct_routing = sum(
        1 for t_bin, p_bin, t_leaf, p_leaf in zip(report_run['all_true_binary'], report_run['all_pred_binary'], report_run['all_true_leaf'], report_run['all_pred_leaf'])
        if t_bin == p_bin and t_leaf != p_leaf
    )
    total_leaf_errors = sum(1 for t, p in zip(report_run['all_true_leaf'], report_run['all_pred_leaf']) if t != p)
    print(f"\n  Error decomposition (seed={SEEDS[0]}):")
    print(f"    Total end-to-end leaf errors: {total_leaf_errors} / {total}")
    print(f"    ...of which caused by Stage-1 misrouting: {misroute_errors}")
    print(f"    ...of which caused by Stage-2 confusion (correctly routed, wrong subtype): {leaf_errors_given_correct_routing}")

    print("\n"+"="*70); print("PART 4: VISUALIZATION"); print("="*70)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_leaf, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_ALL_15, yticklabels=LABELS_ALL_15, ax=ax, cbar=False, linewidths=0.5, linecolor='white', annot_kws={'size': 7})
    ax.set_xlabel('Predicted (end-to-end hierarchical)'); ax.set_ylabel('True')
    ax.set_title(f'Hierarchical Pipeline — 15-class Confusion Matrix (seed={SEEDS[0]})')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR_HIERARCHICAL}/hierarchical_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR_HIERARCHICAL}/hierarchical_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Oranges', xticklabels=LABELS_S1, yticklabels=LABELS_S1, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('Stage 1 Routing — Binary Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR_HIERARCHICAL}/hierarchical_stage1_routing_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {RESULTS_DIR_HIERARCHICAL}/hierarchical_stage1_routing_matrix.png")

    results = {
        'restrict_to_complete_subjects': RESTRICT_TO_COMPLETE_SUBJECTS,
        'stage1_model': 'SVM-RBF', 'stage1_features': s1_groups,
        'stage2a_model': 'Fusion', 'stage2a_features': s2a_groups,
        'stage2b_model': 'Fusion', 'stage2b_features': s2b_groups,
        'stage1_binary_accuracy': {'mean': float(np.mean(stage1_accs)), 'std': float(np.std(stage1_accs))},
        'stage2a_accuracy_given_correct_routing': {'mean': float(np.nanmean(routed_fall_accs)), 'std': float(np.nanstd(routed_fall_accs))},
        'stage2b_accuracy_given_correct_routing': {'mean': float(np.nanmean(routed_adl_accs)), 'std': float(np.nanstd(routed_adl_accs))},
        'end_to_end_accuracy': {'mean': float(np.mean(overall_accs)), 'std': float(np.std(overall_accs))},
        'end_to_end_balanced_accuracy': {'mean': float(np.mean(overall_bals)), 'std': float(np.std(overall_bals))},
        'error_decomposition_seed0': {
            'total_leaf_errors': int(total_leaf_errors),
            'from_stage1_misrouting': int(misroute_errors),
            'from_stage2_confusion_given_correct_routing': int(leaf_errors_given_correct_routing),
            'total_samples': int(total),
        },
        'confusion_matrix_15class': cm_leaf.tolist(),
        'confusion_matrix_stage1': cm_bin.tolist(),
    }
    with open(f'{RESULTS_DIR_HIERARCHICAL}/hierarchical_final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"HIERARCHICAL PIPELINE COMPLETE")
    print(f"  End-to-end accuracy: {np.mean(overall_accs):.4f} ± {np.std(overall_accs):.4f}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
