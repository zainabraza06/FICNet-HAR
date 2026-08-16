import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

from src.config import PROJECT_ROOT, RESULTS_DIR_HIERARCHICAL, RESULTS_DIR, RESULTS_DIR_S2A, RESULTS_DIR_S2B
from src.config import ALL_CODES, FALL_CODES, ADL_CODES_11, LABELS_S1, LABELS_S2A, LABELS_S2B, LABELS_ALL_15, SEEDS, CLASSICAL_MODELS
from src.features.extractors import build_binned_features, build_flat_features
from src.features.extractors import REGISTRY_S1, REGISTRY_S2A, REGISTRY_S2B
from src.data.loader import get_segment
from src.models.deep import train_bilstm, train_fusion
from src.models.classical import get_classical_models
from src.config import DEVICE

os.makedirs(RESULTS_DIR_HIERARCHICAL, exist_ok=True)
RESTRICT_TO_COMPLETE_SUBJECTS = False

def load_stage_result(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing result file {path}. You must run the individual stage script first.")
    with open(path, 'r') as f:
        return json.load(f)

def predict_one(model_name, labels, fitted, sc_bundle, Xb_row, Xf_row):
    if model_name in CLASSICAL_MODELS:
        xf_s = sc_bundle['f'].transform(Xf_row)
        return fitted.predict(xf_s)[0]
    elif model_name == 'BiLSTM':
        per_bin_dim = Xb_row.shape[-1]
        xb_s = sc_bundle['b'].transform(Xb_row.reshape(-1, per_bin_dim)).reshape(Xb_row.shape)
        with torch.no_grad():
            idx = fitted(torch.tensor(xb_s, dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()[0]
        return labels[idx]
    else:  # Fusion
        per_bin_dim = Xb_row.shape[-1]
        xb_s = sc_bundle['b'].transform(Xb_row.reshape(-1, per_bin_dim)).reshape(Xb_row.shape)
        xf_s = sc_bundle['f'].transform(Xf_row)
        with torch.no_grad():
            idx = fitted(torch.tensor(xb_s, dtype=torch.float32).to(DEVICE),
                          torch.tensor(xf_s, dtype=torch.float32).to(DEVICE)).argmax(1).cpu().numpy()[0]
        return labels[idx]

def main():
    print("\n" + "#" * 70)
    print("# HIERARCHICAL PIPELINE")
    print("#" * 70)

    print("\nLoading computed Stage 1/2a/2b results from JSON files...")
    s1 = load_stage_result(f'{RESULTS_DIR}/stage1_final_results.json')
    s2a = load_stage_result(f'{RESULTS_DIR_S2A}/stage2a_final_results.json')
    s2b = load_stage_result(f'{RESULTS_DIR_S2B}/stage2b_final_results.json')

    s1_groups, s1_model = s1['final_feature_groups'], s1['final_model']
    s2a_groups, s2a_model = s2a['final_feature_groups'], s2a['final_model']
    s2b_groups, s2b_model = s2b['final_feature_groups'], s2b['final_model']
    
    print(f"\nStage 1: {s1_model} on {s1_groups}")
    print(f"Stage 2a: {s2a_model} on {s2a_groups}")
    print(f"Stage 2b: {s2b_model} on {s2b_groups}")

    if s1_model not in CLASSICAL_MODELS:
        print("\nNOTE: Stage 1 selected a non-classical model. The routing loop below "
              "currently only implements classical Stage-1 inference; extend the Stage-1 "
              "branch analogously to Stage 2a/2b's DL branches if this occurs.")

    # ---- Build ONE unified per-sample dataset ----
    print("\n" + "=" * 70)
    print("PART 1: BUILDING UNIFIED DATASET")
    print("=" * 70)
    s2a_include_gyro_bins = 'gyro' in s2a_groups
    s2b_include_gyro_bins = 'gyro' in s2b_groups

    Xf_s1_l, Xf_s2a_l, Xb_s2a_l, Xf_s2b_l, Xb_s2b_l = [], [], [], [], []
    y_leaf, y_binary, groups_h = [], [], []

    for code in ALL_CODES:
        for subj in range(1, 68):
            seg = get_segment(code, subj)
            if seg is None:
                continue
            seg = seg.iloc[:1000]
            acc = seg[['acc_x', 'acc_y', 'acc_z']].values
            gyro = seg[['gyro_x', 'gyro_y', 'gyro_z']].values if 'gyro_x' in seg.columns else np.zeros_like(acc)
            roll = seg['roll'].values if 'roll' in seg.columns else np.zeros(len(seg))
            if len(acc) < 20:
                continue

            fc_s1 = build_flat_features(acc, None, None, s1_groups, REGISTRY_S1)
            fc_s2a = build_flat_features(acc, None, None, s2a_groups, REGISTRY_S2A)
            fc_s2b = build_flat_features(acc, gyro, roll, s2b_groups, REGISTRY_S2B)
            fb_s2a = build_binned_features(acc, gyro_data=None, include_gyro=s2a_include_gyro_bins)
            fb_s2b = build_binned_features(acc, gyro_data=gyro, include_gyro=s2b_include_gyro_bins)
            if any(v is None for v in [fc_s1, fc_s2a, fc_s2b, fb_s2a, fb_s2b]):
                continue

            Xf_s1_l.append(fc_s1); Xf_s2a_l.append(fc_s2a); Xb_s2a_l.append(fb_s2a)
            Xf_s2b_l.append(fc_s2b); Xb_s2b_l.append(fb_s2b)
            y_leaf.append(code)
            y_binary.append('FALL' if code in FALL_CODES else 'ADL')
            groups_h.append(subj)

    Xf_s1_h, Xf_s2a_h, Xb_s2a_h = np.array(Xf_s1_l), np.array(Xf_s2a_l), np.array(Xb_s2a_l)
    Xf_s2b_h, Xb_s2b_h = np.array(Xf_s2b_l), np.array(Xb_s2b_l)
    y_leaf, y_binary, groups_h = np.array(y_leaf), np.array(y_binary), np.array(groups_h)
    print(f"Unified dataset: {len(y_leaf)} samples, {len(set(groups_h))} subjects")

    # ---- Subject coverage diagnostic ----
    print("\n" + "=" * 70)
    print("PART 1b: SUBJECT COVERAGE DIAGNOSTIC")
    print("=" * 70)
    subj_ids = sorted(set(groups_h))
    cov = []
    for s in subj_ids:
        mask = groups_h == s
        cov.append({'subject': int(s),
                    'n_fall': int(((groups_h == s) & (y_binary == 'FALL')).sum()),
                    'n_adl': int(((groups_h == s) & (y_binary == 'ADL')).sum()),
                    'n_types': int(len(set(y_leaf[mask])))})
    cov_df = pd.DataFrame(cov)
    print(f"Subjects with data: {len(cov_df)}/67")
    print(f"Subjects with all 15 activity types: {(cov_df['n_types']==15).sum()}/{len(cov_df)}")
    cov_df.to_csv(f'{RESULTS_DIR_HIERARCHICAL}/subject_coverage.csv', index=False)

    if RESTRICT_TO_COMPLETE_SUBJECTS:
        complete = set(cov_df[cov_df['n_types'] == 15]['subject'])
        keep = np.array([g in complete for g in groups_h])
        Xf_s1_h, Xf_s2a_h, Xb_s2a_h = Xf_s1_h[keep], Xf_s2a_h[keep], Xb_s2a_h[keep]
        Xf_s2b_h, Xb_s2b_h = Xf_s2b_h[keep], Xb_s2b_h[keep]
        y_leaf, y_binary, groups_h = y_leaf[keep], y_binary[keep], groups_h[keep]
        print(f"RESTRICT_TO_COMPLETE_SUBJECTS=True: kept {len(groups_h)} samples across {len(complete)} subjects.")

    # ---- Hierarchical LOSO ----
    print("\n" + "=" * 70)
    print("PART 2: HIERARCHICAL LOSO")
    print("=" * 70)
    s2a_l2i = {l: i for i, l in enumerate(LABELS_S2A)}
    s2b_l2i = {l: i for i, l in enumerate(LABELS_S2B)}
    logo = LeaveOneGroupOut()
    seed_results = []

    for seed in SEEDS:
        print(f"\n  -- Seed {seed} --")
        all_true_leaf, all_pred_leaf, all_true_bin, all_pred_bin = [], [], [], []
        corr_fall_t, corr_fall_p, corr_adl_t, corr_adl_p = [], [], [], []

        for tr_idx, te_idx in logo.split(Xf_s1_h, y_binary, groups_h):
            if len(set(y_binary[tr_idx])) < 2:
                continue
            if s1_model not in CLASSICAL_MODELS:
                continue

            sc1 = StandardScaler().fit(Xf_s1_h[tr_idx])
            clf1 = get_classical_models(seed, len(LABELS_S1))[s1_model]
            clf1.fit(sc1.transform(Xf_s1_h[tr_idx]), y_binary[tr_idx])
            pred1 = clf1.predict(sc1.transform(Xf_s1_h[te_idx]))

            tr_fall_idx = tr_idx[y_binary[tr_idx] == 'FALL']
            if len(set(y_leaf[tr_fall_idx])) < 2:
                continue
            if s2a_model in CLASSICAL_MODELS:
                sc2a = StandardScaler().fit(Xf_s2a_h[tr_fall_idx])
                clf2a = get_classical_models(seed, len(LABELS_S2A))[s2a_model]
                clf2a.fit(sc2a.transform(Xf_s2a_h[tr_fall_idx]), y_leaf[tr_fall_idx])
                sc_bundle_2a = {'f': sc2a}
            else:
                scb2a = StandardScaler().fit(Xb_s2a_h[tr_fall_idx].reshape(-1, Xb_s2a_h.shape[2]))
                scf2a = StandardScaler().fit(Xf_s2a_h[tr_fall_idx])
                y2a = np.array([s2a_l2i[l] for l in y_leaf[tr_fall_idx]])
                Xb2a_tr = scb2a.transform(Xb_s2a_h[tr_fall_idx].reshape(-1, Xb_s2a_h.shape[2])).reshape(Xb_s2a_h[tr_fall_idx].shape)
                Xf2a_tr = scf2a.transform(Xf_s2a_h[tr_fall_idx])
                if s2a_model == 'BiLSTM':
                    clf2a = train_bilstm(Xb2a_tr, y2a, len(LABELS_S2A), Xb_s2a_h.shape[2], 300, 8, seed)
                else:
                    clf2a = train_fusion(Xb2a_tr, Xf2a_tr, y2a, len(LABELS_S2A), Xb_s2a_h.shape[2], Xf_s2a_h.shape[1], 300, seed)
                clf2a.eval()
                sc_bundle_2a = {'b': scb2a, 'f': scf2a}

            tr_adl_idx = tr_idx[y_binary[tr_idx] == 'ADL']
            if len(set(y_leaf[tr_adl_idx])) < 2:
                continue
            if s2b_model in CLASSICAL_MODELS:
                sc2b = StandardScaler().fit(Xf_s2b_h[tr_adl_idx])
                clf2b = get_classical_models(seed, len(LABELS_S2B))[s2b_model]
                clf2b.fit(sc2b.transform(Xf_s2b_h[tr_adl_idx]), y_leaf[tr_adl_idx])
                sc_bundle_2b = {'f': sc2b}
            else:
                scb2b = StandardScaler().fit(Xb_s2b_h[tr_adl_idx].reshape(-1, Xb_s2b_h.shape[2]))
                scf2b = StandardScaler().fit(Xf_s2b_h[tr_adl_idx])
                y2b = np.array([s2b_l2i[l] for l in y_leaf[tr_adl_idx]])
                Xb2b_tr = scb2b.transform(Xb_s2b_h[tr_adl_idx].reshape(-1, Xb_s2b_h.shape[2])).reshape(Xb_s2b_h[tr_adl_idx].shape)
                Xf2b_tr = scf2b.transform(Xf_s2b_h[tr_adl_idx])
                if s2b_model == 'BiLSTM':
                    clf2b = train_bilstm(Xb2b_tr, y2b, len(LABELS_S2B), Xb_s2b_h.shape[2], 150, 16, seed)
                else:
                    clf2b = train_fusion(Xb2b_tr, Xf2b_tr, y2b, len(LABELS_S2B), Xb_s2b_h.shape[2], Xf_s2b_h.shape[1], 150, seed)
                clf2b.eval()
                sc_bundle_2b = {'b': scb2b, 'f': scf2b}

            for local_i, global_i in enumerate(te_idx):
                true_leaf, true_bin, pred_bin = y_leaf[global_i], y_binary[global_i], pred1[local_i]
                if pred_bin == 'FALL':
                    pred_leaf = predict_one(s2a_model, LABELS_S2A, clf2a, sc_bundle_2a,
                                             Xb_s2a_h[global_i:global_i+1], Xf_s2a_h[global_i:global_i+1])
                else:
                    pred_leaf = predict_one(s2b_model, LABELS_S2B, clf2b, sc_bundle_2b,
                                             Xb_s2b_h[global_i:global_i+1], Xf_s2b_h[global_i:global_i+1])
                all_true_leaf.append(true_leaf); all_pred_leaf.append(pred_leaf)
                all_true_bin.append(true_bin); all_pred_bin.append(pred_bin)
                if true_bin == 'FALL' and pred_bin == 'FALL':
                    corr_fall_t.append(true_leaf); corr_fall_p.append(pred_leaf)
                if true_bin == 'ADL' and pred_bin == 'ADL':
                    corr_adl_t.append(true_leaf); corr_adl_p.append(pred_leaf)

        overall_acc = accuracy_score(all_true_leaf, all_pred_leaf)
        overall_bal = balanced_accuracy_score(all_true_leaf, all_pred_leaf)
        stage1_acc = accuracy_score(all_true_bin, all_pred_bin)
        routed_fall_acc = accuracy_score(corr_fall_t, corr_fall_p) if corr_fall_t else np.nan
        routed_adl_acc = accuracy_score(corr_adl_t, corr_adl_p) if corr_adl_t else np.nan
        print(f"    Stage1={stage1_acc:.4f}  Fall|routed={routed_fall_acc:.4f}  "
              f"ADL|routed={routed_adl_acc:.4f}  End-to-end={overall_acc:.4f} (bal={overall_bal:.4f})")
        seed_results.append({'overall_acc': overall_acc, 'overall_bal': overall_bal, 'stage1_acc': stage1_acc,
                              'routed_fall_acc': routed_fall_acc, 'routed_adl_acc': routed_adl_acc,
                              'true_leaf': all_true_leaf, 'pred_leaf': all_pred_leaf,
                              'true_bin': all_true_bin, 'pred_bin': all_pred_bin})

    # ---- Aggregate + report ----
    print("\n" + "=" * 70)
    print("PART 3: AGGREGATE RESULTS")
    print("=" * 70)
    overall_accs = [r['overall_acc'] for r in seed_results]
    print(f"END-TO-END accuracy: {np.mean(overall_accs):.4f} ± {np.std(overall_accs):.4f}")

    report_run = seed_results[0]
    print(classification_report(report_run['true_leaf'], report_run['pred_leaf'],
                                 labels=LABELS_ALL_15, target_names=LABELS_ALL_15, zero_division=0))
    cm_leaf = confusion_matrix(report_run['true_leaf'], report_run['pred_leaf'], labels=LABELS_ALL_15)
    cm_bin = confusion_matrix(report_run['true_bin'], report_run['pred_bin'], labels=LABELS_S1)

    misroute = sum(1 for t, p in zip(report_run['true_bin'], report_run['pred_bin']) if t != p)
    total_err = sum(1 for t, p in zip(report_run['true_leaf'], report_run['pred_leaf']) if t != p)
    print(f"Error decomposition (seed 0): total={total_err}, misrouting={misroute}, "
          f"subtype-confusion-given-correct-routing={total_err - misroute}")

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm_leaf, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS_ALL_15, yticklabels=LABELS_ALL_15,
                ax=ax, annot_kws={'size': 7})
    ax.set_title('Hierarchical — 15-class confusion matrix (seed 0)')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR_HIERARCHICAL}/hierarchical_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Oranges', xticklabels=LABELS_S1, yticklabels=LABELS_S1, ax=ax)
    ax.set_title('Stage 1 routing — binary confusion matrix')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR_HIERARCHICAL}/hierarchical_stage1_routing_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    results = {
        'stage1_model': s1_model, 'stage1_features': s1_groups,
        'stage2a_model': s2a_model, 'stage2a_features': s2a_groups,
        'stage2b_model': s2b_model, 'stage2b_features': s2b_groups,
        'end_to_end_accuracy': {'mean': float(np.mean(overall_accs)), 'std': float(np.std(overall_accs))},
        'error_decomposition_seed0': {'total': total_err, 'misrouting': misroute,
                                       'subtype_confusion_given_correct_routing': total_err - misroute},
        'confusion_matrix_15class': cm_leaf.tolist(),
        'confusion_matrix_stage1': cm_bin.tolist(),
    }
    with open(f'{RESULTS_DIR_HIERARCHICAL}/hierarchical_final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("HIERARCHICAL PIPELINE COMPLETE")
    print(f"  End-to-end accuracy: {np.mean(overall_accs):.4f} ± {np.std(overall_accs):.4f}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
