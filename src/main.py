import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.config import MODELS, CORE_GROUPS, CANDIDATE_GROUPS, LABELS_S1, ALPHA, RESULTS_DIR, CLASSICAL_MODELS
from src.data.loader import build_dataset_stage1
from src.features.extractors import feature_names_for
from src.training.evaluate import run_on_model, get_loso_predictions
from src.models.classical import get_classical_models
from src.models.deep import train_bilstm, train_fusion
from src.utils.viz import plot_confusion_matrix, plot_feature_distributions, plot_energy_profiles, plot_binned_boxplots
from src.utils.edge import measure_inference_time_classical, measure_inference_time_bilstm, measure_inference_time_fusion, measure_model_size
import warnings

warnings.filterwarnings('ignore')

def print_seeds(name, metrics):
    accs=[m['accuracy'] for m in metrics]
    bals=[m['balanced_accuracy'] for m in metrics]
    f1s=[m['macro_f1'] for m in metrics]
    print(f"    {name}: acc={np.mean(accs):.4f}±{np.std(accs):.4f} bal={np.mean(bals):.4f}±{np.std(bals):.4f} f1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

def pick_best_model(model_results):
    return max(model_results, key=lambda m: np.mean([x['accuracy'] for x in model_results[m]]))

def main():
    print("\n"+"="*70); print(f"PART 1: MODEL SELECTION on core set {CORE_GROUPS}"); print("="*70)

    Xb_core, Xf_core, y_core, g_core = build_dataset_stage1(CORE_GROUPS)
    print(f"Dataset: {Xb_core.shape}, {Xf_core.shape}, classes: {sorted(set(y_core))}")

    model_results_core = {}
    for model in MODELS:
        print(f"\n  {model}:")
        metrics = run_on_model(model, Xb_core, Xf_core, y_core, g_core, LABELS_S1)
        print_seeds(model, metrics)
        model_results_core[model] = metrics

    model_stage1 = pick_best_model(model_results_core)
    print(f"\n  >>> Model selected on core set: {model_stage1}")


    print("\n"+"="*70); print(f"PART 2: GREEDY FORWARD FEATURE SELECTION on {model_stage1}"); print("="*70)

    current_groups = list(CORE_GROUPS)
    remaining_candidates = list(CANDIDATE_GROUPS)

    Xb_cur, Xf_cur, y_cur, g_cur = build_dataset_stage1(current_groups)
    current_metrics = run_on_model(model_stage1, Xb_cur, Xf_cur, y_cur, g_cur, LABELS_S1)
    print_seeds(f"round 0: {current_groups}", current_metrics)

    selection_log = [{'groups': list(current_groups),
                       'mean_acc': float(np.mean([m['accuracy'] for m in current_metrics])),
                       'accepted': True, 'reason': 'core (always included)'}]

    round_num = 1
    while remaining_candidates:
        print(f"\n  -- Round {round_num}: current set = {current_groups}, candidates = {remaining_candidates} --")
        candidate_results = {}
        for cand in remaining_candidates:
            trial_groups = current_groups + [cand]
            Xb_t, Xf_t, y_t, g_t = build_dataset_stage1(trial_groups)
            trial_metrics = run_on_model(model_stage1, Xb_t, Xf_t, y_t, g_t, LABELS_S1)
            accs_trial = [m['accuracy'] for m in trial_metrics]
            accs_cur = [m['accuracy'] for m in current_metrics]
            t_stat, p = stats.ttest_rel(accs_trial, accs_cur)
            gain = np.mean(accs_trial) - np.mean(accs_cur)
            print_seeds(f"+{cand} -> {trial_groups}", trial_metrics)
            print(f"      gain={gain:+.4f}, paired t-test p={p:.4f} {'(SIG)' if p < ALPHA else '(n.s.)'}")
            candidate_results[cand] = {'groups': trial_groups, 'metrics': trial_metrics, 'gain': gain, 'p': p}

        eligible = {c: r for c, r in candidate_results.items() if r['gain'] > 0 and r['p'] < ALPHA}
        if not eligible:
            print(f"\n  No remaining candidate gives a significant positive improvement — stopping.")
            for c, r in candidate_results.items():
                selection_log.append({'groups': r['groups'], 'mean_acc': float(np.mean([m['accuracy'] for m in r['metrics']])),
                                       'accepted': False, 'reason': f"gain={r['gain']:+.4f}, p={r['p']:.4f} — rejected"})
            break

        winner = max(eligible, key=lambda c: eligible[c]['gain'])
        print(f"\n  >>> ACCEPTED: '{winner}' (gain={eligible[winner]['gain']:+.4f}, p={eligible[winner]['p']:.4f})")
        for c, r in candidate_results.items():
            if c == winner:
                selection_log.append({'groups': r['groups'], 'mean_acc': float(np.mean([m['accuracy'] for m in r['metrics']])),
                                       'accepted': True, 'reason': f"gain={r['gain']:+.4f}, p={r['p']:.4f}"})
            else:
                selection_log.append({'groups': r['groups'], 'mean_acc': float(np.mean([m['accuracy'] for m in r['metrics']])),
                                       'accepted': False, 'reason': f"gain={r['gain']:+.4f}, p={r['p']:.4f} — beaten by '{winner}'"})

        current_groups = candidate_results[winner]['groups']
        current_metrics = candidate_results[winner]['metrics']
        remaining_candidates.remove(winner)
        round_num += 1

    FINAL_GROUPS_S1 = current_groups
    print(f"\n  >>> FINAL FEATURE SET (Stage 1, model={model_stage1}): {FINAL_GROUPS_S1}")

    Xb_final, Xf_final, y_final, g_final = build_dataset_stage1(FINAL_GROUPS_S1)
    FINAL_FEATURE_NAMES = feature_names_for(FINAL_GROUPS_S1)


    print("\n"+"="*70); print(f"PART 2b: RE-VERIFY MODEL on final feature set {FINAL_GROUPS_S1}"); print("="*70)

    model_results_final = {model_stage1: current_metrics} 
    for model in MODELS:
        if model == model_stage1: continue
        print(f"\n  {model}:")
        metrics = run_on_model(model, Xb_final, Xf_final, y_final, g_final, LABELS_S1)
        print_seeds(model, metrics)
        model_results_final[model] = metrics

    best_model = pick_best_model(model_results_final)
    if best_model != model_stage1:
        print(f"\n  >>> Model ranking CHANGED with final feature set: {model_stage1} -> {best_model}")
    else:
        print(f"\n  >>> Model choice confirmed: {best_model}")

    FINAL_Xb, FINAL_Xf, FINAL_y, FINAL_g = Xb_final, Xf_final, y_final, g_final
    FINAL_FEATURE_SET = f"{FINAL_GROUPS_S1} ({Xf_final.shape[1]}-dim)"


    print("\n"+"="*70); print("PART 3: PER-CLASS PERFORMANCE (final config)"); print("="*70)

    all_true, all_pred = get_loso_predictions(best_model, FINAL_Xb, FINAL_Xf, FINAL_y, FINAL_g, LABELS_S1)

    print(f"\nClassification Report ({best_model}, feature set: {FINAL_FEATURE_SET}):")
    print(classification_report(all_true, all_pred, target_names=LABELS_S1))

    cm = confusion_matrix(all_true, all_pred, labels=LABELS_S1)
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=LABELS_S1, columns=LABELS_S1))


    print("\n"+"="*70); print("PART 4: VISUALIZATION"); print("="*70)

    plot_confusion_matrix(cm, best_model, FINAL_FEATURE_SET)
    plot_feature_distributions(best_model, FINAL_Xf, FINAL_y, FINAL_FEATURE_NAMES)
    plot_energy_profiles(FINAL_Xb, FINAL_y)
    plot_binned_boxplots(FINAL_Xb, FINAL_y)


    print("\n"+"="*70); print("PART 5: EDGE ANALYSIS"); print("="*70)
    print(f"\n  Measuring {best_model} ({FINAL_FEATURE_SET})...")

    if best_model in CLASSICAL_MODELS:
        sc = StandardScaler().fit(FINAL_Xf)
        clf = get_classical_models(0)[best_model]
        clf.fit(sc.transform(FINAL_Xf), FINAL_y)
        time_ms, time_std = measure_inference_time_classical(clf, sc.transform(FINAL_Xf))
        size_mb = measure_model_size(clf, is_torch=False)
    elif best_model == 'BiLSTM':
        per_bin_dim = FINAL_Xb.shape[2]
        l2i = {l:i for i,l in enumerate(LABELS_S1)}
        y_idx = np.array([l2i[l] for l in FINAL_y])
        sc = StandardScaler().fit(FINAL_Xb.reshape(-1, per_bin_dim))
        Xb_s = sc.transform(FINAL_Xb.reshape(-1, per_bin_dim)).reshape(FINAL_Xb.shape)
        model = train_bilstm(Xb_s, y_idx, len(LABELS_S1), per_bin_dim, 200, 8, 0)
        time_ms, time_std = measure_inference_time_bilstm(model, Xb_s)
        size_mb = measure_model_size(model, is_torch=True)
    else:  # Fusion
        per_bin_dim = FINAL_Xb.shape[2]; flat_dim = FINAL_Xf.shape[1]
        l2i = {l:i for i,l in enumerate(LABELS_S1)}
        y_idx = np.array([l2i[l] for l in FINAL_y])
        scb = StandardScaler().fit(FINAL_Xb.reshape(-1, per_bin_dim))
        scf = StandardScaler().fit(FINAL_Xf)
        Xb_s = scb.transform(FINAL_Xb.reshape(-1, per_bin_dim)).reshape(FINAL_Xb.shape)
        Xf_s = scf.transform(FINAL_Xf)
        model = train_fusion(Xb_s, Xf_s, y_idx, len(LABELS_S1), per_bin_dim, flat_dim, 200, 0)
        time_ms, time_std = measure_inference_time_fusion(model, Xb_s, Xf_s)
        size_mb = measure_model_size(model, is_torch=True)

    throughput = 100 / (time_ms / 1000) if time_ms > 0 else float('inf')
    power = 50 if time_ms < 1 else 100 if time_ms < 5 else 400

    print(f"    Inference Time: {time_ms:.3f} ± {time_std:.3f} ms")
    print(f"    Model Size: {size_mb:.3f} MB")
    print(f"    Throughput: {throughput:.1f} samples/sec")
    print(f"    Estimated Power: {power:.1f} mW")
    print(f"    Edge Suitability: {'Excellent' if time_ms < 5 and size_mb < 1 else 'Good' if time_ms < 20 else 'Poor'}")


    results = {
        'model_selected_on_core': model_stage1,
        'final_feature_groups': FINAL_GROUPS_S1,
        'final_model': best_model,
        'model_changed_after_feature_selection': best_model != model_stage1,
        'feature_selection_log': selection_log,
        'model_comparison_core': {k: [{'acc':m['accuracy'],'bal':m['balanced_accuracy'],'f1':m['macro_f1']} for m in v]
                                   for k,v in model_results_core.items()},
        'model_comparison_final_features': {k: [{'acc':m['accuracy'],'bal':m['balanced_accuracy'],'f1':m['macro_f1']} for m in v]
                                             for k,v in model_results_final.items()},
        'confusion_matrix': cm.tolist(),
        'edge_metrics': {'inference_time_ms': time_ms, 'model_size_mb': size_mb,
                          'throughput_samples_sec': throughput, 'power_mw': power},
    }
    with open(f'{RESULTS_DIR}/stage1_final_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"STAGE 1 COMPLETE")
    print(f"  Model selected on core set: {model_stage1}")
    print(f"  Final feature groups (greedy, significance-gated): {FINAL_GROUPS_S1}")
    print(f"  Final model (re-verified): {best_model}")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
