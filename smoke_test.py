"""Smoke test script for MobiAct HAR preprocessing and feature extraction."""

import sys
import numpy as np
from src.data_loader import MobiActLoader
from src.features import (
    extract_stage1_features,
    extract_stage2a_features,
    extract_stage2b_features,
)
from src.pipeline import (
    build_stage1_dataset,
    build_stage2a_dataset,
    build_stage2b_dataset,
)


def run_smoke_test():
    print("Initializing MobiActLoader...")
    data_root = "MobiAct_Dataset_v2.0/Annotated Data"
    loader = MobiActLoader(data_root)

    # 1. Test data loader listing subjects
    print("\n--- 1. Testing list_subjects ---")
    fall_code = "BSC"
    adl_code = "STD"
    
    fall_subjects = loader.list_subjects(fall_code)
    adl_subjects = loader.list_subjects(adl_code)
    
    print(f"Subjects for {fall_code} (found {len(fall_subjects)}): {fall_subjects[:5]}...")
    print(f"Subjects for {adl_code} (found {len(adl_subjects)}): {adl_subjects[:5]}...")

    if not fall_subjects or not adl_subjects:
        print("Error: No subjects found in the annotated data directory.")
        sys.exit(1)

    # 2. Test get_segment and features on a single file
    print("\n--- 2. Testing segment loading & single-file features ---")
    subj_id = fall_subjects[0]
    print(f"Loading fall segment for code={fall_code}, subject={subj_id}...")
    fall_seg = loader.get_segment(fall_code, subj_id)
    if fall_seg is None:
        print(f"Error: Segment for {fall_code} and subject {subj_id} is None.")
        sys.exit(1)
    print(f"Loaded segment shape: {fall_seg.shape}")
    print(f"Columns: {list(fall_seg.columns)}")
    print(f"Label transitions in file: {loader.get_label_transitions(fall_code, subj_id)}")

    # Extract features
    s1_feat = extract_stage1_features(fall_seg)
    print(f"Stage 1 features: {s1_feat} (shape={s1_feat.shape if s1_feat is not None else 'None'})")

    s2a_feat = extract_stage2a_features(fall_seg)
    print(f"Stage 2a features shape: {s2a_feat.shape if s2a_feat is not None else 'None'} (Expected: (34,))")

    s2b_feat = extract_stage2b_features(fall_seg)
    print(f"Stage 2b features shape: {s2b_feat.shape if s2b_feat is not None else 'None'} (Expected: (35,))")

    assert s1_feat is not None, "Stage 1 extraction failed"
    assert s2a_feat is not None, "Stage 2a extraction failed"
    assert s2b_feat is not None, "Stage 2b extraction failed"
    assert s1_feat.shape == (5,), f"Stage 1 feature shape expected (5,), got {s1_feat.shape}"
    assert s2a_feat.shape == (34,), f"Stage 2a feature shape expected (34,), got {s2a_feat.shape}"
    assert s2b_feat.shape == (35,), f"Stage 2b feature shape expected (35,), got {s2b_feat.shape}"

    # 3. Test dataset building on a small subset of subjects
    print("\n--- 3. Testing dataset building pipelines ---")
    test_subjs = sorted(list(set(fall_subjects[:3] + adl_subjects[:3])))
    print(f"Building test datasets for subjects: {test_subjs}")

    print("Building Stage 1 dataset...")
    X1, y1, g1 = build_stage1_dataset(loader, test_subjs)
    print(f"Stage 1: X={X1.shape}, y={y1.shape}, groups={g1.shape}")
    print(f"Stage 1 labels: {np.unique(y1)}")

    print("Building Stage 2a dataset...")
    X2a, y2a, g2a = build_stage2a_dataset(loader, test_subjs)
    print(f"Stage 2a: X={X2a.shape}, y={y2a.shape}, groups={g2a.shape}")
    print(f"Stage 2a labels: {np.unique(y2a)}")

    print("Building Stage 2b dataset (11-class)...")
    X2b, y2b, g2b = build_stage2b_dataset(loader, test_subjs, n_classes=11)
    print(f"Stage 2b: X={X2b.shape}, y={y2b.shape}, groups={g2b.shape}")
    print(f"Stage 2b labels: {np.unique(y2b)}")

    print("\nAll smoke tests passed successfully!")


if __name__ == "__main__":
    run_smoke_test()
