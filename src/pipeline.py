"""Dataset builders — assemble (X, y, groups) arrays for LOSO evaluation."""

import numpy as np
from .data_loader import MobiActLoader, FALL_CODES, ADL_CODES_9, ADL_CODES_11
from .features import extract_stage1_features, extract_stage2a_features, extract_stage2b_features


def build_dataset(loader: MobiActLoader, codes: list[str], subjects: list[int],
                   extractor, **extractor_kwargs):
    """Generic builder: applies `extractor` to every (code, subject) pair."""
    X, y, groups = [], [], []
    for code in codes:
        for subj in subjects:
            seg = loader.get_segment(code, subj)
            if seg is None:
                continue
            feat = extractor(seg, **extractor_kwargs)
            if feat is not None:
                X.append(feat)
                y.append(code)
                groups.append(subj)
    return np.array(X), np.array(y), np.array(groups)


def build_stage1_dataset(loader, subjects, n_bins=5):
    codes = FALL_CODES + ADL_CODES_9  # binary label assigned downstream (code in FALL_CODES)
    return build_dataset(loader, codes, subjects, extract_stage1_features, n_bins=n_bins)


def build_stage2a_dataset(loader, subjects, n_bins=5):
    return build_dataset(loader, FALL_CODES, subjects, extract_stage2a_features, n_bins=n_bins)


def build_stage2b_dataset(loader, subjects, n_classes=11, n_bins=5):
    codes = ADL_CODES_11 if n_classes == 11 else ADL_CODES_9
    return build_dataset(loader, codes, subjects, extract_stage2b_features, n_bins=n_bins)
