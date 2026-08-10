"""Dataset builders — assemble (X, y, groups) arrays for LOSO evaluation."""

import numpy as np
from .data_loader import MobiActLoader, FALL_CODES, ADL_CODES_9, ADL_CODES_11
from .features import (
    extract_stage1_features,
    extract_stage2a_features,
    extract_stage2b_features,
    build_binned_features,
    build_stage1_flat_features,
)



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


def build_stage1_neural_dataset(loader, fall_codes, adl_codes, subjects, n_bins=5, max_samples=1000):
    """
    Build Stage 1 dataset for neural network evaluation.
    Returns:
        X_bins (np.ndarray): Binned temporal features of shape (N, n_bins, 8)
        X_flat (np.ndarray): Flat statistical features of shape (N, 26)
        y (np.ndarray): Labels ('FALL' or 'ADL') of shape (N,)
        groups (np.ndarray): Subject IDs of shape (N,)
    """
    X_bins, X_flat, y, groups = [], [], [], []
    for code in fall_codes:
        for subj in subjects:
            seg = loader.get_segment(code, subj)
            if seg is None:
                continue
            fb = build_binned_features(seg, n_bins, max_samples)
            fc = build_stage1_flat_features(seg, n_bins, max_samples)
            if fb is not None and fc is not None:
                X_bins.append(fb)
                X_flat.append(fc)
                y.append('FALL')
                groups.append(subj)
    for code in adl_codes:
        for subj in subjects:
            seg = loader.get_segment(code, subj)
            if seg is None:
                continue
            fb = build_binned_features(seg, n_bins, max_samples)
            fc = build_stage1_flat_features(seg, n_bins, max_samples)
            if fb is not None and fc is not None:
                X_bins.append(fb)
                X_flat.append(fc)
                y.append('ADL')
                groups.append(subj)
    return np.array(X_bins), np.array(X_flat), np.array(y), np.array(groups)

