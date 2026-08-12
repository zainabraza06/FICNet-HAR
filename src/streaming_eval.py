"""
StreamingFusionNet architecture and LOSO evaluation for the streaming ablation.

Model
-----
StreamingFusionNet uses a two-level recurrent hierarchy:
  Inner LSTM  — processes the sub-bins inside each window →
                produces a per-window representation.
  Outer LSTM  — processes the sequence of window representations across
                the subject stream → captures long-range context.
Both the outer LSTM output and the MLP branch (processing flat features)
are cross-attended before final classification, following the same
cross-attention design used in DualBranchFusionNet (src/models.py).

Evaluation
----------
run_streaming_loso() performs Leave-One-Subject-Out cross-validation
where each subject's entire stream (pre-windowed) is held out as the
test set.  Metrics are stratified by window purity:

  Overall   — all windows
  Pure      — windows where the dominant label covers >= purity_thresh
  Boundary  — windows where dominant label coverage <  purity_thresh

This stratification matches the analysis in run_stage2b_streaming_final.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Model
# ============================================================

class StreamingFusionNet(nn.Module):
    """
    Two-level hierarchical fusion network for streaming HAR.

    Parameters
    ----------
    per_bin_dim  : int  — feature dimension per sub-bin (e.g. 8)
    flat_dim     : int  — flat feature vector dimension
    n_classes    : int  — number of output classes
    inner_hidden : int  — inner LSTM hidden size (default 16)
    outer_hidden : int  — outer LSTM hidden size (default 16)
    mlp_hidden   : int  — MLP branch hidden size (default 32)
    """

    def __init__(self, per_bin_dim: int, flat_dim: int, n_classes: int,
                 inner_hidden: int = 16, outer_hidden: int = 16, mlp_hidden: int = 32):
        super().__init__()
        inner_out = inner_hidden * 2  # bidirectional
        outer_out = outer_hidden * 2

        # Level 1: encode sub-bins within a window
        self.inner_lstm = nn.LSTM(per_bin_dim, inner_hidden,
                                  batch_first=True, bidirectional=True)

        # Level 2: encode the sequence of window representations
        self.outer_lstm = nn.LSTM(inner_out, outer_hidden,
                                  batch_first=True, bidirectional=True)

        # MLP branch for flat features
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, outer_out),
        )

        # Cross-attention projections (DL → ML direction)
        self.q_proj_dl = nn.Linear(outer_out, outer_out)
        self.k_proj_ml = nn.Linear(outer_out, outer_out)
        self.v_proj_ml = nn.Linear(outer_out, outer_out)

        # Cross-attention projections (ML → DL direction)
        self.q_proj_ml = nn.Linear(outer_out, outer_out)
        self.k_proj_dl = nn.Linear(outer_out, outer_out)
        self.v_proj_dl = nn.Linear(outer_out, outer_out)

        self.dropout    = nn.Dropout(0.4)
        self.classifier = nn.Linear(outer_out * 2, n_classes)

    def forward(self, x_bins: torch.Tensor, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_bins : (1, N_windows, sub_bins, per_bin_dim)
        x_flat : (1, N_windows, flat_dim)

        Returns
        -------
        logits : (N_windows, n_classes)
        """
        b, nw, sb, pd = x_bins.shape

        # Inner LSTM: (B*Nw, sub_bins, pd) → window representations
        xf_seq = x_bins.view(b * nw, sb, pd)
        _, (h_n, _) = self.inner_lstm(xf_seq)
        win_repr = torch.cat([h_n[-2], h_n[-1]], dim=1).view(b, nw, -1)

        # Outer LSTM: (B, Nw, inner_out) → context-aware representations
        dl_out, _ = self.outer_lstm(win_repr)
        dl_out = self.dropout(dl_out)

        # MLP branch
        ml_out = self.mlp(x_flat.view(b * nw, -1)).view(b, nw, -1)
        ml_out = self.dropout(ml_out)

        # Cross-attention: DL queries attend to ML keys/values
        q_dl    = self.q_proj_dl(dl_out)
        k_ml    = self.k_proj_ml(ml_out)
        v_ml    = self.v_proj_ml(ml_out)
        scale   = dl_out.size(-1) ** 0.5
        attn_dl = torch.softmax((q_dl * k_ml).sum(-1, keepdim=True) / scale, dim=-1)
        dl_att  = attn_dl * v_ml + dl_out

        # Cross-attention: ML queries attend to DL keys/values
        q_ml    = self.q_proj_ml(ml_out)
        k_dl    = self.k_proj_dl(dl_out)
        v_dl    = self.v_proj_dl(dl_out)
        attn_ml = torch.softmax((q_ml * k_dl).sum(-1, keepdim=True) / scale, dim=-1)
        ml_att  = attn_ml * v_dl + ml_out

        # Classify each window independently
        return self.classifier(
            torch.cat([dl_att, ml_att], dim=-1)
        ).squeeze(0)   # (N_windows, n_classes)


# ============================================================
# Training helper
# ============================================================

def _train_streaming_fusion(
    train_data: dict,
    label_to_idx: dict,
    n_classes: int,
    per_bin_dim: int,
    flat_dim: int,
    epochs: int = 50,
    seed: int = 0,
) -> StreamingFusionNet:
    torch.manual_seed(seed)
    model = StreamingFusionNet(per_bin_dim, flat_dim, n_classes).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-2)

    model.train()
    for _ in range(epochs):
        for subj, (Xb, Xf, yw, _) in train_data.items():
            y_idx = np.array([label_to_idx[l] for l in yw])
            Xb_t  = torch.tensor(Xb, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            Xf_t  = torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            y_t   = torch.tensor(y_idx, dtype=torch.long).to(DEVICE)
            opt.zero_grad()
            F.cross_entropy(model(Xb_t, Xf_t), y_t).backward()
            opt.step()

    return model


# ============================================================
# Metrics helpers
# ============================================================

def _purity_stratified_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    purities: np.ndarray,
    thresh: float = 0.90,
) -> dict:
    """Return overall, pure-window, and boundary-window accuracy metrics."""
    overall_acc = accuracy_score(y_true, y_pred)
    overall_bal = balanced_accuracy_score(y_true, y_pred)

    pm = purities >= thresh
    if pm.sum() > 0:
        pure_acc = accuracy_score(y_true[pm], y_pred[pm])
        pure_bal = balanced_accuracy_score(y_true[pm], y_pred[pm])
    else:
        pure_acc = pure_bal = None

    bm = ~pm
    if bm.sum() > 0:
        boundary_acc = accuracy_score(y_true[bm], y_pred[bm])
        boundary_bal = balanced_accuracy_score(y_true[bm], y_pred[bm])
    else:
        boundary_acc = boundary_bal = None

    return {
        'overall_accuracy':              overall_acc,
        'overall_balanced_accuracy':     overall_bal,
        'pure_window_accuracy':          pure_acc,
        'pure_window_balanced_accuracy': pure_bal,
        'pure_window_fraction':          float(pm.mean()),
        'boundary_window_accuracy':      boundary_acc,
        'boundary_window_balanced_accuracy': boundary_bal,
        'boundary_window_fraction':      float(bm.mean()),
    }


# ============================================================
# LOSO evaluation
# ============================================================

def run_streaming_loso(
    data: dict,
    label_list: list[str],
    epochs: int = 50,
    seed: int = 0,
    purity_thresh: float = 0.90,
    verbose: bool = True,
) -> dict:
    """
    Leave-One-Subject-Out evaluation of StreamingFusionNet.

    Parameters
    ----------
    data          : {subj_id: (X_bins, X_flat, y_labels, purities)}
    label_list    : ordered list of class label strings
    epochs        : training epochs per fold
    seed          : random seed for weight initialisation
    purity_thresh : window purity threshold for stratification
    verbose       : print per-fold progress

    Returns
    -------
    dict with overall, pure-window, and boundary-window metrics.
    """
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    n_classes    = len(label_list)

    subj_ids     = list(data.keys())
    all_true, all_pred, all_purity = [], [], []

    for fold_i, test_subj in enumerate(subj_ids):
        train_data = {s: data[s] for s in subj_ids if s != test_subj}
        if len(train_data) < 2:
            continue

        Xb, Xf, yw, purity = data[test_subj]
        per_bin_dim = Xb.shape[2]
        flat_dim    = Xf.shape[1]

        model = _train_streaming_fusion(
            train_data, label_to_idx, n_classes,
            per_bin_dim, flat_dim, epochs, seed,
        )
        model.eval()

        y_idx = np.array([label_to_idx[l] for l in yw])
        with torch.no_grad():
            logits = model(
                torch.tensor(Xb, dtype=torch.float32).unsqueeze(0).to(DEVICE),
                torch.tensor(Xf, dtype=torch.float32).unsqueeze(0).to(DEVICE),
            )
            pred = logits.argmax(dim=1).cpu().numpy()

        all_true.extend(y_idx)
        all_pred.extend(pred)
        all_purity.extend(purity)

        if verbose and fold_i % 10 == 0:
            print(f"    fold {fold_i}/{len(subj_ids)} done")

    return _purity_stratified_metrics(
        np.array(all_true), np.array(all_pred), np.array(all_purity),
        thresh=purity_thresh,
    )
