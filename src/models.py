"""
Neural network architectures for MobiAct HAR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBiLSTM(nn.Module):
    """
    Bidirectional LSTM classifier for binned temporal features.
    
    Parameters
    ----------
    per_bin_dim : int
        Dimension of features in each bin (e.g. 8).
    n_classes : int
        Number of output classes (e.g. 2 for Stage 1 fall gate).
    hidden : int, default=16
        LSTM hidden state dimension.
    """
    def __init__(self, per_bin_dim, n_classes, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(per_bin_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # Concatenate final forward and backward hidden states
        h = self.dropout(torch.cat([h_n[-2], h_n[-1]], dim=1))
        return self.classifier(h)


class DualBranchFusionNet(nn.Module):
    """
    Dual-branch fusion network combining sequence features (BiLSTM branch)
    and flat statistical features (MLP branch) with cross-attention.
    
    Parameters
    ----------
    per_bin_dim : int
        Dimension of features in each bin (e.g. 8).
    flat_dim : int
        Dimension of flat features (e.g. 26).
    n_classes : int
        Number of output classes.
    lstm_hidden : int, default=16
        LSTM hidden state dimension.
    mlp_hidden : int, default=32
        MLP branch hidden state dimension.
    """
    def __init__(self, per_bin_dim, flat_dim, n_classes, lstm_hidden=16, mlp_hidden=32):
        super().__init__()
        out_dim = lstm_hidden * 2
        
        # Branch 1: Sequence processing
        self.lstm = nn.LSTM(per_bin_dim, lstm_hidden, batch_first=True, bidirectional=True)
        
        # Branch 2: Flat stats processing
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden, out_dim)
        )
        
        # Cross-attention projections
        self.q_proj_dl = nn.Linear(out_dim, out_dim)
        self.k_proj_ml = nn.Linear(out_dim, out_dim)
        self.v_proj_ml = nn.Linear(out_dim, out_dim)
        
        self.q_proj_ml = nn.Linear(out_dim, out_dim)
        self.k_proj_dl = nn.Linear(out_dim, out_dim)
        self.v_proj_dl = nn.Linear(out_dim, out_dim)
        
        self.dropout = nn.Dropout(0.4)
        self.classifier = nn.Linear(out_dim * 2, n_classes)

    def forward(self, x_bins, x_flat):
        # DL Sequence representation
        _, (h_n, _) = self.lstm(x_bins)
        dl_repr = self.dropout(torch.cat([h_n[-2], h_n[-1]], dim=1))
        
        # ML Flat stats representation
        ml_repr = self.dropout(self.mlp(x_flat))
        
        # Cross-Attention: DL query attending to ML keys/values
        q_dl = self.q_proj_dl(dl_repr)
        k_ml = self.k_proj_ml(ml_repr)
        v_ml = self.v_proj_ml(ml_repr)
        attn_dl = torch.softmax((q_dl * k_ml).sum(1, keepdim=True) / (q_dl.size(1) ** 0.5), dim=1)
        dl_att = attn_dl * v_ml + dl_repr
        
        # Cross-Attention: ML query attending to DL keys/values
        q_ml = self.q_proj_ml(ml_repr)
        k_dl = self.k_proj_dl(dl_repr)
        v_dl = self.v_proj_dl(dl_repr)
        attn_ml = torch.softmax((q_ml * k_dl).sum(1, keepdim=True) / (q_ml.size(1) ** 0.5), dim=1)
        ml_att = attn_ml * v_dl + ml_repr
        
        # Concatenate and classify
        return self.classifier(torch.cat([dl_att, ml_att], dim=1))
