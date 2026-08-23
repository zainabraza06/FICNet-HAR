import torch
import torch.nn as nn
import torch.nn.functional as F


class LightCNN(nn.Module):
    """
    CNN feature extractor with lightweight self-attention temporal head.

    Used for ERM and SAM modes. Pooling is a learned linear query
    (attn_pool: Linear(d_model, 1) -> softmax over time) — blind to
    physical priors, supervised only by class labels.

    Architecture
    ------------
    Conv1d(C,32,k=5) -> BN -> ReLU -> MaxPool(2)
    Conv1d(32,64,k=5) -> BN -> ReLU -> MaxPool(2)
    Conv1d(64,d_model,k=3) -> BN -> ReLU
    + sinusoidal-like learnable positional embedding
    -> MultiheadAttention (self) -> LayerNorm -> FFN -> LayerNorm
    -> attn_pool (Linear(d_model,1) -> softmax) -> weighted sum
    -> Dropout(0.3) -> Linear(d_model, n_classes)
    """

    def __init__(self, n_channels, n_classes, win_len,
                 d_model=64, n_heads=4, seq_len_hint=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(),
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, seq_len_hint, d_model) * 0.02
        )
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1      = nn.LayerNorm(d_model)
        self.ffn        = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2      = nn.LayerNorm(d_model)
        self.attn_pool  = nn.Linear(d_model, 1)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_model, n_classes))

    def forward(self, x):
        # x: (B, T, C)
        z = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, T', d_model)
        T = z.shape[1]
        z = z + self.pos_embed[:, :T, :]
        attn_out, _ = self.self_attn(z, z, z)
        z   = self.norm1(z + attn_out)
        seq = self.norm2(z + self.ffn(z))
        w   = torch.softmax(self.attn_pool(seq), dim=1)       # (B, T', 1)
        pooled = (seq * w).sum(dim=1)                          # (B, d_model)
        return self.classifier(pooled)


class FICNet(nn.Module):
    """
    Feature-Invariance-Conditioned network.

    Same conv trunk as LightCNN, but pooling uses cross-attention against a
    handcrafted feature vector instead of the blind Linear(d_model,1) query.
    This gives the pooling step a physical prior about WHERE in the window to
    look (impact frame, transition point) without relying on the data-starved
    label signal for rare classes (CHU, SIT, fall types).

    Also exposes a feat_proj branch so the caller can compute the per-sample
    FIC consistency loss (see fic_combined_loss).

    Used for both FIC and SAM+FIC modes — the only difference between those
    two is the optimizer (Adam vs SAM), not the architecture.

    Parameters
    ----------
    n_feat : int — dimension of the SELECTED (post-filtering) handcrafted
                   feature vector for this fold
    """

    def __init__(self, n_channels, n_classes, n_feat, win_len,
                 d_model=64, n_heads=4, seq_len_hint=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model), nn.ReLU(),
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, seq_len_hint, d_model) * 0.02
        )
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1     = nn.LayerNorm(d_model)
        self.ffn       = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Cross-attention pooling conditioned on handcrafted features
        self.feat_encoder = nn.Sequential(
            nn.Linear(n_feat, d_model), nn.ReLU(), nn.LayerNorm(d_model),
        )
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.pool_norm  = nn.LayerNorm(d_model)

        # Projection branch for consistency loss
        self.feat_proj = nn.Sequential(
            nn.Linear(n_feat, d_model), nn.ReLU(), nn.Linear(d_model, d_model),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(d_model, n_classes))

    def forward(self, x, feat, return_embed=False):
        # x: (B, T, C),  feat: (B, n_feat)
        z = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        T = z.shape[1]
        z = z + self.pos_embed[:, :T, :]
        attn_out, _ = self.self_attn(z, z, z)
        z   = self.norm1(z + attn_out)
        seq = self.norm2(z + self.ffn(z))

        query       = self.feat_encoder(feat).unsqueeze(1)         # (B, 1, d)
        pooled, _   = self.cross_attn(query, seq, seq)
        pooled      = self.pool_norm(pooled.squeeze(1))            # (B, d)

        logits = self.classifier(pooled)
        if return_embed:
            feat_target = self.feat_proj(feat)
            return logits, pooled, feat_target
        return logits


# ── FIC loss helpers ──────────────────────────────────────────────────────────

def fic_consistency_loss(pooled_embed, feat_target):
    """
    Per-sample cosine consistency between the CNN pooled embedding and the
    handcrafted-feature projection.

    Because this operates per-sample, it can never starve the way IRM's
    per-environment penalty did under subject/class sparsity.
    """
    return 1.0 - F.cosine_similarity(pooled_embed, feat_target, dim=1).mean()


def fic_combined_loss(model, Xb, yb, featb, class_weights, fic_loss_weight):
    """
    CE + FIC_LOSS_WEIGHT * consistency.

    Shared by 'fic' and 'sam_fic' modes so both training branches are
    identical in loss composition — only the optimizer step differs.

    Returns
    -------
    total, ce, cons : scalar tensors (total is the one to call .backward() on)
    """
    logits, pooled, feat_target = model(Xb, featb, return_embed=True)
    ce    = F.cross_entropy(logits, yb, weight=class_weights)
    cons  = fic_consistency_loss(pooled, feat_target)
    total = ce + fic_loss_weight * cons
    return total, ce, cons
