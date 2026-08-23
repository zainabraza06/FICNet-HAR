import numpy as np
import torch
import torch.nn.functional as F
from src.config import (DEVICE, BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS,
                        SAM_RHO, FIC_LOSS_WEIGHT, PROGRESS_EVERY)
from src.models.cnn import LightCNN, FICNet, fic_combined_loss
from src.models.sam import SAM


# ── Class-weight helper ───────────────────────────────────────────────────────

def get_class_weights(y_idx, n_classes):
    """Inverse-frequency weights, placed on DEVICE."""
    counts = np.bincount(y_idx, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(mode, X_train, y_train_idx, subj_train, n_classes,
                win_len, n_channels, epochs=EPOCHS, seed=0,
                feat_train=None, use_class_weights=True, verbose_tag=''):
    """
    Train one model for a single fold / seed.

    Parameters
    ----------
    mode          : str   — 'erm' | 'sam' | 'fic' | 'sam_fic'
    X_train       : (N, win_len, n_channels) float32
    y_train_idx   : (N,) int64
    subj_train    : (N,) int64  — kept for parity / logging
    feat_train    : (N, n_feat_selected) float32 or None
                    Required (not None) when mode in ('fic', 'sam_fic').
    verbose_tag   : str   — prefix for progress prints

    Notes
    -----
    The entire training split is kept resident on GPU (torch.tensor(..., device=DEVICE))
    and batches are drawn by indexing with torch.randperm, avoiding per-batch host→device
    transfers.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    uses_fic = mode in ('fic', 'sam_fic')
    if uses_fic:
        assert feat_train is not None, f"mode='{mode}' requires feat_train"
        model = FICNet(n_channels, n_classes,
                       n_feat=feat_train.shape[1], win_len=win_len).to(DEVICE)
    else:
        model = LightCNN(n_channels, n_classes, win_len).to(DEVICE)

    class_weights = get_class_weights(y_train_idx, n_classes) if use_class_weights else None

    # Keep whole fold resident on GPU
    Xt = torch.tensor(X_train,     dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(y_train_idx, dtype=torch.long,    device=DEVICE)
    ft = torch.tensor(feat_train,  dtype=torch.float32, device=DEVICE) if uses_fic else None
    n  = len(Xt)

    # Optimiser
    if mode in ('erm', 'fic'):
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    else:  # sam, sam_fic
        opt = SAM(model.parameters(), torch.optim.Adam,
                  rho=SAM_RHO, lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    n_batches = n // BATCH_SIZE

    for epoch in range(epochs):
        loss_sum, ce_sum, cons_sum, n_b = 0.0, 0.0, 0.0, 0
        perm = torch.randperm(n, device=DEVICE)

        for b in range(n_batches):
            idx = perm[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            Xb, yb = Xt[idx], yt[idx]

            if mode == 'erm':
                opt.zero_grad()
                loss = F.cross_entropy(model(Xb), yb, weight=class_weights)
                loss.backward()
                opt.step()
                loss_sum += loss.item()

            elif mode == 'sam':
                loss1 = F.cross_entropy(model(Xb), yb, weight=class_weights)
                loss1.backward()
                opt.first_step(zero_grad=True)
                loss2 = F.cross_entropy(model(Xb), yb, weight=class_weights)
                loss2.backward()
                opt.second_step(zero_grad=True)
                loss_sum += loss2.item()

            elif mode == 'fic':
                featb = ft[idx]
                opt.zero_grad()
                total, ce, cons = fic_combined_loss(
                    model, Xb, yb, featb, class_weights, FIC_LOSS_WEIGHT
                )
                total.backward()
                opt.step()
                loss_sum += total.item(); ce_sum += ce.item(); cons_sum += cons.item()

            elif mode == 'sam_fic':
                featb = ft[idx]
                total1, ce1, cons1 = fic_combined_loss(
                    model, Xb, yb, featb, class_weights, FIC_LOSS_WEIGHT
                )
                total1.backward()
                opt.first_step(zero_grad=True)
                total2, ce2, cons2 = fic_combined_loss(
                    model, Xb, yb, featb, class_weights, FIC_LOSS_WEIGHT
                )
                total2.backward()
                opt.second_step(zero_grad=True)
                loss_sum += total2.item(); ce_sum += ce2.item(); cons_sum += cons2.item()

            n_b += 1

        if (epoch + 1) % PROGRESS_EVERY == 0 or epoch == 0 or epoch == epochs - 1:
            avg = loss_sum / max(n_b, 1)
            if uses_fic:
                print(f"    [{verbose_tag}] epoch {epoch+1}/{epochs}  "
                      f"loss={avg:.4f}  ce={ce_sum/max(n_b,1):.4f}  "
                      f"cons={cons_sum/max(n_b,1):.4f}", flush=True)
            else:
                print(f"    [{verbose_tag}] epoch {epoch+1}/{epochs}  "
                      f"loss={avg:.4f}", flush=True)

    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, X, labels, mode, feat=None, batch_size=BATCH_SIZE):
    """
    Batched inference (host tensors, moved to DEVICE per batch to avoid OOM
    when the test split is large).

    Returns an array of string label predictions parallel to X.
    """
    model.eval()
    uses_fic = mode in ('fic', 'sam_fic')
    Xt = torch.tensor(X, dtype=torch.float32)
    Ft = torch.tensor(feat, dtype=torch.float32) if (uses_fic and feat is not None) else None

    preds = []
    for i in range(0, len(Xt), batch_size):
        xb = Xt[i:i + batch_size].to(DEVICE)
        if uses_fic:
            logits = model(xb, Ft[i:i + batch_size].to(DEVICE))
        else:
            logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())

    return np.array([labels[i] for i in np.concatenate(preds)])
