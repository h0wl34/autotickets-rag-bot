import torch.nn.functional as F
import torch

def focal_loss(logits, targets, alpha=None, gamma=2, reduction="mean"):
    """
    Focal loss for binary & multiclass classification.
    
    Args:
        logits: (N, C) for multiclass, (N,) for binary
        targets: (N,) long for multiclass, float for binary
        alpha: 
            - None (default): no class weighting
            - Tensor of shape (C,) for multiclass (like CrossEntropyLoss weight)
            - Float scalar (0<alpha<1) for binary (applied to positive class)
        gamma: focusing parameter
        reduction: 'mean' or 'sum'
    """
    if logits.ndim == 1 or logits.shape[1] == 1:  
        # ---- Binary case ----
        logits = logits.view(-1)
        targets = targets.float().view(-1)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce_loss)  # p_t

        if isinstance(alpha, float):  
            # alpha is weight for positive class
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            focal = alpha_t * (1 - pt) ** gamma * bce_loss
        else:
            focal = (1 - pt) ** gamma * bce_loss

    else:
        # ---- Multiclass case ----
        ce_loss = F.cross_entropy(
            logits, targets.long(), weight=alpha, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** gamma * ce_loss

    if reduction == "mean":
        return focal.mean()
    elif reduction == "sum":
        return focal.sum()
    else:
        return focal