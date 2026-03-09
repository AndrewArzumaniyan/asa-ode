from __future__ import annotations

import torch



def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes mean squared error only over observed entries defined by mask."""
    diff = (pred - target) ** 2
    weighted = diff * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return weighted.sum() / denom
