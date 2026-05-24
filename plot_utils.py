"""Small helpers for inspecting character language models."""

import torch


def normalize_counts(counts: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Convert a tensor of counts into probabilities along one dimension."""
    totals = counts.sum(dim=dim, keepdim=True)
    probabilities = counts.float() / totals.clamp_min(1)
    return torch.nan_to_num(probabilities)
