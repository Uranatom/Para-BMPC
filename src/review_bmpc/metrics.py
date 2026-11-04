import torch
from torch.nn import functional as F

def observed_margin_cost(clearance: torch.Tensor, margin_m: float = .5,
                         smoothing_m: float = .25) -> torch.Tensor:
    """Smooth episode minimum-clearance deficit, measured in meters."""
    return smoothing_m * F.softplus((margin_m - clearance) / smoothing_m)
