"""Exact discrete CVaR risk-envelope maximization for two branches."""
from __future__ import annotations

import torch


def cvar_weights(loss: torch.Tensor, probability: torch.Tensor, confidence: float, *, validate_probability: bool = True) -> torch.Tensor:
    """Return maximizing q for loss[...,2] and matching probabilities.

    The interval for q0 is [max(0,1-p1/(1-alpha)), min(1,p0/(1-alpha))].
    A tie returns the nominal distribution to avoid arbitrary branch switching.
    """
    if not 0.0 <= confidence < 1.0:
        raise ValueError("CVaR confidence must be in [0, 1).")
    if loss.shape[-1] != 2 or probability.shape[-1] != 2:
        raise ValueError("The minimal implementation supports exactly two branches.")
    if validate_probability and (bool(torch.any(probability < 0)) or not bool(torch.allclose(probability.sum(-1), torch.ones_like(probability[..., 0]), atol=1e-5))):
        raise ValueError("Branch probabilities must be nonnegative and sum to one.")
    probability = probability.expand_as(loss)
    cap = probability / (1.0 - confidence)
    lo = (1.0 - cap[..., 1]).clamp(min=0.0)
    hi = cap[..., 0].clamp(max=1.0)
    q0 = torch.where(loss[..., 0] > loss[..., 1], hi, lo)
    q0 = torch.where(torch.isclose(loss[..., 0], loss[..., 1]), probability[..., 0], q0)
    return torch.stack((q0, 1.0 - q0), dim=-1)


def cvar_value(loss: torch.Tensor, probability: torch.Tensor, confidence: float, *, validate_probability: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    q = cvar_weights(loss, probability, confidence, validate_probability=validate_probability)
    return (q * loss).sum(-1), q
