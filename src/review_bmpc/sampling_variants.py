"""True elite CEM and weighted MPPI variants of the branch-policy sampler."""
from __future__ import annotations

import math

import torch

from .controller import BranchMPC, _feedback
from .model import clip_action, safe_brake_action, step_vehicle


class _SamplingVariant(BranchMPC):
    def _finish(self, name: str, ego: torch.Tensor, predictions: torch.Tensor, chosen: dict,
                branch_offset: torch.Tensor, record_plan: bool) -> dict:
        chosen["branch_offset"] = branch_offset
        failed = chosen["max_violation"] > self.config.fallback_violation_m
        chosen["action"] = torch.where(failed[:, None], safe_brake_action(ego, self.vehicle),
                                       chosen["action"])
        chosen["failed"] = failed
        if record_plan:
            states = ego[0].expand(2, 5).clone()
            planned = []
            branch_index = torch.arange(2, device=self.device)
            for step in range(self.config.horizon):
                action = (chosen["action"][0].expand(2, 2) if step == 0 else
                          clip_action(_feedback(name, states, predictions[0, :, step - 1],
                                                branch_index, self.vehicle) + branch_offset[0], self.vehicle))
                states = step_vehicle(states, action, self.vehicle)
                planned.append(states)
            chosen["planned_ego"] = torch.stack(planned, dim=1)
        return chosen

    def _proposals(self, center: torch.Tensor, branches: torch.Tensor,
                   root_scale: torch.Tensor, branch_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        count = center.shape[0]
        candidates = self.config.candidates
        root = center[:, None, :] + torch.randn((count, candidates, 2), device=self.device,
                                                generator=self.generator) * root_scale
        branch = branches[:, None, :, :] + torch.randn((count, candidates, 2, 2), device=self.device,
                                                        generator=self.generator) * branch_scale
        root[:, 0] = center
        branch[:, 0] = branches
        return root, branch


class CEMBranchMPC(_SamplingVariant):
    """Cross-entropy method with an elite distribution update each iteration."""

    @torch.no_grad()
    def solve(self, name, ego, other, predictions, probabilities, parameters,
              record_plan: bool = False) -> dict:
        count = ego.shape[0]
        mean = torch.zeros((count, 2), device=self.device)
        branch_mean = torch.zeros((count, 2, 2), device=self.device)
        root_scale = torch.tensor([2.0, 0.32], device=self.device).expand(count, 2).clone()
        branch_scale = torch.tensor([1.8, 0.28], device=self.device).expand(count, 2, 2).clone()
        best = None
        best_branch = None
        for _ in range(self.config.cem_iterations):
            root, branch = self._proposals(mean, branch_mean, root_scale[:, None, :],
                                           branch_scale[:, None, :, :])
            result = self._score(name, ego, other, predictions, probabilities, parameters, root, branch)
            indices = result["objective"].argmin(dim=1)
            rows = torch.arange(count, device=self.device)
            chosen = {key: value[rows, indices] for key, value in result.items()}
            if best is None:
                best, best_branch = chosen, branch[rows, indices]
            else:
                better = chosen["objective"] < best["objective"]
                best = {key: torch.where(better.view(-1, *([1] * (value.ndim - 1))),
                                         value, best[key]) for key, value in chosen.items()}
                best_branch = torch.where(better[:, None, None], branch[rows, indices], best_branch)
            elite_count = max(2, math.ceil(self.config.candidates / 5))
            elite = result["objective"].topk(elite_count, largest=False, dim=1).indices
            elite_root = torch.gather(root, 1, elite[..., None].expand(-1, -1, 2))
            elite_branch = torch.gather(branch, 1, elite[..., None, None].expand(-1, -1, 2, 2))
            mean = elite_root.mean(dim=1)
            branch_mean = elite_branch.mean(dim=1)
            root_scale = elite_root.std(dim=1, unbiased=False).clamp_min(0.05)
            branch_scale = elite_branch.std(dim=1, unbiased=False).clamp_min(0.03)
        return self._finish(name, ego, predictions, best, best_branch, record_plan)


class MPPIBranchMPC(_SamplingVariant):
    """Path-integral weighted action update over the same branch-policy family."""

    def __init__(self, *args, temperature: float = 3.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    @torch.no_grad()
    def solve(self, name, ego, other, predictions, probabilities, parameters,
              record_plan: bool = False) -> dict:
        count = ego.shape[0]
        mean = torch.zeros((count, 2), device=self.device)
        branch_mean = torch.zeros((count, 2, 2), device=self.device)
        root_scale = torch.tensor([2.0, 0.32], device=self.device)
        branch_scale = torch.tensor([1.8, 0.28], device=self.device)
        for _ in range(self.config.cem_iterations):
            root, branch = self._proposals(mean, branch_mean, root_scale, branch_scale)
            result = self._score(name, ego, other, predictions, probabilities, parameters, root, branch)
            objective = result["objective"]
            weights = torch.softmax(-(objective - objective.amin(dim=1, keepdim=True)) /
                                    self.temperature, dim=1)
            mean = (weights[..., None] * root).sum(dim=1)
            branch_mean = (weights[..., None, None] * branch).sum(dim=1)
            root_scale = root_scale * 0.7
            branch_scale = branch_scale * 0.7
        final = self._score(name, ego, other, predictions, probabilities, parameters,
                            mean[:, None, :], branch_mean[:, None, :, :])
        chosen = {key: value[:, 0] for key, value in final.items()}
        return self._finish(name, ego, predictions, chosen, branch_mean, record_plan)
