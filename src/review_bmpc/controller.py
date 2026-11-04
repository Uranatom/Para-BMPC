"""Batched two-branch receding-horizon controller with corrected CVaR."""
from __future__ import annotations

from dataclasses import dataclass
import torch

from .model import VEHICLE, VehicleConfig, clip_action, safe_brake_action, smooth_clearance, step_vehicle
from .risk import cvar_value
from .scenarios import branch_reference, other_action


@dataclass(frozen=True)
class ControllerConfig:
    horizon: int = 15
    confidence: float = 0.9
    candidates: int = 32
    cem_iterations: int = 2
    safety_margin_m: float = 0.5
    constraint_penalty: float = 500.0
    fallback_violation_m: float = 0.2
    base_parameters: tuple[float, ...] | None = None
    interaction_version: str = "v3"
    controller_version: str = "legacy"
    warm_start: bool = True
    explicit_seeds: bool = True
    feasibility_first: bool = True
    release_reference: bool = True
    jerk_weight: float = 0.0
    road_corridor: bool | None = None
    lateral_reference_protocol: str | None = None
    effective_action_protocol: str | None = None
    clearance_recovery_protocol: str = 'constant_v1'
    clearance_recovery_rate_mps: float = 0.05
    planner_seed_protocol: str = 'legacy_v1'

    def __post_init__(self):
        if self.effective_action_protocol in ('bounded_compact_v2','neutral_bounded_v3','neutral_bounded_v4') and (
                self.controller_version != 'v7' or self.interaction_version != 'v6' or self.horizon != 10):
            raise ValueError('Bounded compact tuning is declared only for v7/v6 with N10.')
        if self.clearance_recovery_protocol not in ('constant_v1', 'initial_gap_ramp_v1', 'persistent_gap_ramp_v2', 'feasible_seed_envelope_v3', 'observed_gap_envelope_v4', 'observed_gap_objective_v5', 'nominal_gate_objective_v6'):
            raise ValueError('Unsupported clearance recovery protocol.')
        if self.clearance_recovery_protocol != 'constant_v1' and (
                self.controller_version != 'v7' or self.interaction_version != 'v6' or self.horizon != 10):
            raise ValueError('Initial-gap recovery is declared only for v7/v6 with N10.')
        if not 0 < self.clearance_recovery_rate_mps <= 1.:
            raise ValueError('Clearance recovery rate must be in (0, 1] m/s.')
        if self.planner_seed_protocol not in ('legacy_v1', 'retained_seed_v1'):
            raise ValueError('Unsupported planner seed protocol.')
        if self.planner_seed_protocol != 'legacy_v1' and self.clearance_recovery_protocol not in ('persistent_gap_ramp_v2','feasible_seed_envelope_v3'):
            raise ValueError('Retained seeds require versioned persistent recovery.')
        if self.planner_seed_protocol != 'legacy_v1' and not self.warm_start:
            raise ValueError('Retained seeds require warm-start candidate auditing.')
        if self.controller_version == "v7":
            if self.effective_action_protocol is None:
                object.__setattr__(self, 'effective_action_protocol', 'scenario_compact_v1')
            if self.effective_action_protocol not in ('scenario_compact_v1', 'bounded_compact_v2','neutral_bounded_v3','neutral_bounded_v4'):
                raise ValueError('Unsupported v7 effective action protocol.')
            if self.lateral_reference_protocol is None:
                object.__setattr__(self, 'lateral_reference_protocol', 'quintic40_curvature_feedback_v1')
            if self.lateral_reference_protocol != 'quintic40_curvature_feedback_v1':
                raise ValueError('Unsupported v7 lateral reference protocol.')
            if self.road_corridor is None:
                object.__setattr__(self, 'road_corridor', True)
            if self.base_parameters is None:
                base = list(BASE_PARAMETERS)
                base[10] = .05
                object.__setattr__(self, "base_parameters", tuple(base))
            if self.jerk_weight == 0.0:
                object.__setattr__(self, "jerk_weight", 1.0)
        if self.clearance_recovery_protocol == 'nominal_gate_objective_v6' and self.effective_action_protocol not in ('neutral_bounded_v3','neutral_bounded_v4'):
            raise ValueError('Nominal feasibility gates require the neutral-preserving bounded tuner.')
        if self.jerk_weight < 0:
            raise ValueError("Jerk weight must be nonnegative.")


BASE_PARAMETERS = (1.0, 1.0, 1.0, 1.0, 0.1, 1.0, 1.0, 1.0, 0.1, 1.0, 0.1)


def policy_parameters(policy_output: torch.Tensor | None, adaptation: str, count: int,
                      device: torch.device, *,
                      base_parameters: tuple[float, ...] | list[float] | None = None) -> torch.Tensor:
    """Map bounded actor outputs to positive cost weights and a clearance parameter."""
    values = BASE_PARAMETERS if base_parameters is None else base_parameters
    if len(values) != 11 or any(value <= 0 for value in values):
        raise ValueError("Base controller parameters must contain 11 positive values.")
    base = torch.tensor(values, dtype=torch.float32, device=device).expand(count, -1)
    if adaptation == "fixed":
        return base.clone()
    if policy_output is None or policy_output.shape != (count, 11):
        raise ValueError("Adaptive control requires actor output shaped [batch, 11].")
    if adaptation not in ("cbf_only", "weights_only", "joint"):
        raise ValueError(f"Unknown adaptation: {adaptation}")
    factors = torch.exp((policy_output.clamp(-1.0, 1.0)) * 1.38629436)
    parameters = base * factors
    if adaptation == "cbf_only":
        parameters[:, :10] = base[:, :10]
    if adaptation == "weights_only":
        parameters[:, 10] = base[:, 10]
    return parameters


def _feedback(name: str, state: torch.Tensor, other: torch.Tensor,
              branch: int | torch.Tensor, vehicle: VehicleConfig = VEHICLE) -> torch.Tensor:
    y_ref, heading_ref, speed_ref = branch_reference(name, state, other, branch, vehicle)
    heading_error = torch.atan2(torch.sin(heading_ref - state[..., 2]), torch.cos(heading_ref - state[..., 2]))
    target_steer = (0.13 * (y_ref - state[..., 1]) + 0.8 * heading_error).clamp(-0.38, 0.38)
    steer_rate = (2.2 * (target_steer - state[..., 4])).clamp(-0.7, 0.7)
    accel = (1.0 * (speed_ref - state[..., 3])).clamp(-4.0, 2.0)
    return torch.stack((accel, steer_rate), dim=-1)


class BranchMPC:
    """Sample feedback-policy branches with a shared first action.

    This is a compact batched shooting solver. It is not a port of MATLAB's
    active-set QP, so solver performance must be reported separately.
    """

    def __init__(self, config: ControllerConfig, vehicle: VehicleConfig = VEHICLE, *, seed: int = 1, device: str = "cpu") -> None:
        if config.interaction_version not in ("v3", "v4", "v5", "v6"):
            raise ValueError("Unknown interaction version.")
        self.config = config
        self.vehicle = vehicle
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def solve(
        self,
        name: str,
        ego: torch.Tensor,
        other: torch.Tensor,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        parameters: torch.Tensor,
        record_plan: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch = ego.shape[0]
        candidates = self.config.candidates
        root_center = torch.zeros((batch, 2), device=self.device)
        branch_center = torch.zeros((batch, 2, 2), device=self.device)
        root_scale = torch.tensor([2.0, 0.32], device=self.device)
        branch_scale = torch.tensor([1.8, 0.28], device=self.device)
        best = None
        for iteration in range(self.config.cem_iterations):
            root = root_center[:, None, :] + torch.randn((batch, candidates, 2), device=self.device, generator=self.generator) * root_scale
            branch = branch_center[:, None, :, :] + torch.randn((batch, candidates, 2, 2), device=self.device, generator=self.generator) * branch_scale
            root[:, 0] = root_center
            branch[:, 0] = branch_center
            result = self._score(name, ego, other, predictions, probabilities, parameters, root, branch)
            index = result["objective"].argmin(dim=1)
            row = torch.arange(batch, device=self.device)
            chosen = {key: value[row, index] for key, value in result.items()}
            chosen["branch_offset"] = branch[row, index]
            root_center = root[row, index]
            branch_center = branch[row, index]
            root_scale = root_scale * 0.5
            branch_scale = branch_scale * 0.5
            if best is None:
                best = chosen
            else:
                improve = chosen["objective"] < best["objective"]
                best = {key: torch.where(improve.view(-1, *([1] * (value.ndim - 1))), value, best[key]) for key, value in chosen.items()}
        assert best is not None
        failed = best["max_violation"] > self.config.fallback_violation_m
        brake = safe_brake_action(ego, self.vehicle)
        best["action"] = torch.where(failed[:, None], brake, best["action"])
        best["failed"] = failed
        if record_plan:
            planned = []
            planned_other = []
            state = ego[0].expand(2, 5).clone()
            other_state = other[0].expand(2, 5).clone()
            branch_index = torch.arange(2, device=self.device)
            for step in range(self.config.horizon):
                action = best["action"][0].expand(2, 2) if step == 0 else clip_action(
                    _feedback(name, state,
                              other_state if self.config.interaction_version in ("v4", "v5", "v6") else predictions[0, :, step - 1],
                              branch_index, self.vehicle)
                    + best["branch_offset"][0], self.vehicle
                )
                if self.config.interaction_version in ("v4", "v5", "v6"):
                    other_state = step_vehicle(other_state, other_action(
                        name, other_state, branch_index, self.vehicle,
                        ego=state, interaction_version=self.config.interaction_version), self.vehicle)
                else:
                    other_state = predictions[0, :, step]
                state = step_vehicle(state, action, self.vehicle)
                planned.append(state)
                planned_other.append(other_state)
            best["planned_ego"] = torch.stack(planned, dim=1)
            best["planned_other"] = torch.stack(planned_other, dim=1)
        return best

    def _score(self, name, ego, other, predictions, probabilities, parameters, root, branch):
        batch, candidates = root.shape[:2]
        x = ego[:, None, None, :].expand(batch, candidates, 2, 5).clone()
        branch_cost = torch.zeros((batch, candidates, 2), device=self.device)
        max_violation = torch.zeros_like(branch_cost)
        first_action = None
        stage_weights = parameters[:, None, None, 4:8]
        terminal_weights = parameters[:, None, None, :4]
        input_weights = parameters[:, None, None, 8:10]
        kappa = parameters[:, None, None, 10]
        # Score predicted geometric clearance with a tunable margin. The
        # violation penalty does not imply a forward-invariance guarantee.
        effective_margin = self.config.safety_margin_m + 0.05 / kappa.clamp_min(0.01)
        branch_index = torch.arange(2, device=self.device).view(1, 1, 2)
        interactive = self.config.interaction_version in ("v4", "v5", "v6")
        if interactive:
            other_rollout = other[:, None, None, :].expand(batch, candidates, 2, 5).clone()
        branch_feedback = _feedback(
            name, ego[:, None, :].expand(-1, 2, -1),
            other[:, None, :].expand(-1, 2, -1),
            branch_index.view(1, 2), self.vehicle)
        root_feedback = (probabilities[..., None] * branch_feedback).sum(dim=1)
        for step in range(self.config.horizon):
            if step == 0:
                u = clip_action(root_feedback[:, None, :] + root, self.vehicle)
                first_action = u
                u = u[:, :, None, :].expand(batch, candidates, 2, 2)
            else:
                previous_other = (other_rollout if interactive else
                                  predictions[:, None, :, step - 1, :].expand_as(x))
                u = clip_action(_feedback(name, x, previous_other, branch_index, self.vehicle)
                                + branch, self.vehicle)
            if interactive:
                other_rollout = step_vehicle(other_rollout, other_action(
                    name, other_rollout, branch_index, self.vehicle,
                    ego=x, interaction_version=self.config.interaction_version), self.vehicle)
            x = step_vehicle(x, u, self.vehicle)
            other_at_step = (other_rollout if interactive else
                             predictions[:, None, :, step, :].expand(batch, candidates, 2, 5))
            h = smooth_clearance(x, other_at_step, self.vehicle) - effective_margin
            violation = (-h).clamp_min(0)
            max_violation = torch.maximum(max_violation, violation)
            y_ref, heading_ref, speed_ref = branch_reference(
                name, x, other_at_step, branch_index, self.vehicle)
            error = torch.stack((x[..., 1] - y_ref, x[..., 2] - heading_ref, x[..., 3] - speed_ref, x[..., 4]), dim=-1)
            stage = (error.square() * stage_weights).sum(-1) + (u.square() * input_weights).sum(-1)
            branch_cost = branch_cost + self.vehicle.dt_s * stage + self.config.constraint_penalty * violation.square()
            if step == self.config.horizon - 1:
                branch_cost = branch_cost + (error.square() * terminal_weights).sum(-1)
        objective, q = cvar_value(branch_cost, probabilities[:, None, :], self.config.confidence, validate_probability=False)
        return {
            "objective": objective,
            "action": first_action,
            "q": q,
            "branch_cost": branch_cost,
            "max_violation": max_violation.amax(-1),
            "terminal_state": x[:, :, 0, :],
        }
