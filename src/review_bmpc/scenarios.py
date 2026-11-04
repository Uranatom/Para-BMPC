"""Paired two-mode lane merge and unsignalized intersection scenarios."""
from __future__ import annotations

import math
import torch

from .model import VEHICLE, VehicleConfig, step_vehicle

SCENARIOS = ("merge", "intersection")
INTERSECTION_RESERVE_M = 0.8
INTERSECTION_YIELD_BUFFER_V5_M = 3.5


def intersection_stop_distance(speed: torch.Tensor, vehicle: VehicleConfig) -> torch.Tensor:
    """Conservative discrete-Euler stop distance including one decision latency."""
    decel = vehicle.max_brake_mps2 * vehicle.longitudinal_gain
    return speed * vehicle.dt_s + speed.square() / (2.0 * decel) + .5 * vehicle.dt_s * speed


def initial_states(name: str, count: int, device: torch.device, generator: torch.Generator, *,
                   randomize: bool = True, assert_probability: float | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}")
    noise = torch.rand((count, 4), device=device, generator=generator) - 0.5 if randomize else torch.zeros((count, 4), device=device)
    if name == "merge":
        ego = torch.tensor([0.0, 0.0, 0.0, 8.0, 0.0], device=device).repeat(count, 1)
        other = torch.tensor([13.0, 3.5, 0.0, 7.0, 0.0], device=device).repeat(count, 1)
        ego[:, 0] += 2.0 * noise[:, 0]
        ego[:, 3] += noise[:, 1]
        other[:, 0] += 3.0 * noise[:, 2]
        other[:, 3] += noise[:, 3]
        p_assert = 0.55 + 0.2 * noise[:, 2]
    else:
        ego = torch.tensor([-16.0, 0.0, 0.0, 8.0, 0.0], device=device).repeat(count, 1)
        other = torch.tensor([0.0, -15.0, math.pi / 2, 6.5, 0.0], device=device).repeat(count, 1)
        ego[:, 0] += 2.0 * noise[:, 0]
        ego[:, 3] += noise[:, 1]
        other[:, 1] += 2.0 * noise[:, 2]
        other[:, 3] += noise[:, 3]
        p_assert = 0.5 + 0.2 * noise[:, 2]
    if assert_probability is not None:
        if not 0.0 < assert_probability < 1.0:
            raise ValueError("assert_probability must lie in (0, 1).")
        p_assert = torch.full_like(p_assert, assert_probability)
    probabilities = torch.stack((p_assert, 1.0 - p_assert), dim=-1)
    mode = (torch.rand((count,), device=device, generator=generator) > p_assert).long()
    return ego, other, probabilities, mode


def reference(name: str, ego: torch.Tensor, protocol: str = 'legacy') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if name == "merge":
        length=40. if protocol=='v7' else 24.
        s = ((ego[..., 0] - 8.0) / length).clamp(0.0, 1.0)
        if protocol=='v7':
            y=3.5*(10*s**3-15*s**4+6*s**5)
            slope=3.5/length*30*s.square()*(1-s).square()
        else:
            y = 3.5 * (3.0 * s * s - 2.0 * s * s * s)
            slope = (21.0 / 24.0) * s * (1.0 - s)
        heading = torch.atan(slope)
    else:
        y = torch.zeros_like(ego[..., 0])
        heading = torch.zeros_like(y)
    speed = torch.full_like(y, 8.0)
    return y, heading, speed


def branch_reference(name: str, ego: torch.Tensor, other: torch.Tensor,
                     branch: int | torch.Tensor,
                     vehicle: VehicleConfig = VEHICLE, protocol: str = 'legacy') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ego references conditional on a predicted surrounding-vehicle branch.

    Branch 0 predicts an asserting surrounding vehicle and yields; branch 1
    predicts a yielding vehicle and proceeds. References use only the state
    within the hypothetical branch, never the sampled true mode.
    """
    if ego.shape != other.shape or ego.shape[-1] != 5:
        raise ValueError("Ego and surrounding states must have identical [..., 5] shapes.")
    if isinstance(branch, int) and branch not in (0, 1):
        raise ValueError("Branch indices must be 0 or 1.")
    branch_index = torch.as_tensor(branch, device=ego.device)
    is_yield = branch_index == 0
    pass_y, pass_heading, pass_speed = reference(name, ego,protocol)
    if name == "merge":
        gap = other[..., 0] - ego[..., 0]
        ahead_phase = ((gap - 16.0) / 6.0).clamp(0.0, 1.0)
        behind_phase = ((-gap - 7.0) / 6.0).clamp(0.0, 1.0)
        lane_phase = ((ego[..., 1] - 0.5) / 0.7).clamp(0.0, 1.0)
        phase = torch.maximum(torch.maximum(ahead_phase, behind_phase), lane_phase)
        gate = phase.square() * (3.0 - 2.0 * phase)
        yield_y = gate * pass_y
        yield_heading = torch.atan(gate * torch.tan(pass_heading))
        # A slow observed neighbor can be passed in the current lane before merging.
        slow_neighbor = other[..., 3] <= 3.5
        yield_speed = torch.where(slow_neighbor, pass_speed, 3.0 + 5.0 * gate)
    elif name == "intersection":
        other_yaw = other[..., 2]
        other_center_y = other[..., 1] + vehicle.lr_m * other_yaw.sin()
        other_half_y = (0.5 * vehicle.body_length_m * other_yaw.sin().abs() +
                        0.5 * vehicle.body_width_m * other_yaw.cos().abs())
        crossing_clear = other_center_y - other_half_y > 1.75 + INTERSECTION_RESERVE_M
        ego_yaw = ego[..., 2]
        ego_front_x = (ego[..., 0] + vehicle.lr_m * ego_yaw.cos() +
                       0.5 * vehicle.body_length_m * ego_yaw.cos().abs() +
                       0.5 * vehicle.body_width_m * ego_yaw.sin().abs())
        distance_to_entry = (-1.75 - INTERSECTION_RESERVE_M - ego_front_x).clamp_min(0.0)
        approach_speed = torch.sqrt(6.0 * distance_to_entry).clamp_max(8.0)
        yield_speed = torch.where(crossing_clear, pass_speed, approach_speed)
        yield_y, yield_heading = pass_y, pass_heading
    else:
        raise ValueError(f"Unknown scenario: {name}")
    return (torch.where(is_yield, yield_y, pass_y),
            torch.where(is_yield, yield_heading, pass_heading),
            torch.where(is_yield, yield_speed, pass_speed))


def other_action(name: str, other: torch.Tensor, mode: torch.Tensor,
                 vehicle: VehicleConfig = VEHICLE, *, ego: torch.Tensor | None = None,
                 interaction_version: str = "v3") -> torch.Tensor:
    if interaction_version not in ("v3", "v4", "v5", "v6"):
        raise ValueError(f"Unknown interaction version: {interaction_version}")
    if interaction_version in ("v4", "v5", "v6"):
        if ego is None or ego.shape != other.shape:
            raise ValueError("Interactive traffic requires a matching ego state.")
        speed = other[..., 3]
        asserting = mode == 0
        desired_speed = torch.where(asserting, torch.full_like(speed, 9.5),
                                    torch.full_like(speed, 7.0))
        acceleration = 2.0 * (1.0 - (speed / desired_speed).pow(4))
        if name == "merge":
            # A merging ego becomes a soft leader only when it is ahead.
            center_ego_x = ego[..., 0] + vehicle.lr_m * ego[..., 2].cos()
            center_other_x = other[..., 0] + vehicle.lr_m * other[..., 2].cos()
            lateral_gap = (ego[..., 1] - other[..., 1]).abs()
            phase = ((lateral_gap - 0.75 * vehicle.body_width_m) /
                     vehicle.body_width_m).clamp(0.0, 1.0)
            overlap = 1.0 - phase.square() * (3.0 - 2.0 * phase)
            net_gap = (center_ego_x - center_other_x - vehicle.body_length_m).clamp_min(0.1)
            headway = torch.where(asserting, torch.full_like(speed, 0.7),
                                  torch.full_like(speed, 1.6))
            minimum_gap = torch.where(asserting, torch.full_like(speed, 1.0),
                                      torch.full_like(speed, 2.5))
            desired_gap = minimum_gap + (speed * headway + speed * (speed - ego[..., 3]) /
                                         (2.0 * math.sqrt(2.0 * 3.0))).clamp_min(0.0)
            following = 2.0 * overlap * (desired_gap / net_gap).square()
            acceleration = acceleration - torch.where(center_ego_x > center_other_x,
                                                       following, torch.zeros_like(following))
            # A yielding neighbor leaves an opening even before the ego merges.
            yielding_target = torch.where(other[..., 0] < 28.0,
                                          torch.full_like(speed, 4.5), desired_speed)
            acceleration = torch.where(asserting, acceleration,
                torch.minimum(acceleration, (yielding_target - speed) * 1.5))
        elif name == "intersection":
            body_front = vehicle.lr_m + 0.5 * vehicle.body_length_m
            body_rear = vehicle.lr_m - 0.5 * vehicle.body_length_m
            ego_rear_x = ego[..., 0] + body_rear
            ego_front_x = ego[..., 0] + body_front
            ego_clear = ego_rear_x > 1.75 + INTERSECTION_RESERVE_M
            ego_occupies = ((ego_front_x > -1.75 - INTERSECTION_RESERVE_M) &
                            (ego_rear_x < 1.75 + INTERSECTION_RESERVE_M))
            stop_buffer = INTERSECTION_YIELD_BUFFER_V5_M if interaction_version in ("v5", "v6") else 2.0
            stop_rear_y = -1.75 - INTERSECTION_RESERVE_M - body_front - stop_buffer
            distance_to_stop = (stop_rear_y - other[..., 1]).clamp_min(0.0)
            target_speed = torch.sqrt(2.0 * 4.0 * distance_to_stop).clamp_max(7.0)
            if interaction_version in ("v5", "v6"):
                # Solve v*dt + v^2/(2b) <= distance, rather than using the
                # continuous stopping-speed curve with a clipped late brake.
                decel = vehicle.max_brake_mps2 * vehicle.longitudinal_gain
                target_speed = (torch.sqrt((decel * vehicle.dt_s) ** 2
                    + 2.0 * decel * distance_to_stop) - decel * vehicle.dt_s).clamp_max(7.0)
            must_yield = ~asserting & ~ego_clear
            # An asserting neighbor responds to a vehicle already in the conflict
            # region if it can still stop before the reserved entry.
            can_stop = distance_to_stop >= speed.square() / (2.0 * 4.0) + vehicle.dt_s * speed
            if interaction_version in ("v5", "v6"):
                # The yielding target has extra stand-off. An asserting SV's
                # ability to avoid the actual conflict entry must not be tested
                # against that earlier comfort target after it has passed it.
                distance_to_entry = (-1.75-INTERSECTION_RESERVE_M-body_front-other[...,1]).clamp_min(0.)
                can_stop = distance_to_entry >= intersection_stop_distance(speed, vehicle)
            must_yield = must_yield | (asserting & ego_occupies & can_stop)
            acceleration = torch.where(must_yield,
                torch.minimum(acceleration, (target_speed - speed) / vehicle.dt_s),
                acceleration)
        else:
            raise ValueError(f"Unknown scenario: {name}")
        acceleration = acceleration.clamp(-vehicle.max_brake_mps2,
                                           vehicle.max_accel_mps2)
        return torch.stack((acceleration, torch.zeros_like(acceleration)), dim=-1)
    # Mode 0 asserts; mode 1 yields near the conflict region.
    distance = other[..., 0] if name == "merge" else other[..., 1]
    if name == "intersection":
        body_front_from_rear = vehicle.lr_m + 0.5 * vehicle.body_length_m
        stop_rear_y = -1.75 - INTERSECTION_RESERVE_M - body_front_from_rear - 2.0
        distance_to_stop = (stop_rear_y - distance).clamp_min(0.0)
        stop_deceleration = min(4.0, vehicle.max_brake_mps2)
        target_speed = torch.sqrt(2.0 * stop_deceleration * distance_to_stop).clamp_max(6.5)
        requested = ((target_speed - other[..., 3]) / vehicle.dt_s).clamp(-stop_deceleration, 0.0)
        accel = torch.where(mode == 1, requested, torch.zeros_like(requested))
    else:
        brake = (mode == 1) & (distance < 28.0) & (other[..., 3] > 2.5)
        accel = torch.where(brake, torch.full_like(distance, -2.0), torch.zeros_like(distance))
    return torch.stack((accel, torch.zeros_like(accel)), dim=-1)


def predict_other(name: str, other: torch.Tensor, horizon: int, vehicle: VehicleConfig = VEHICLE, *,
                  ego: torch.Tensor | None = None, interaction_version: str = "v3") -> torch.Tensor:
    states = other[:, None, :].expand(-1, 2, -1).clone()
    mode = torch.arange(2, device=other.device).view(1, 2).expand(other.shape[0], -1)
    if interaction_version in ("v4", "v5", "v6"):
        if ego is None or ego.shape != other.shape:
            raise ValueError("Interactive prediction requires a matching ego state.")
        ego_rollout = ego[:, None, :].expand(-1, 2, -1).clone()
    predictions = []
    for _ in range(horizon):
        action = other_action(name, states, mode, vehicle,
                              ego=ego_rollout if interaction_version in ("v4", "v5", "v6") else None,
                              interaction_version=interaction_version)
        states = step_vehicle(states, action, vehicle)
        if interaction_version in ("v4", "v5", "v6"):
            ego_rollout = step_vehicle(ego_rollout, torch.zeros_like(action), vehicle)
        predictions.append(states)
    return torch.stack(predictions, dim=2)


def update_mode_probabilities(prior: torch.Tensor, one_step_prediction: torch.Tensor,
                              observed_other: torch.Tensor) -> torch.Tensor:
    """Update branch beliefs from the observed surrounding-vehicle motion."""
    velocity_error = (one_step_prediction[..., 3] - observed_other[:, None, 3]) / 0.25
    position_error = (one_step_prediction[..., :2] - observed_other[:, None, :2]) / 0.25
    log_likelihood = -0.5 * (velocity_error.square() + position_error.square().sum(-1))
    posterior = torch.softmax(prior.clamp_min(1e-6).log() + log_likelihood, dim=-1)
    return posterior.clamp_min(1e-5) / posterior.clamp_min(1e-5).sum(-1, keepdim=True)
