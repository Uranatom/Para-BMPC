"""Full-task CMDP environment with separate task reward and collision cost."""
from __future__ import annotations

import math

import torch
from .cuda_timing import StageTimer

from .controller import BranchMPC, policy_parameters
from .model import VEHICLE, VehicleConfig, clearance, safe_brake_action, step_vehicle, effective_actuator_command, road_clearance
from .scenarios import (INTERSECTION_RESERVE_M, initial_states, other_action,
                        predict_other, reference, update_mode_probabilities)


ACTION_DIMS = {"fixed": 0, "cbf_only": 1, "weights_only": 10, "joint": 11,
               "grouped_weights_only": 4, "grouped_joint": 5,
               "longitudinal_weights_only": 3, "longitudinal_joint": 4,
               "effective_weights_only": 4, "effective_joint": 5}


def effective_adaptation(adaptation: str, schema: str = "legacy", scenario: str | None = None) -> str:
    if schema == "effective_v2" and adaptation in ("weights_only", "joint"):
        if scenario not in ("merge", "intersection"):
            raise ValueError("Effective actions require a declared scenario.")
        return ("longitudinal_" if scenario == "intersection" else "effective_") + adaptation
    if schema == "grouped_v1" and adaptation in ("weights_only", "joint"):
        return "grouped_" + adaptation
    return adaptation


def append_controller_context(controller, observation, previous_action, previous_tuner, running_minimum,
                              previous_acceleration=None):
    if controller.config.controller_version not in ("v5", "v6", "v7"):
        return observation
    scale = observation.new_tensor([controller.vehicle.max_brake_mps2,
                                   controller.vehicle.max_steer_rate_rad_s])
    context = (controller.context_observation(len(observation))
               if hasattr(controller, "context_observation") else observation[:, :0])
    items = [observation,context,previous_action/scale,previous_tuner,running_minimum[:,None]/40.]
    if controller.config.controller_version == 'v7':
        acceleration = (torch.zeros(len(observation),device=observation.device)
                        if previous_acceleration is None else previous_acceleration)
        items.append(acceleration[:,None]/5.)
    if hasattr(controller,'recovery_observation'):
        items.append(controller.recovery_observation(len(observation)))
    return torch.cat(items,-1)
SUCCESS = 1
COLLISION = 2
TIMEOUT = 3


def expand_safe_action(action: torch.Tensor | None, adaptation: str,
                       count: int, device: torch.device, *, protocol: str = "legacy",
                       base_parameters=None, effective_action_protocol: str | None = None) -> torch.Tensor | None:
    """Map active dimensions to the legacy controller's 11-parameter interface."""
    if adaptation == "fixed":
        return None
    width = ACTION_DIMS[adaptation]
    if action is None or action.shape != (count, width):
        raise ValueError(f"{adaptation} requires action shaped ({count}, {width}).")
    if not torch.isfinite(action).all():
        raise ValueError('Nonfinite tuner action; inspect learner gradient and loss diagnostics.')
    bounded = effective_action_protocol in ('bounded_compact_v2','neutral_bounded_v3','neutral_bounded_v4')
    if effective_action_protocol not in (None, 'scenario_compact_v1', 'bounded_compact_v2','neutral_bounded_v3','neutral_bounded_v4'):
        raise ValueError('Unsupported effective action protocol.')
    if bounded and (protocol != 'v7' or adaptation not in ('cbf_only', 'longitudinal_weights_only',
            'longitudinal_joint', 'effective_weights_only', 'effective_joint')):
        raise ValueError('Bounded compact actions require the declared v7 effective coordinates.')
    full = torch.zeros((count, 11), device=device, dtype=action.dtype)
    effective = adaptation.startswith(("longitudinal_", "effective_"))
    if adaptation.startswith("longitudinal_"):
        full[:, 6] = action[:, 0]
        full[:, 2] = (action[:, 0] + .5 * action[:, 1]).clamp(-1., 1.)
        full[:, 8] = action[:, 2]
        full[:, 9] = action[:, 2]
    elif adaptation.startswith("effective_"):
        # Fixed lateral stage priorities avoid an empirically inactive actor
        # coordinate under shared branch-reference steering. The terminal
        # ratio still changes both lateral and longitudinal endpoint priorities.
        full[:, [2, 6]] = action[:, :1]
        full[:, 8] = action[:, 1]
        full[:, :4] = (full[:, :4] + .5 * action[:, 2:3]).clamp(-1., 1.)
        full[:, [3, 7, 9]] = action[:, 3:4]
    elif adaptation.startswith("grouped_"):
        # Lateral/heading, speed, effort and terminal/stage ratio.
        full[:, [0, 1, 4, 5]] = action[:, :1]
        full[:, [2, 6]] = action[:, 1:2]
        full[:, [3, 7, 8, 9]] = action[:, 2:3]
        full[:, :4] = (full[:, :4] + .5 * action[:, 3:4]).clamp(-1., 1.)
    elif adaptation != "cbf_only":
        full[:, :10] = action[:, :10]
    if bounded:
        # Restrict each expanded log-weight to +/-log(sqrt(2)), keeping the
        # physical actuator limits and the fixed comparator unchanged.
        full[:, :10] *= .25
    if not adaptation.endswith("weights_only"):
        raw_margin = action[:, -1].clamp(-1.0, 1.0)
        margin_m = 1.675 + .175 * raw_margin if bounded else 1.5 + 0.5 * raw_margin
        if effective_action_protocol == 'neutral_bounded_v3':
            # The negative half-range projects to the common neutral margin.
            margin_m = 1.5 + .35 * raw_margin.clamp_min(0.)
        elif effective_action_protocol == 'neutral_bounded_v4':
            margin_m = 1.5 + .5 * raw_margin.clamp_min(0.)
        kappa = 0.05 / (margin_m - 0.5)
        neutral = (base_parameters[10] if base_parameters is not None else .05) if protocol == "v7" else .1
        full[:, 10] = torch.log(kappa / neutral) / math.log(4.0)
    return full


def body_extents(state: torch.Tensor, vehicle: VehicleConfig) -> tuple[torch.Tensor, ...]:
    """Return center and axis-aligned half extents for a rotated body rectangle."""
    yaw = state[:, 2]
    cosine, sine = yaw.cos(), yaw.sin()
    cx = state[:, 0] + vehicle.lr_m * cosine
    cy = state[:, 1] + vehicle.lr_m * sine
    half_x = 0.5 * vehicle.body_length_m * cosine.abs() + 0.5 * vehicle.body_width_m * sine.abs()
    half_y = 0.5 * vehicle.body_length_m * sine.abs() + 0.5 * vehicle.body_width_m * cosine.abs()
    return cx, cy, half_x, half_y


def geometric_goal(scenario: str, state: torch.Tensor, vehicle: VehicleConfig) -> torch.Tensor:
    cx, cy, half_x, half_y = body_extents(state, vehicle)
    aligned = state[:, 2].abs() <= math.radians(10.0)
    if scenario == "merge":
        return ((cx - half_x > 32.0) & ((cy - 3.5).abs() + half_y <= 1.75) & aligned)
    if scenario == "intersection":
        return ((cx - half_x > 1.75) & (cy.abs() + half_y <= 1.75) & aligned)
    raise ValueError(f"Unknown scenario: {scenario}")


def _potential(scenario: str, state: torch.Tensor) -> torch.Tensor:
    start, finish = (0.0, 35.0) if scenario == "merge" else (-16.0, 5.0)
    return 4.0 * ((state[:, 0] - start) / (finish - start)).clamp(0.0, 1.0)


def intersection_emergency_action(ego: torch.Tensor, other: torch.Tensor,
                                  proposed: torch.Tensor, vehicle: VehicleConfig,
                                  reserve_m: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a braking rule while an observed crossing vehicle blocks entry."""
    ego_cx, _, ego_half_x, _ = body_extents(ego, vehicle)
    _, other_cy, _, other_half_y = body_extents(other, vehicle)
    other_clear = other_cy - other_half_y > 1.75 + reserve_m
    stopped_before_crossing = ((other_cy + other_half_y < -1.75 - reserve_m - 0.5) &
                               (other[:, 3] < 0.3))
    distance_to_entry = -1.75 - reserve_m - (ego_cx + ego_half_x)
    decel = vehicle.max_brake_mps2 * vehicle.longitudinal_gain + vehicle.rolling_decel_mps2
    stopping_distance = ego[:, 3].square() / (2.0 * decel) + vehicle.dt_s * ego[:, 3]
    must_brake = (~other_clear) & (~stopped_before_crossing) & (distance_to_entry <= stopping_distance)
    brake = safe_brake_action(ego, vehicle)
    return torch.where(must_brake[:, None], brake, proposed), must_brake


class SafeTrafficBatch:
    """Compact active vehicles before solving; every slot owns one full episode."""

    def __init__(self, name: str, count: int, controller: BranchMPC, *, seed: int = 1,
                 vehicle: VehicleConfig = VEHICLE, max_steps: int | None = None,
                 randomize: bool = True, reward_discount: float = 0.995,
                 intersection_shield: bool = True, stress_fraction: float = 0.0,
                 interaction_version: str = "v3", assert_probability: float | None = None,
                 reward_version: str = "legacy", action_change_weight: float = .005) -> None:
        if not 0.0 <= stress_fraction < 1.0:
            raise ValueError("stress_fraction must lie in [0, 1).")
        self.name = name
        self.count = count
        self.controller = controller
        self.vehicle = vehicle
        self.device = controller.device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.max_steps = max_steps or (180 if name == "merge" else 100)
        self.randomize = randomize
        self.reward_discount = reward_discount
        self.intersection_shield = intersection_shield
        self.stress_fraction = stress_fraction
        if interaction_version not in ("v3", "v4", "v5", "v6"):
            raise ValueError("Unknown interaction version.")
        if controller.config.interaction_version != interaction_version:
            raise ValueError("Controller and environment interaction versions differ.")
        self.interaction_version = interaction_version
        self.assert_probability = assert_probability if assert_probability is not None else (
            0.7 if interaction_version in ("v4", "v5", "v6") else None)
        if reward_version not in ("legacy", "normalized_v1", "efficiency_v2", "robust_v1", 'robust_balance_v2', 'robust_fast_v2', 'robust_balance_v3', 'robust_light_v4','robust_task_v4') or action_change_weight < 0:
            raise ValueError("Invalid reward version or action-change weight.")
        self.reward_version = reward_version
        self.action_change_weight = action_change_weight
        self.reset()

    def reset(self) -> torch.Tensor:
        self.ego, self.other, self.probabilities, self.mode = initial_states(
            self.name, self.count, self.device, self.generator, randomize=self.randomize,
            assert_probability=self.assert_probability
        )
        if self.interaction_version in ("v5", "v6") and self.name == "intersection":
            self.other[:, 1] -= 1.5
        if self.interaction_version == "v6":
            self._broaden_nominal_starts()
        self.stress_mask = torch.rand(self.count, device=self.device,
                                      generator=self.generator) < self.stress_fraction
        self.stress_mask[0] = False
        self.initial_rejections = 0
        self.initial_candidates = int(self.stress_mask.sum())
        if self.stress_fraction:
            perturb = torch.rand((self.count, 4), device=self.device, generator=self.generator)
            self._apply_stress(self.stress_mask, perturb)
            for _ in range(128):
                invalid = self.stress_mask & (clearance(self.ego, self.other, self.vehicle) <= 0.0)
                if self.interaction_version in ("v4", "v5", "v6"):
                    invalid |= self.stress_mask & (self._stress_feasibility_margin() < 0.0)
                rejected = int(invalid.sum())
                if not rejected:
                    break
                self.initial_rejections += rejected
                self.initial_candidates += rejected
                perturb = torch.rand((self.count, 4), device=self.device, generator=self.generator)
                self._apply_stress(invalid, perturb)
            else:
                raise RuntimeError("Stress-state rejection did not find noncolliding starts.")
        if bool((clearance(self.ego, self.other, self.vehicle) <= 0.0).any()):
            raise RuntimeError("An episode was initialized in collision.")
        self.initial_feasibility_margin_m = self._stress_feasibility_margin()
        self.ego_initial = self.ego.clone()
        self.other_initial = self.other.clone()
        self.previous_clearance = clearance(self.ego, self.other, self.vehicle)
        self.last_q = self.probabilities.clone()
        self.step_index = 0
        self.finished = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self.goal_streak = torch.zeros(self.count, dtype=torch.int32, device=self.device)
        self.task_completed = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        self.terminal_reason = torch.zeros_like(self.goal_streak)
        self.episode_return = torch.zeros(self.count, device=self.device)
        self.episode_cost = torch.zeros(self.count, device=self.device)
        self.previous_acceleration = torch.zeros(self.count, device=self.device)
        self.episode_discounted_return = torch.zeros(self.count, device=self.device)
        self.episode_component_sums = {key: torch.zeros(self.count, device=self.device) for key in
            ('potential','time','lateral','speed','effort','action_change','jerk','success','timeout')}
        self.previous_action = torch.zeros((self.count, 2), device=self.device)
        self.previous_tuner = torch.zeros((self.count, 11), device=self.device)
        self.running_minimum = self.previous_clearance.clone()
        if hasattr(self.controller, "reset_context"):
            self.controller.reset_context(self.count)
        return self.observation()

    def _broaden_nominal_starts(self):
        """Sweep speed/arrival windows within the accepted physical envelope."""
        p = torch.rand((self.count, 4), device=self.device, generator=self.generator)
        self.ego[:, 3] = 5.0 + 6.0 * p[:, 0]
        self.other[:, 3] = 4.0 + 6.0 * p[:, 1]
        if self.name == "intersection":
            from .scenarios import intersection_stop_distance, INTERSECTION_YIELD_BUFFER_V5_M
            front = self.vehicle.lr_m + .5*self.vehicle.body_length_m
            stop = intersection_stop_distance(self.ego[:, 3], self.vehicle)
            distance = stop + 1.0 + 6.0*p[:, 2]
            self.ego[:, 0] = -1.75 - INTERSECTION_RESERVE_M - front - distance
            sv_stop = intersection_stop_distance(self.other[:, 3], self.vehicle)
            arrival_distance = self.other[:, 3] * (distance/self.ego[:, 3] + .3 + 2.0*p[:, 3])
            self.other[:, 1] = (-1.75 - INTERSECTION_RESERVE_M - front -
                torch.maximum(sv_stop + INTERSECTION_YIELD_BUFFER_V5_M + 1., arrival_distance))
        else:
            self.other[:, 0] = self.ego[:, 0] + 6.0 + 8.0*p[:, 2]

    def _apply_stress(self, mask: torch.Tensor, perturb: torch.Tensor) -> None:
        if self.interaction_version in ("v4", "v5", "v6"):
            if self.name == "merge":
                self.ego[mask, 0] = 8.0 + 8.0 * perturb[mask, 0]
                self.ego[mask, 1] = 0.1 + 0.2 * perturb[mask, 1]
                self.other[mask, 0] = self.ego[mask, 0] + 4.0 + 5.0 * perturb[mask, 2]
                self.ego[mask, 3] = 8.0 + 2.0 * perturb[mask, 3]
                self.other[mask, 3] = 6.0 + 2.0 * perturb[mask, 0]
            else:
                speed = 7.5 + perturb[mask, 3]
                self.ego[mask, 3] = speed
                decel = (self.vehicle.max_brake_mps2 * self.vehicle.longitudinal_gain +
                         self.vehicle.rolling_decel_mps2)
                stop_distance = (speed * self.vehicle.dt_s + speed.square() / (2.0 * decel)
                                 + 0.8 + 0.25 + 1.5 * perturb[mask, 0])
                front_offset = self.vehicle.lr_m + 0.5 * self.vehicle.body_length_m
                self.ego[mask, 0] = -1.75 - INTERSECTION_RESERVE_M - front_offset - stop_distance
                # Align arrival windows while leaving a brake-feasible ego stop.
                self.other[mask, 1] = -14.0 + 3.0 * perturb[mask, 1]
                if self.interaction_version in ("v5", "v6"):
                    self.other[mask, 1] = -18.0 - 4.0 * perturb[mask, 1]
                if self.interaction_version == "v6":
                    # Couple the SV start to its sampled speed: resampling only
                    # position cannot make a high-speed, short-distance start feasible.
                    from .scenarios import intersection_stop_distance, INTERSECTION_YIELD_BUFFER_V5_M
                    sv_speed = self.other[mask, 3]
                    sv_distance = (intersection_stop_distance(sv_speed, self.vehicle)
                                   + INTERSECTION_YIELD_BUFFER_V5_M + .25
                                   + 2.0 * perturb[mask, 1])
                    self.other[mask, 1] = (-1.75 - INTERSECTION_RESERVE_M
                                         - front_offset - sv_distance)
            return
        if self.name == "merge":
            self.ego[mask, 0] = 8.0 + 8.0 * perturb[mask, 0]
            self.ego[mask, 1] = 0.8 + 0.95 * perturb[mask, 1]
            self.other[mask, 0] = self.ego[mask, 0] - 4.0 + 8.0 * perturb[mask, 2]
            self.ego[mask, 3] = 8.0 + 3.0 * perturb[mask, 3]
            self.other[mask, 3] = 4.0 + 4.0 * perturb[mask, 0]
        else:
            self.ego[mask, 0] = -13.0 + 3.0 * perturb[mask, 0]
            self.other[mask, 1] = -12.0 + 3.0 * perturb[mask, 1]

    def _stress_feasibility_margin(self) -> torch.Tensor:
        if self.name == "merge":
            # Keeping the current lane and braking is a common safe first
            # action for both neighbor modes when lateral bodies do not overlap.
            return (self.other[:, 1] - self.ego[:, 1]).abs() - self.vehicle.body_width_m - 0.5
        front_offset = self.vehicle.lr_m + 0.5 * self.vehicle.body_length_m
        distance = -1.75 - INTERSECTION_RESERVE_M - self.ego[:, 0] - front_offset
        decel = (self.vehicle.max_brake_mps2 * self.vehicle.longitudinal_gain +
                 self.vehicle.rolling_decel_mps2)
        stop = (self.ego[:, 3] * self.vehicle.dt_s +
                self.ego[:, 3].square() / (2.0 * decel) + 0.8)
        margin = distance - stop
        if self.interaction_version in ("v5", "v6"):
            from .scenarios import INTERSECTION_YIELD_BUFFER_V5_M, intersection_stop_distance
            sv_distance = (-1.75 - INTERSECTION_RESERVE_M - front_offset
                - INTERSECTION_YIELD_BUFFER_V5_M - self.other[:,1])
            margin = torch.minimum(margin, sv_distance - intersection_stop_distance(self.other[:,3],self.vehicle))
        return margin

    def observation(self) -> torch.Tensor:
        separation = clearance(self.ego, self.other, self.vehicle)
        derivative = (separation - self.previous_clearance) / self.vehicle.dt_s
        elapsed = torch.full_like(separation, self.step_index * self.vehicle.dt_s / 18.0)
        remaining = torch.full_like(separation, (self.max_steps - self.step_index) * self.vehicle.dt_s / 18.0)
        active = (~self.finished).to(separation.dtype)
        observation = torch.cat((
            active[:, None], self.ego, self.other, self.ego_initial, self.other_initial,
            elapsed[:, None], separation[:, None], derivative[:, None], self.probabilities,
            self.last_q, remaining[:, None],
        ), dim=-1)
        return append_controller_context(self.controller,observation,self.previous_action,
                                         self.previous_tuner,self.running_minimum,self.previous_acceleration)

    @torch.no_grad()
    def step(self, action: torch.Tensor | None, adaptation: str, *,
             record_plan: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        active = ~self.finished
        ids = active.nonzero(as_tuple=True)[0]
        reward = torch.zeros(self.count, device=self.device)
        cost = torch.zeros_like(reward)
        done = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        collision = done.clone()
        success = done.clone()
        timeout = done.clone()
        failed = done.clone()
        shielded = done.clone()
        residual = torch.zeros_like(reward)
        applied_action = torch.zeros((self.count, 2), device=self.device)
        reward_components = {key: torch.zeros_like(reward) for key in
            ('potential', 'time', 'lateral', 'speed', 'effort', 'action_change', 'jerk', 'success', 'timeout')}
        planned = planned_other = prediction = None
        if ids.numel():
            before = self.ego[ids]
            prior_minimum = self.running_minimum[ids].clone() if self.reward_version.startswith('robust_') else None
            before_clearance = clearance(before, self.other[ids], self.vehicle)
            prediction = predict_other(self.name, self.other[ids], self.controller.config.horizon,
                                       self.vehicle, ego=self.ego[ids],
                                       interaction_version=self.interaction_version)
            active_action = action[ids] if action is not None else None
            full_action = expand_safe_action(active_action, adaptation, len(ids), self.device,
                protocol=self.controller.config.controller_version, base_parameters=self.controller.config.base_parameters,
                effective_action_protocol=self.controller.config.effective_action_protocol)
            controller_adaptation = adaptation.removeprefix("grouped_").removeprefix("longitudinal_").removeprefix("effective_")
            parameters = policy_parameters(full_action, controller_adaptation, len(ids), self.device,
                                           base_parameters=self.controller.config.base_parameters)
            if hasattr(self.controller, "set_active_ids"):
                self.controller.set_active_ids(ids)
            with StageTimer(self.device, enabled=self.controller.config.controller_version == 'v7' and self.step_index%10==0) as solver_timer:
                result = self.controller.solve(self.name, before, self.other[ids], prediction,
                                               self.probabilities[ids], parameters,
                                               record_plan=record_plan)
            if self.name == "intersection" and self.intersection_shield:
                result["action"], must_brake = intersection_emergency_action(
                    before, self.other[ids], result["action"], self.vehicle)
                shielded[ids] = must_brake
                if record_plan and bool(must_brake[0]):
                    state = before[0].expand(2, 5).clone()
                    emergency = []
                    for _ in range(self.controller.config.horizon):
                        state = step_vehicle(state, result["action"][0].expand(2, 2), self.vehicle)
                        emergency.append(state)
                    result["planned_ego"] = torch.stack(emergency, dim=1)
            next_ego = step_vehicle(before, result["action"], self.vehicle)
            next_other = step_vehicle(self.other[ids], other_action(
                self.name, self.other[ids], self.mode[ids], self.vehicle,
                ego=before, interaction_version=self.interaction_version), self.vehicle)
            self.probabilities[ids] = update_mode_probabilities(
                self.probabilities[ids], prediction[:, :, 0], next_other)
            self.ego[ids] = next_ego
            self.other[ids] = next_other
            self.previous_clearance[ids] = before_clearance
            self.last_q[ids] = result["q"]
            separation = clearance(next_ego, next_other, self.vehicle)
            self.running_minimum[ids] = torch.minimum(self.running_minimum[ids], separation)
            collided = separation < 0.0
            goal_now = geometric_goal(self.name, next_ego, self.vehicle)
            streak = torch.where(goal_now, self.goal_streak[ids] + 1, torch.zeros_like(self.goal_streak[ids]))
            self.goal_streak[ids] = streak
            self.task_completed[ids] |= streak >= 3
            succeeded = (streak >= 3) & ~collided
            timed_out = (~collided) & (~succeeded) & (self.step_index + 1 >= self.max_steps)
            just_done = collided | succeeded | timed_out
            terminal = torch.where(collided, COLLISION,
                         torch.where(succeeded, SUCCESS, torch.where(timed_out, TIMEOUT, 0)))
            self.terminal_reason[ids] = terminal.to(self.terminal_reason.dtype)
            self.finished[ids] = just_done

            y_ref, _, speed_ref = reference(self.name, next_ego,self.controller.config.controller_version)
            progress_shaping = self.reward_discount * _potential(self.name, next_ego) - _potential(self.name, before)
            task_reward = (progress_shaping - 0.015 - 0.025 * (next_ego[:, 1] - y_ref).abs()
                           - 0.01 * (next_ego[:, 3] - speed_ref).abs()
                           - 0.005 * result["action"].square().sum(-1)
                           + 20.0 * succeeded.float() - 3.0 * timed_out.float())
            if self.reward_version != 'legacy':
                # Zero absorbing-state potential removes terminal potential bias.
                progress_shaping = self.reward_discount * torch.where(
                    just_done, torch.zeros_like(separation), _potential(self.name, next_ego)) - _potential(self.name, before)
                scale = result["action"].new_tensor([self.vehicle.max_brake_mps2,
                                                      self.vehicle.max_steer_rate_rad_s])
                effort = (result["action"] / scale).square().sum(-1)
                change = ((result["action"] - self.previous_action[ids])/scale).square().sum(-1)
                task_reward = (progress_shaping - .015 - .025*(next_ego[:,1]-y_ref).abs()
                               - .005*effort - self.action_change_weight*change
                               + 20.*succeeded.float() - 3.*timed_out.float())
            components = {'potential': progress_shaping, 'time': torch.full_like(separation,-.015),
                'lateral': -.025*(next_ego[:,1]-y_ref).abs(),
                'speed': -.01*(next_ego[:,3]-speed_ref).abs() if self.reward_version=='legacy' else torch.zeros_like(separation),
                'effort': -.005*(result['action'].square().sum(-1) if self.reward_version=='legacy' else effort),
                'action_change': -self.action_change_weight*change if self.reward_version != 'legacy' else torch.zeros_like(separation),
                'success': 20.*succeeded.float(), 'timeout': -3.*timed_out.float()}
            components['jerk'] = torch.zeros_like(separation)
            observed_acceleration = (next_ego[:,3]-before[:,3])/self.vehicle.dt_s
            observed_jerk = (observed_acceleration-self.previous_acceleration[ids])/self.vehicle.dt_s
            if self.step_index == 0:
                observed_jerk = torch.zeros_like(observed_jerk)
            if self.reward_version == 'efficiency_v2' or self.reward_version.startswith('robust_'):
                # Seconds/10 s and observed jerk/5 m/s^3 are declared task scales.
                components['time'] = torch.full_like(separation, -self.vehicle.dt_s/10.)
                components['jerk'] = -.1*self.vehicle.dt_s*(observed_jerk/5.).square()
                if self.reward_version.startswith('robust_'):
                    from .metrics import observed_margin_cost
                    components['jerk'] *= 4. if self.reward_version in ('robust_v1', 'robust_balance_v3','robust_light_v4','robust_task_v4') else 2.
                    if self.reward_version != 'robust_v1':
                        components['time'] *= 3. if self.reward_version in ('robust_balance_v2', 'robust_balance_v3','robust_light_v4','robust_task_v4') else 5.
                    deficit_weight=1. if self.reward_version=='robust_light_v4' else 0. if self.reward_version=='robust_task_v4' else 2.
                    components['clearance_deficit'] = -deficit_weight * (
                        observed_margin_cost(self.running_minimum[ids], 2., .25) -
                        observed_margin_cost(prior_minimum, 2., .25)).clamp_min(0.)
                task_reward = sum(components.values())
            self.previous_acceleration[ids] = observed_acceleration
            if self.controller.config.controller_version == 'v7':
                self.controller.previous_acceleration[ids] = observed_acceleration
            for key, value in components.items():
                if key not in self.episode_component_sums:
                    self.episode_component_sums[key] = torch.zeros(self.count, device=self.device)
                if key not in reward_components:
                    reward_components[key] = torch.zeros(self.count, device=self.device)
                self.episode_component_sums[key][ids] += value
                reward_components[key][ids] = value
            self.previous_action[ids] = result["action"]
            if full_action is not None:
                self.previous_tuner[ids] = full_action
            reward[ids] = task_reward
            cost[ids] = collided.float()
            done[ids] = just_done
            collision[ids] = collided
            success[ids] = succeeded
            timeout[ids] = timed_out
            failed[ids] = result["failed"]
            residual[ids] = result["max_violation"]
            applied_action[ids] = result["action"]
            planned = result.get("planned_ego")
            planned_other = result.get("planned_other")
            self.episode_discounted_return[ids] += (self.reward_discount**self.step_index)*task_reward
            self.episode_return[ids] += task_reward
            self.episode_cost[ids] += collided.float()
        self.step_index += 1
        diagnostic = {
            "active": active, "collision": collision, "success": success,
            "task_completed": self.task_completed.clone(),
            "timeout": timeout, "terminal_reason": self.terminal_reason.clone(),
            "solver_failed": failed, "constraint_residual_m": residual,
            "shield_intervened": shielded,
            "reward_components": reward_components,
            "actual_clearance_m": clearance(self.ego, self.other, self.vehicle),
            "actual_road_clearance_m":road_clearance(self.ego,self.name,self.vehicle),
            "action": applied_action, "q": self.last_q.clone(),
            "effective_actuator_command": effective_actuator_command(before, result["action"], self.vehicle) if ids.numel() else None,
            "prediction": prediction, "planned_ego": planned,
            "planned_other": planned_other,
            "mode": self.mode.clone(),
            "stress_mask": self.stress_mask.clone(),
            "planner": ({key: value for key, value in result.items() if key in (
                "acceptable_count", "soft_choice_replaced", "candidate_source",
                "candidate_coverage_failure", "weight_ess", "shared_first_action_error",
                "smooth_exact_gap", "branch_reference", "action_before_fallback",
                "sat_violation_m", "yield_viability_violation_m", "road_violation_m",
                "requested_margin_m", "required_endpoint_margin_m", "observed_margin_cap_reduction_m",
                "objective_constraint_violation_m", "nominal_feasibility_margin_m")}
                if ids.numel() else {}),
            "solver_host_ms": solver_timer.host_ms if ids.numel() else 0.,
            "solver_cuda_ms": solver_timer.cuda_ms if ids.numel() else None,
        }
        return self.observation(), reward, cost, done, diagnostic
