"""Shared rear-drive, front-steer vehicle configuration and low-slip dynamics."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

import torch


@dataclass(frozen=True)
class VehicleConfig:
    # Shared with heavy_sedan_v1 in vehicle_drift/configs/vehicle_tire_params.yaml.
    mass_kg: float = 1615.587
    yaw_inertia_kgm2: float = 2965.570
    lf_m: float = 1.11
    lr_m: float = 1.75
    half_track_m: float = 0.775
    wheel_radius_m: float = 0.325
    body_length_m: float = 4.76
    body_width_m: float = 1.55
    mu: float = 1.0
    dt_s: float = 0.1
    min_speed_mps: float = 0.0
    max_speed_mps: float = 18.0
    max_steer_rad: float = 0.45
    max_steer_rate_rad_s: float = 0.7
    max_accel_mps2: float = 2.5
    max_brake_mps2: float = 5.0
    longitudinal_gain: float = 1.0
    steering_gain: float = 1.0
    rolling_decel_mps2: float = 0.0
    dynamics_version: str = "kinematic_v1"
    steering_rate_gain: float = 1.0
    steering_decay_s: float = 0.0
    steering_speed_decay: float = 0.0
    yaw_speed_gain: float = 0.0
    yaw_rate_response: float = 0.0
    speed_drag_gain: float = 0.0
    turning_drag_gain: float = 0.0

    @property
    def wheelbase_m(self) -> float:
        return self.lf_m + self.lr_m

    @property
    def cover_radius_m(self) -> float:
        # Three circles conservatively cover the full 4.76 by 1.55 m rectangle.
        return math.hypot(self.body_length_m / 6.0, self.body_width_m / 2.0)

    def to_dict(self) -> dict:
        return asdict(self)


VEHICLE = VehicleConfig()


def load_vehicle_calibration(path: str | None) -> VehicleConfig:
    if not path:
        return VEHICLE
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data.get("vehicle_base"), dict) or not {
        "mass_kg", "yaw_inertia_kgm2", "lf_m", "lr_m", "half_track_m", "wheel_radius_m",
        "body_length_m", "body_width_m", "mu", "dt_s", "min_speed_mps", "max_speed_mps",
        "max_steer_rad", "max_steer_rate_rad_s", "max_accel_mps2", "max_brake_mps2",
        "longitudinal_gain", "steering_gain", "rolling_decel_mps2"
    }.issubset(data["vehicle_base"]):
        raise ValueError("Calibration must include the complete shared base vehicle.")
    base = VehicleConfig(**data.get("vehicle_base", {})).to_dict()
    if data.get("status") != "validated" or base != VEHICLE.to_dict():
        raise ValueError("Calibration must be a validated CarSim result for the current shared vehicle configuration.")
    extra = data.get("actuation", {})
    allowed = {"dynamics_version", "steering_rate_gain", "steering_decay_s",
               "steering_speed_decay", "yaw_speed_gain", "yaw_rate_response",
               "speed_drag_gain", "turning_drag_gain"}
    if set(extra) - allowed or extra.get("dynamics_version", "kinematic_v1") not in ("kinematic_v1", "actuated_v1", "actuated_v2", "actuated_v3"):
        raise ValueError("Unknown calibrated actuation schema.")
    limits = data.get("limits", {})
    if set(limits) - {"max_steer_rad", "max_speed_mps", "max_steer_rate_rad_s", "max_accel_mps2"} or any(
            not 0 < float(value) <= getattr(VEHICLE, key) for key, value in limits.items()):
        raise ValueError("Calibrated limits must narrow the declared base envelope.")
    result = replace(VEHICLE, longitudinal_gain=float(data["longitudinal_gain"]),
                     steering_gain=float(data["steering_gain"]),
                     rolling_decel_mps2=float(data["rolling_decel_mps2"]), **extra, **limits)
    if result.dynamics_version == 'actuated_v3' and (
            result.max_speed_mps>11 or result.max_steer_rad>.06 or result.max_steer_rate_rad_s>.15):
        raise ValueError('Actuated v3 requires its declared N10 envelope: speed <=11 m/s, steering <=.06 rad and steering rate <=.15 rad/s.')
    if not all(math.isfinite(value) for value in result.to_dict().values() if isinstance(value, (int, float))):
        raise ValueError("Calibration values must be finite.")
    if result.steering_rate_gain <= 0 or min(result.steering_decay_s, result.steering_speed_decay,
                                           result.speed_drag_gain, result.turning_drag_gain) < 0:
        raise ValueError("Invalid steering response or dissipative drag.")
    return result


def clip_action(u: torch.Tensor, vehicle: VehicleConfig = VEHICLE) -> torch.Tensor:
    return torch.stack(
        (
            u[..., 0].clamp(-vehicle.max_brake_mps2, vehicle.max_accel_mps2),
            u[..., 1].clamp(-vehicle.max_steer_rate_rad_s, vehicle.max_steer_rate_rad_s),
        ),
        dim=-1,
    )


def safe_brake_action(state: torch.Tensor, vehicle: VehicleConfig = VEHICLE) -> torch.Tensor:
    """Taper emergency deceleration near rest to avoid reverse wheel torque."""
    available_deceleration = state[..., 3].clamp_min(0.0) / (
        2.0 * vehicle.dt_s * max(vehicle.longitudinal_gain, 1e-6))
    deceleration = torch.minimum(torch.full_like(available_deceleration,
                                                vehicle.max_brake_mps2),
                                 available_deceleration)
    return clip_action(torch.stack((-deceleration, -2.0 * state[..., 4]), dim=-1), vehicle)


def effective_actuator_command(x: torch.Tensor, u: torch.Tensor,
                               vehicle: VehicleConfig = VEHICLE) -> torch.Tensor:
    """Return the versioned clipped command before actuator response.

    This is a torque-adapter target, not achieved acceleration or steer rate.
    Historical dynamics keep their original command semantics.
    V3 also tapers positive commands using remaining speed headroom.
    """
    command = clip_action(u, vehicle)
    if vehicle.dynamics_version in ("actuated_v2", "actuated_v3"):
        limit = x[...,3].clamp_min(0.) / (2*vehicle.dt_s*vehicle.longitudinal_gain)
        command = torch.stack((torch.maximum(command[...,0],-limit),command[...,1]),-1)
        if vehicle.dynamics_version == "actuated_v3":
            headroom = (vehicle.max_speed_mps-x[...,3]).clamp_min(0.) / (
                2*vehicle.dt_s*vehicle.longitudinal_gain)
            command = torch.stack((torch.minimum(command[...,0], headroom), command[...,1]), -1)
    return command


def step_vehicle(x: torch.Tensor, u: torch.Tensor, vehicle: VehicleConfig = VEHICLE) -> torch.Tensor:
    """Legacy Euler or versioned low-slip actuator-response integration.

    Actuated v2 tapers all braking near rest and uses mean endpoint speed for
    position, retaining current-heading and Euler yaw/steering updates.
    V3 additionally limits positive commands before speed saturation.
    """
    u = effective_actuator_command(x, u, vehicle)
    px, py, yaw, speed, steer = x.unbind(-1)
    dt = vehicle.dt_s
    if vehicle.dynamics_version in ("actuated_v1", "actuated_v2", "actuated_v3"):
        target_steer = (steer + dt * u[..., 1]).clamp(-vehicle.max_steer_rad, vehicle.max_steer_rad)
        applied_rate = (target_steer - steer) / dt
        drag = (vehicle.rolling_decel_mps2 + vehicle.speed_drag_gain * speed.square()
                + vehicle.turning_drag_gain * speed.square() * steer.square())
        yaw_rate = speed / vehicle.wheelbase_m * (
            (vehicle.steering_gain + vehicle.yaw_speed_gain * speed.square()) * torch.tan(steer)
            + vehicle.yaw_rate_response * dt * applied_rate)
        steer_rate = (vehicle.steering_rate_gain * applied_rate
                      - (vehicle.steering_decay_s + vehicle.steering_speed_decay * speed.square()) * steer)
        next_speed = (speed + dt * (vehicle.longitudinal_gain * u[..., 0] - drag * (speed > 0).to(speed.dtype))).clamp(vehicle.min_speed_mps, vehicle.max_speed_mps)
        position_speed = .5 * (speed + next_speed) if vehicle.dynamics_version in ("actuated_v2", "actuated_v3") else speed
        return torch.stack((px + dt * position_speed * torch.cos(yaw), py + dt * position_speed * torch.sin(yaw),
            yaw + dt * yaw_rate,
            next_speed,
            (steer + dt * steer_rate).clamp(-vehicle.max_steer_rad, vehicle.max_steer_rad)), dim=-1)
    return torch.stack(
        (
            px + dt * speed * torch.cos(yaw),
            py + dt * speed * torch.sin(yaw),
            yaw + dt * vehicle.steering_gain * speed * torch.tan(steer) / vehicle.wheelbase_m,
            (speed + dt * (vehicle.longitudinal_gain * u[..., 0] - vehicle.rolling_decel_mps2 * (speed > 0).to(speed.dtype))).clamp(vehicle.min_speed_mps, vehicle.max_speed_mps),
            (steer + dt * u[..., 1]).clamp(-vehicle.max_steer_rad, vehicle.max_steer_rad),
        ),
        dim=-1,
    )


def road_axis_clearances(state: torch.Tensor, scenario: str, vehicle: VehicleConfig = VEHICLE,
                   *, smooth: bool = False) -> torch.Tensor:
    """Body-to-road corridor margin; the smooth version is a lower bound."""
    yaw=state[...,2];sine=yaw.sin();cosine=yaw.cos()
    center=state[...,1]+vehicle.lr_m*sine
    abs_sine=(sine.square()+.01**2).sqrt() if smooth else sine.abs()
    abs_cosine=(cosine.square()+.01**2).sqrt() if smooth else cosine.abs()
    half=.5*vehicle.body_length_m*abs_sine+.5*vehicle.body_width_m*abs_cosine
    top=5.25 if scenario=='merge' else 1.75
    gaps=torch.stack((center-half+1.75,top-center-half),-1)
    return gaps


def road_clearance(state: torch.Tensor, scenario: str, vehicle: VehicleConfig = VEHICLE,
                   *, smooth: bool = False) -> torch.Tensor:
    gaps=road_axis_clearances(state,scenario,vehicle,smooth=smooth)
    return -.02*torch.logsumexp(-gaps/.02,-1) if smooth else gaps.amin(-1)


def _sat_axes(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    direction = torch.stack((x[..., 2].cos(), x[..., 2].sin()), dim=-1)
    lateral = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
    return direction, lateral


def sat_axis_separations(x: torch.Tensor, other: torch.Tensor,
                         vehicle: VehicleConfig = VEHICLE, *,
                         smooth_epsilon_m: float | None = None) -> torch.Tensor:
    """Return four oriented-rectangle SAT gaps; positive means axis separation.

    The state origin is the rear axle. A smooth result is a lower bound on each
    exact axis gap, so its positive sign certifies separation in the model.
    """
    if x.shape != other.shape or x.shape[-1] != 5:
        raise ValueError("SAT states must have identical [..., 5] shapes.")
    own_forward, own_lateral = _sat_axes(x)
    other_forward, other_lateral = _sat_axes(other)
    axes = torch.stack((own_forward, own_lateral, other_forward, other_lateral), dim=-2)
    own_center = x[..., :2] + vehicle.lr_m * own_forward
    other_center = other[..., :2] + vehicle.lr_m * other_forward
    displacement = own_center - other_center
    projection = (axes * displacement[..., None, :]).sum(-1)
    own_long = (axes * own_forward[..., None, :]).sum(-1)
    own_lat = (axes * own_lateral[..., None, :]).sum(-1)
    other_long = (axes * other_forward[..., None, :]).sum(-1)
    other_lat = (axes * other_lateral[..., None, :]).sum(-1)
    if smooth_epsilon_m is None:
        center_absolute = projection.abs()
        radius_absolute = torch.abs
    else:
        if smooth_epsilon_m <= 0:
            raise ValueError("The SAT smoothing epsilon must be positive.")
        # The position projection uses meters; direction projections are dimensionless.
        # Their smoothing constants share a numerical value in the declared SI model.
        epsilon = smooth_epsilon_m
        center_absolute = torch.sqrt(projection.square() + epsilon * epsilon) - epsilon
        radius_absolute = lambda value: torch.sqrt(value.square() + epsilon * epsilon)
    radius = (0.5 * vehicle.body_length_m * (radius_absolute(own_long) + radius_absolute(other_long))
              + 0.5 * vehicle.body_width_m * (radius_absolute(own_lat) + radius_absolute(other_lat)))
    return center_absolute - radius


def clearance(x: torch.Tensor, other: torch.Tensor,
              vehicle: VehicleConfig = VEHICLE) -> torch.Tensor:
    """Exact SAT signed axis margin: positive separated, nonpositive contact."""
    return sat_axis_separations(x, other, vehicle).amax(dim=-1)


def smooth_clearance(x: torch.Tensor, other: torch.Tensor,
                     vehicle: VehicleConfig = VEHICLE, *,
                     epsilon_m: float = 0.01, temperature_m: float = 0.05) -> torch.Tensor:
    """Differentiable conservative lower bound on the exact SAT margin."""
    if temperature_m <= 0:
        raise ValueError("The SAT smoothing temperature must be positive.")
    gaps = sat_axis_separations(x, other, vehicle, smooth_epsilon_m=epsilon_m)
    return temperature_m * torch.logsumexp(gaps / temperature_m, dim=-1) - temperature_m * math.log(4.0)
