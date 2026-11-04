"""Construct comparable branch controllers from one configuration."""
from __future__ import annotations

from .controller import BranchMPC, ControllerConfig
from .ltv_mpc import LTVBranchMPC
from .sampling_variants import CEMBranchMPC, MPPIBranchMPC


CONTROLLERS = {"shooting": BranchMPC, "cem": CEMBranchMPC,
               "mppi": MPPIBranchMPC, "ltv": LTVBranchMPC}


def make_controller(kind: str, config: ControllerConfig, vehicle, *, seed: int,
                    device: str):
    if config.clearance_recovery_protocol != 'constant_v1' and vehicle.dynamics_version != 'actuated_v3':
        raise ValueError('Initial-gap recovery requires the declared actuated_v3 vehicle.')
    if kind not in CONTROLLERS:
        raise ValueError(f"Unknown controller kind: {kind}")
    if config.clearance_recovery_protocol in ('feasible_seed_envelope_v3','observed_gap_envelope_v4','observed_gap_objective_v5','nominal_gate_objective_v6') and kind != 'shooting':
        raise ValueError('This experimental clearance protocol is declared only for shooting; other solvers require separate audits.')
    if kind=='ltv' and config.controller_version=='v7':
        from .ltv_mpc import EffectiveLTVBranchMPC
        return EffectiveLTVBranchMPC(config,vehicle,seed=seed,device=device)
    if config.controller_version in ("v5", "v6", "v7") and kind != "ltv":
        from .planner_v5 import SequenceBranchMPC
        return SequenceBranchMPC(config, vehicle, seed=seed, device=device, algorithm=kind)
    if config.controller_version not in ("legacy", "v5", "v6", "v7"):
        raise ValueError("Unknown controller version.")
    return CONTROLLERS[kind](config, vehicle, seed=seed, device=device)
