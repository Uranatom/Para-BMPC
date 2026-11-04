"""Two-branch linear time-varying MPC with an OSQP quadratic subproblem."""
from __future__ import annotations

import math

import numpy as np
import osqp
from scipy import sparse
import torch

from .controller import BranchMPC, _feedback
from .model import clip_action, safe_brake_action, step_vehicle, clearance, smooth_clearance, road_axis_clearances
from .risk import cvar_value
from .scenarios import branch_reference, other_action


def linearize_bicycle(state: np.ndarray, action: np.ndarray, vehicle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jacobian of the selected discrete model, away from clamp boundaries."""
    if vehicle.dynamics_version == 'actuated_v3':
        with torch.enable_grad():
            point=torch.tensor(state,dtype=torch.float64,requires_grad=True)
            command=torch.tensor(action,dtype=torch.float64,requires_grad=True)
            first,second=torch.autograd.functional.jacobian(
                lambda x,u:step_vehicle(x,u,vehicle),(point,command))
            result=step_vehicle(point,command,vehicle)
        matrix_a=first.detach().numpy();matrix_b=second.detach().numpy()
        return matrix_a,matrix_b,result.detach().numpy()-matrix_a@state-matrix_b@action
    dt = vehicle.dt_s
    _, _, yaw, speed, steer = state
    matrix_a = np.eye(5)
    matrix_a[0, 2] = -dt * speed * math.sin(yaw)
    matrix_a[0, 3] = dt * math.cos(yaw)
    matrix_a[1, 2] = dt * speed * math.cos(yaw)
    matrix_a[1, 3] = dt * math.sin(yaw)
    matrix_a[2, 3] = dt * vehicle.steering_gain * math.tan(steer) / vehicle.wheelbase_m
    matrix_a[2, 4] = (dt * vehicle.steering_gain * speed /
                      (vehicle.wheelbase_m * math.cos(steer) ** 2))
    matrix_b = np.zeros((5, 2))
    matrix_b[3, 0] = dt * vehicle.longitudinal_gain
    matrix_b[4, 1] = dt
    if vehicle.dynamics_version in ("actuated_v1", "actuated_v2"):
        target = steer + dt * action[1]
        target_saturated = abs(target) >= vehicle.max_steer_rad
        applied_rate = (np.clip(target, -vehicle.max_steer_rad, vehicle.max_steer_rad) - steer) / dt
        gain = vehicle.steering_gain + vehicle.yaw_speed_gain * speed ** 2
        matrix_a[2, 3] = dt / vehicle.wheelbase_m * (
            (vehicle.steering_gain + 3 * vehicle.yaw_speed_gain * speed ** 2) * math.tan(steer)
            + vehicle.yaw_rate_response * dt * applied_rate)
        matrix_a[2, 4] = dt * speed * gain / (vehicle.wheelbase_m * math.cos(steer) ** 2)
        matrix_a[3, 3] = 1 - 2 * dt * speed * (vehicle.speed_drag_gain + vehicle.turning_drag_gain * steer ** 2)
        matrix_a[3, 4] = -2 * dt * vehicle.turning_drag_gain * speed ** 2 * steer
        matrix_a[4, 3] = -2 * dt * vehicle.steering_speed_decay * speed * steer
        matrix_a[4, 4] = 1 - dt * (vehicle.steering_decay_s + vehicle.steering_speed_decay * speed ** 2)
        matrix_b[2, 1] = dt ** 2 * speed * vehicle.yaw_rate_response / vehicle.wheelbase_m
        matrix_b[4, 1] = dt * vehicle.steering_rate_gain
        if target_saturated:
            matrix_a[2, 4] -= dt * speed * vehicle.yaw_rate_response / vehicle.wheelbase_m
            matrix_a[4, 4] -= vehicle.steering_rate_gain
            matrix_b[2, 1] = matrix_b[4, 1] = 0.
        if vehicle.dynamics_version == "actuated_v2":
            available=max(speed,0.)/(2*dt*vehicle.longitudinal_gain)
            acceleration=max(action[0],-available)
            if action[0]<-available:
                matrix_a[3,3]-=.5
                matrix_b[3,0]=0.
            drag=vehicle.rolling_decel_mps2+vehicle.speed_drag_gain*speed**2+vehicle.turning_drag_gain*speed**2*steer**2
            unclipped=speed+dt*(vehicle.longitudinal_gain*acceleration-drag*(speed>0))
            next_speed=float(np.clip(unclipped,vehicle.min_speed_mps,vehicle.max_speed_mps))
            if unclipped<vehicle.min_speed_mps or unclipped>vehicle.max_speed_mps:
                matrix_a[3,:]=0.;matrix_b[3,:]=0.
            mean_speed=.5*(speed+next_speed)
            for axis,projection,derivative in ((0,math.cos(yaw),-math.sin(yaw)),
                                               (1,math.sin(yaw),math.cos(yaw))):
                matrix_a[axis,2]=dt*mean_speed*derivative
                matrix_a[axis,3]=.5*dt*(1+matrix_a[3,3])*projection
                matrix_a[axis,4]=.5*dt*matrix_a[3,4]*projection
                matrix_b[axis,:]=.5*dt*matrix_b[3,:]*projection
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float64)[None]
        action_tensor = torch.tensor(action, dtype=torch.float64)[None]
        next_state = step_vehicle(state_tensor, action_tensor, vehicle)[0].numpy()
    affine = next_state - matrix_a @ state - matrix_b @ action
    return matrix_a, matrix_b, affine


def linearized_clearance(state: np.ndarray, other: np.ndarray, vehicle,
                         margin_m: float) -> tuple[float, np.ndarray]:
    """Linearize the conservative smooth SAT margin at the nominal ego state."""
    with torch.enable_grad():
        ego = torch.tensor(state, dtype=torch.float64, requires_grad=True)[None]
        neighbor = torch.tensor(other, dtype=torch.float64)[None]
        margin = smooth_clearance(ego, neighbor, vehicle)[0] - margin_m
        gradient = torch.autograd.grad(margin, ego)[0][0]
    return float(margin.detach()), gradient.detach().numpy()


class LTVBranchMPC(BranchMPC):
    """Sequentially linearized branch MPC with shared first control and CVaR reweighting.

    The optimization is a convex QP for fixed linearization and risk weights.
    Collision clearance uses a linearized soft inequality with explicit slack.
    This is a faithful LTV-MPC formulation, but not byte-equivalent to the
    legacy MATLAB program, which used different slack and CVaR conventions.
    """

    def _reference(self, name, ego, other, branch):
        if self.config.controller_version in ("v5", "v6", "v7"):
            from .planner_v5 import feedback
            return feedback(name, ego, other, branch, self.vehicle,
                            self.config.release_reference,self.config.controller_version)[1]
        return branch_reference(name, ego, other, branch, self.vehicle,self.config.controller_version)

    def _nominal(self, name: str, ego: torch.Tensor, other: torch.Tensor,
                 predictions: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        states_before = []
        states_after = []
        actions = []
        current = ego.expand(2, 5).clone()
        branch_index = torch.arange(2, device=ego.device)
        for step in range(self.config.horizon):
            predicted_other = (other.expand(2, 5) if step == 0 else
                               predictions[0, :, step - 1, :])
            if self.config.controller_version in ("v5", "v6", "v7"):
                from .planner_v5 import feedback
                action, _ = feedback(name,current,predicted_other,branch_index,self.vehicle,
                                     self.config.release_reference,self.config.controller_version)
            else:
                action = clip_action(_feedback(name, current, predicted_other,
                                               branch_index, self.vehicle), self.vehicle)
            actions.append(action.numpy())
            states_before.append(current.numpy())
            current = step_vehicle(current, action, self.vehicle)
            states_after.append(current.numpy())
        return (np.stack(states_before, axis=1), np.stack(states_after, axis=1),
                np.stack(actions, axis=1))

    def _qp(self, name: str, ego: np.ndarray, other: np.ndarray, nominal_before: np.ndarray,
            nominal_after: np.ndarray,
            nominal_action: np.ndarray, parameters: np.ndarray, risk: np.ndarray,
            previous_acceleration=0.,previous_command=0.):
        horizon = self.config.horizon
        jerk_rows=[]
        state_variables = 2 * horizon * 5
        control_variables = 2 * horizon * 2
        slack_variables = 2 * horizon
        dimension = state_variables + control_variables + slack_variables

        def x_index(branch, step, component):
            return (branch * horizon + step) * 5 + component

        def u_index(branch, step, component):
            return state_variables + (branch * horizon + step) * 2 + component

        def s_index(branch, step):
            return state_variables + control_variables + branch * horizon + step

        diagonal = np.full(dimension, 2e-5)
        linear = np.zeros(dimension)
        lower = np.full(dimension, -np.inf)
        upper = np.full(dimension, np.inf)
        eq_rows, eq_cols, eq_data, eq_rhs = [], [], [], []
        ineq_rows, ineq_cols, ineq_data, ineq_rhs = [], [], [], []
        margin = self.config.safety_margin_m + 0.05 / max(parameters[10], 0.01)
        state_weights = parameters[4:8]
        terminal_weights = parameters[:4]
        input_weights = parameters[8:10]
        for branch in range(2):
            for step in range(horizon):
                state_nominal = nominal_before[branch, step] if nominal_before.ndim == 3 else nominal_before[step]
                future_nominal = nominal_after[branch, step] if nominal_after.ndim == 3 else nominal_after[step]
                control_nominal = nominal_action[branch, step] if nominal_action.ndim == 3 else nominal_action[step]
                A, B, affine = linearize_bicycle(state_nominal, control_nominal, self.vehicle)
                rhs = (A @ ego + affine) if step == 0 else affine
                for component in range(5):
                    row = len(eq_rhs)
                    eq_rhs.append(rhs[component])
                    eq_rows.append(row)
                    eq_cols.append(x_index(branch, step, component))
                    eq_data.append(1.0)
                    if step:
                        for previous in range(5):
                            if A[component, previous]:
                                eq_rows.append(row)
                                eq_cols.append(x_index(branch, step - 1, previous))
                                eq_data.append(-A[component, previous])
                    for control in range(2):
                        if B[component, control]:
                            eq_rows.append(row)
                            eq_cols.append(u_index(branch, step, control))
                            eq_data.append(-B[component, control])
                state_reference = torch.tensor(future_nominal, dtype=torch.float64)[None]
                other_reference = torch.tensor(other[branch, step], dtype=torch.float64)[None]
                y_ref, heading_ref, speed_ref = self._reference(
                    name, state_reference, other_reference, branch)
                desired = (float(y_ref[0]), float(heading_ref[0]), float(speed_ref[0]), 0.0)
                for offset, target in enumerate(desired):
                    index = x_index(branch, step, offset + 1)
                    weight = risk[branch] * (self.vehicle.dt_s * state_weights[offset] +
                                               (terminal_weights[offset] if step == horizon - 1 else 0.0))
                    diagonal[index] += 2.0 * weight
                    linear[index] -= 2.0 * weight * target
                for control in range(2):
                    index = u_index(branch, step, control)
                    diagonal[index] += 2.0 * risk[branch] * self.vehicle.dt_s * input_weights[control]
                    lower[index] = -self.vehicle.max_brake_mps2 if control == 0 else -self.vehicle.max_steer_rate_rad_s
                    upper[index] = self.vehicle.max_accel_mps2 if control == 0 else self.vehicle.max_steer_rate_rad_s
                if self.config.controller_version=='v7':
                    dt=self.vehicle.dt_s
                    coefficients={x_index(branch,step,3):1/dt**2}
                    constant=-ego[3]/dt**2-previous_acceleration/dt
                    if step:
                        coefficients[x_index(branch,step-1,3)]=-2/dt**2
                        constant=ego[3]/dt**2 if step==1 else 0.
                        if step>1:coefficients[x_index(branch,step-2,3)]=1/dt**2
                    weight=(risk[branch]*dt*self.config.jerk_weight*parameters[9]
                            /self.config.base_parameters[9]/25.)
                    jerk_rows.append((coefficients,weight))
                    for variable,value in coefficients.items():linear[variable]+=2*weight*constant*value
                    if step==0:
                        variable=u_index(branch,step,0)
                        lower[variable]=max(lower[variable],previous_command-.8)
                        upper[variable]=min(upper[variable],previous_command+.8)
                    else:
                        for sign in (1.,-1.):
                            row=len(ineq_rhs);ineq_rhs.append(.8)
                            ineq_rows.extend((row,row));ineq_cols.extend((u_index(branch,step,0),u_index(branch,step-1,0)))
                            ineq_data.extend((sign,-sign))
                lower[x_index(branch, step, 3)] = self.vehicle.min_speed_mps
                upper[x_index(branch, step, 3)] = self.vehicle.max_speed_mps
                lower[x_index(branch, step, 4)] = -self.vehicle.max_steer_rad
                upper[x_index(branch, step, 4)] = self.vehicle.max_steer_rad
                slack = s_index(branch, step)
                lower[slack] = 0.0
                diagonal[slack] += 2.0 * self.config.constraint_penalty
                h0, gradient = linearized_clearance(future_nominal, other[branch, step], self.vehicle, margin)
                row = len(ineq_rhs)
                ineq_rhs.append(h0 - gradient @ future_nominal)
                for component in range(5):
                    if gradient[component]:
                        ineq_rows.append(row)
                        ineq_cols.append(x_index(branch, step, component))
                        ineq_data.append(-gradient[component])
                ineq_rows.append(row)
                ineq_cols.append(slack)
                ineq_data.append(-1.0)
                if self.config.road_corridor:
                    with torch.enable_grad():
                        point=torch.tensor(future_nominal,dtype=torch.float64,requires_grad=True)
                        gaps=road_axis_clearances(point,name,self.vehicle,smooth=True)
                        for side in (0,1):
                            gradient=torch.autograd.grad(gaps[side],point,retain_graph=side==0)[0].detach().numpy()
                            row=len(ineq_rhs);ineq_rhs.append(float(gaps[side].detach())-gradient@future_nominal)
                            for component in range(5):
                                if gradient[component]:
                                    ineq_rows.append(row);ineq_cols.append(x_index(branch,step,component));ineq_data.append(-gradient[component])
                            ineq_rows.append(row);ineq_cols.append(slack);ineq_data.append(-1.)
        for control in range(2):
            row = len(eq_rhs)
            eq_rhs.append(0.0)
            eq_rows.extend((row, row))
            eq_cols.extend((u_index(0, 0, control), u_index(1, 0, control)))
            eq_data.extend((1.0, -1.0))
        equality = sparse.coo_matrix((eq_data, (eq_rows, eq_cols)),
                                     shape=(len(eq_rhs), dimension)).tocsc()
        inequality = sparse.coo_matrix((ineq_data, (ineq_rows, ineq_cols)),
                                       shape=(len(ineq_rhs), dimension)).tocsc()
        constraint = sparse.vstack((equality, inequality, sparse.eye(dimension)), format="csc")
        constraint_lower = np.r_[eq_rhs, np.full(len(ineq_rhs), -np.inf), lower]
        constraint_upper = np.r_[eq_rhs, ineq_rhs, upper]
        solver = osqp.OSQP()
        quadratic=sparse.diags(diagonal,format='csc')
        for coefficients,weight in jerk_rows:
            indices=list(coefficients);values=list(coefficients.values())
            rr,cc=np.meshgrid(indices,indices,indexing='ij')
            quadratic+=sparse.coo_matrix((2*weight*np.outer(values,values).ravel(),(rr.ravel(),cc.ravel())),shape=(dimension,dimension)).tocsc()
        solver.setup(P=sparse.triu(quadratic,format='csc'), q=linear, A=constraint,
                     l=constraint_lower, u=constraint_upper,
                     verbose=False, eps_abs=1e-4, eps_rel=1e-4,
                     max_iter=4000, polishing=False)
        result = solver.solve(raise_error=False)
        if result.x is None or result.info.status not in ("solved", "solved inaccurate"):
            return None, result.info.status
        states = result.x[:state_variables].reshape(2, horizon, 5)
        controls = result.x[state_variables:state_variables + control_variables].reshape(2, horizon, 2)
        slacks = result.x[state_variables + control_variables:].reshape(2, horizon)
        return (states, controls, slacks), result.info.status

    @torch.no_grad()
    def solve(self, name, ego, other, predictions, probabilities, parameters,
              record_plan: bool = False) -> dict:
        if ego.device.type != "cpu":
            raise ValueError("The LTV-QP comparison solver runs on CPU.")
        first_actions, weights, costs, violations, failed_flags = [], [], [], [], []
        plans = []
        statuses = []
        for index in range(ego.shape[0]):
            history_id=int(self.ids[index]) if hasattr(self,'ids') and self.ids is not None else index
            history_acc=float(self.previous_acceleration[history_id]) if self.config.controller_version=='v7' else 0.
            history_command=float(self.previous_command[history_id,0]) if self.config.controller_version=='v7' else 0.
            current = ego[index:index+1].detach().double()
            nominal_before, nominal_after, nominal_action = self._nominal(
                name, current, other[index:index+1].detach().double(),
                predictions[index:index+1].detach().double())
            prediction = predictions[index].detach().double().numpy()
            parameter = parameters[index].detach().double().numpy()
            p = probabilities[index].detach().double().numpy()
            q = p.copy()
            solution = None
            status = "not solved"
            branch_costs = np.zeros(2)
            for iteration in range(2):
                solution, status = self._qp(name, current[0].numpy(), prediction,
                                            nominal_before, nominal_after, nominal_action, parameter, q,
                                            history_acc,history_command)
                if solution is None:
                    break
                states, controls, slacks = solution
                for branch in range(2):
                    state_tensor = torch.tensor(states[branch], dtype=torch.float64)
                    other_tensor = torch.tensor(prediction[branch], dtype=torch.float64)
                    y_ref, heading_ref, speed_ref = self._reference(
                        name, state_tensor, other_tensor, branch)
                    error = torch.stack((state_tensor[:, 1] - y_ref,
                                         state_tensor[:, 2] - heading_ref,
                                         state_tensor[:, 3] - speed_ref,
                                         state_tensor[:, 4]), dim=-1)
                    effort = torch.tensor(controls[branch], dtype=torch.float64)
                    stage = ((error.square() * torch.tensor(parameter[4:8])).sum(-1)
                             + (effort.square() * torch.tensor(parameter[8:10])).sum(-1))
                    branch_costs[branch] = (self.vehicle.dt_s * stage.sum()
                                            + (error[-1].square() * torch.tensor(parameter[:4])).sum()).item()
                    if self.config.controller_version=='v7':
                        speeds=np.r_[float(current[0,3]),states[branch,:,3]]
                        acc=np.diff(speeds)/self.vehicle.dt_s
                        jerk=np.diff(np.r_[history_acc,acc])/self.vehicle.dt_s
                        branch_costs[branch]+=self.vehicle.dt_s*self.config.jerk_weight*parameter[9]/self.config.base_parameters[9]*float(np.sum((jerk/5.)**2))
                    branch_costs[branch] += self.config.constraint_penalty * float(np.sum(slacks[branch] ** 2))
                _, q_tensor = cvar_value(torch.tensor(branch_costs)[None],
                                         torch.tensor(p)[None], self.config.confidence)
                new_q = q_tensor[0].numpy()
                q = new_q
                if iteration == 1:
                    break
                # Re-linearize each branch around the nonlinear rollout of its QP controls.
                for branch in range(2):
                    state = current.clone()
                    surrounding = other[index:index+1].detach().double().clone()
                    for step in range(self.config.horizon):
                        control = torch.tensor(controls[branch, step], dtype=torch.float64)[None]
                        nominal_before[branch, step] = state[0].numpy()
                        nominal_action[branch, step] = control[0].numpy()
                        if self.config.controller_version in ("v5", "v6", "v7") and self.config.interaction_version in ("v4", "v5", "v6"):
                            surrounding=step_vehicle(surrounding,other_action(name,surrounding,torch.tensor([branch]),
                                self.vehicle,ego=state,interaction_version=self.config.interaction_version),self.vehicle)
                            prediction[branch,step]=surrounding[0].numpy()
                        state = step_vehicle(state, control, self.vehicle)
                        nominal_after[branch, step] = state[0].numpy()
            statuses.append(status)
            if solution is None:
                first_actions.append(safe_brake_action(current.to(ego.dtype), self.vehicle)[0])
                weights.append(torch.tensor(q, dtype=ego.dtype))
                costs.append(torch.tensor([1e6, 1e6], dtype=ego.dtype))
                violations.append(torch.tensor(1e3, dtype=ego.dtype))
                failed_flags.append(True)
                plans.append(current[0].to(ego.dtype).expand(2, self.config.horizon, 5))
                continue
            states, controls, slacks = solution
            first = torch.tensor(controls[0, 0], dtype=ego.dtype)[None]
            predicted = torch.tensor(states, dtype=ego.dtype)
            maximum_violation = torch.zeros((), dtype=ego.dtype)
            margin = self.config.safety_margin_m + 0.05 / max(parameter[10], 0.01)
            audited = None
            if self.config.controller_version in ("v5", "v6", "v7"):
                from .planner_v5 import SequenceBranchMPC
                auditor=SequenceBranchMPC(self.config,self.vehicle,seed=0,device="cpu")
                if self.config.controller_version=='v7':
                    auditor.reset_context(1)
                    auditor.previous_acceleration[0]=history_acc
                    auditor.previous_command[0]=self.previous_command[history_id]
                control_tensor=torch.tensor(controls,dtype=ego.dtype)[None,None]
                audited=auditor._rollout(name,ego[index:index+1],other[index:index+1],
                    predictions[index:index+1],probabilities[index:index+1],parameters[index:index+1],
                    controls=control_tensor,record=record_plan)
                maximum_violation=audited["max_violation"][0,0]
                branch_costs=audited["branch_cost"][0,0].numpy()
                q=audited["q"][0,0].numpy()
                if self.config.controller_version=='v7':
                    control_tensor=audited['controls']
                    first=control_tensor[0,0,0,0][None].clone()
            for branch in range(2):
                if audited is not None:
                    break
                other_branch = predictions[index, branch]
                residual = (margin - clearance(predicted[branch], other_branch, self.vehicle)).clamp_min(0)
                maximum_violation = torch.maximum(maximum_violation, residual.max())
            failed = bool(maximum_violation > self.config.fallback_violation_m)
            if failed:
                first = safe_brake_action(current.to(ego.dtype), self.vehicle)
            first_actions.append(clip_action(first, self.vehicle)[0])
            weights.append(torch.tensor(q, dtype=ego.dtype))
            costs.append(torch.tensor(branch_costs, dtype=ego.dtype))
            violations.append(maximum_violation)
            failed_flags.append(failed)
            if record_plan and index == 0:
                if audited is not None:
                    executed = control_tensor[0, 0].clone()
                    executed[:, 0] = first_actions[-1]
                    if failed:
                        executed[:] = first_actions[-1]
                    recorded = auditor._record_plan(name, ego[index:index+1],
                        other[index:index+1], predictions[index:index+1], executed[None])
                    plans.append(recorded["planned_ego"])
                    continue
                plan = []
                for branch in range(2):
                    state = ego[index].clone()
                    trace = []
                    for step in range(self.config.horizon):
                        action = first_actions[-1] if step == 0 else torch.tensor(controls[branch, step], dtype=ego.dtype)
                        state = step_vehicle(state[None], action[None], self.vehicle)[0]
                        trace.append(state)
                    plan.append(torch.stack(trace))
                plans.append(torch.stack(plan))
        answer = {
            "action": torch.stack(first_actions), "q": torch.stack(weights),
            "branch_cost": torch.stack(costs), "max_violation": torch.stack(violations),
            "failed": torch.tensor(failed_flags, dtype=torch.bool),
            "objective": torch.tensor([float(np.dot(w.numpy(), c.numpy()))
                                        for w, c in zip(weights, costs)], dtype=ego.dtype),
            "solver_status": statuses,
        }
        if record_plan:
            answer["planned_ego"] = plans[0] if plans else torch.zeros((2, self.config.horizon, 5))
            if 'recorded' in locals():
                answer.update(recorded)
        return answer


class EffectiveLTVBranchMPC(LTVBranchMPC):
    """V7 acceleration history; QP has no shifted shooting-proposal cache."""
    def reset_context(self,count):
        self.ids=torch.arange(count,device=self.device)
        self.previous_command=torch.zeros(count,2,device=self.device)
        self.previous_acceleration=torch.zeros(count,device=self.device)
    def set_active_ids(self,ids):self.ids=ids
    def context_observation(self,count):
        return torch.zeros(count,4*self.config.horizon+1,device=self.device)
    def state_dict(self):
        return {'algorithm':'ltv_qp_v7','previous_command':self.previous_command,
                'previous_acceleration':self.previous_acceleration}
    def load_state_dict(self,state):
        if state['algorithm']!='ltv_qp_v7':raise ValueError('LTV history version changed.')
        self.previous_command=state['previous_command'].to(self.device)
        self.previous_acceleration=state['previous_acceleration'].to(self.device)
    def solve(self,*args,**kwargs):
        answer=super().solve(*args,**kwargs)
        self.previous_command[self.ids]=answer['action']
        return answer
