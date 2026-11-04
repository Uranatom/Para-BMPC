"""Feasibility-first sequence BMPC with observable stop/release references.

The restart envelope uses the declared acceleration bound, independently of
posterior mode. It is a model check, not a forward-invariance certificate.
"""
from __future__ import annotations
import math
import torch
from .controller import BranchMPC
from .model import clip_action, safe_brake_action, step_vehicle, clearance, smooth_clearance, road_clearance
from .scenarios import branch_reference, other_action, INTERSECTION_RESERVE_M
from .risk import cvar_value


def release_allowed(ego, other, vehicle, protocol="v5"):
    """Compare body-clearance passage time with an immediate SV restart envelope."""
    yaw = other[..., 2]
    front = (other[..., 1] + vehicle.lr_m*yaw.sin() +
             .5*vehicle.body_length_m*yaw.sin().abs() + .5*vehicle.body_width_m*yaw.cos().abs())
    rear = (other[..., 1] + vehicle.lr_m*yaw.sin() -
            .5*vehicle.body_length_m*yaw.sin().abs() - .5*vehicle.body_width_m*yaw.cos().abs())
    body_rear = vehicle.lr_m - .5*vehicle.body_length_m
    distance = (1.75 + INTERSECTION_RESERVE_M - (ego[..., 0]+body_rear)).clamp_min(0.)
    # Deliberately conservative EV acceleration for the release-time estimate.
    ego_accel = max(.1, .5*vehicle.max_accel_mps2*vehicle.longitudinal_gain)
    pass_time = ((ego[..., 3].square()+2*ego_accel*distance).sqrt()-ego[..., 3])/ego_accel
    sv_distance = (-1.75-INTERSECTION_RESERVE_M-front).clamp_min(0.)
    sv_accel = max(.1, vehicle.max_accel_mps2*vehicle.longitudinal_gain)
    reach_time = ((other[..., 3].square()+2*sv_accel*sv_distance).sqrt()-other[..., 3])/sv_accel
    if protocol in ("v6", "v7"):
        # Under reactive interaction v5 an SV that can stop yields to an EV
        # occupying the conflict region. Test entry precedence, then audit
        # both complete reactive branch trajectories in the planner.
        from .scenarios import intersection_stop_distance
        ego_front = (ego[...,0] + vehicle.lr_m*ego[...,2].cos()
            + .5*vehicle.body_length_m*ego[...,2].cos().abs()
            + .5*vehicle.body_width_m*ego[...,2].sin().abs())
        entry_distance = (-1.75-INTERSECTION_RESERVE_M-ego_front).clamp_min(0.)
        entry_time = ((ego[...,3].square()+2*ego_accel*entry_distance).sqrt()-ego[...,3])/ego_accel
        stop_distance = (sv_distance-2.0).clamp_min(0.)
        can_stop = stop_distance + 1e-5 >= intersection_stop_distance(other[...,3],vehicle)
        return ((rear > 1.75+INTERSECTION_RESERVE_M) |
                ((front < -1.75-INTERSECTION_RESERVE_M) & can_stop &
                 ((ego_front > -1.75-INTERSECTION_RESERVE_M) |
                  ((other[...,3] < .3) & (entry_time+vehicle.dt_s < reach_time)))))
    return ((rear > 1.75+INTERSECTION_RESERVE_M) |
            ((front < -1.75-INTERSECTION_RESERVE_M) & (other[..., 3] < .3) &
             (pass_time + vehicle.dt_s < reach_time)))


def feedback(name, ego, other, branch, vehicle, release=True, protocol="v5"):
    y, heading, speed = branch_reference(name, ego, other, branch, vehicle,protocol)
    if name == "intersection" and release:
        speed = torch.where(release_allowed(ego, other, vehicle, protocol), torch.full_like(speed,8.), speed)
    error = torch.atan2((heading-ego[...,2]).sin(),(heading-ego[...,2]).cos())
    if protocol == 'v7':
        # Curvature feedforward and deliberately damped rear-reference tracking.
        # The steering envelope is inherited from the calibrated response map.
        curvature = torch.zeros_like(y)
        if name == 'merge':
            s=((ego[...,0]-8.)/40.).clamp(0.,1.)
            slope=3.5/40.*30*s.square()*(1-s).square()
            second=3.5/40.**2*60*s*(1-s)*(1-2*s)
            curvature=second/(1+slope.square()).pow(1.5)
            _, pass_heading, _ = branch_reference(name,ego,other,1,vehicle,protocol)
            gate=torch.where(pass_heading.abs()>1e-5,
                heading.tan()/pass_heading.tan().clamp_min(1e-5),torch.ones_like(y))
            curvature=curvature*gate.clamp(0.,1.)
        steer=(torch.atan(vehicle.wheelbase_m*curvature)+.03*(y-ego[...,1])+.5*error).clamp(
            -vehicle.max_steer_rad,vehicle.max_steer_rad)
        steering_rate=(6.*(steer-ego[...,4])).clamp(-.7,.7)
    else:
        steer = (.13*(y-ego[...,1])+.8*error).clamp(-.38,.38)
        steering_rate=(2.2*(steer-ego[...,4])).clamp(-.7,.7)
    return clip_action(torch.stack(((speed-ego[...,3]).clamp(-4.,2.),
        steering_rate), -1),vehicle), (y,heading,speed)


def select_candidate(objective, violation, tolerance, feasibility_first=True):
    """Lexicographic finite/acceptable/violation/objective selection."""
    finite = torch.isfinite(objective) & torch.isfinite(violation)
    acceptable = finite & (violation <= tolerance)
    soft = objective.masked_fill(~finite,float("inf")).argmin(-1)
    if not feasibility_first:
        return soft, acceptable
    min_violation = violation.masked_fill(~finite,float("inf")).amin(-1,keepdim=True)
    pool = torch.where(acceptable.any(-1,keepdim=True), acceptable,
                       finite & (violation == min_violation))
    return objective.masked_fill(~pool,float("inf")).argmin(-1), acceptable


class SequenceBranchMPC(BranchMPC):
    """Shared first physical action, branch-specific time-varying controls."""
    def __init__(self,*args,algorithm="shooting",**kwargs):
        super().__init__(*args,**kwargs)
        if self.config.candidates < 4 or self.config.horizon < 2:
            raise ValueError("Sequence BMPC requires at least four candidates and two steps.")
        self.algorithm=algorithm
        self.ids=None
        self.warm=None
        self.warm_valid=None
        self.previous_command=None
        self.previous_acceleration=None
        self.margin_reference=None
        self.margin_reference_valid=None

    def reset_context(self,count):
        self.warm=torch.zeros((count,2,self.config.horizon,2),device=self.device)
        self.warm_valid=torch.zeros(count,dtype=torch.bool,device=self.device)
        self.ids=torch.arange(count,device=self.device)
        self.previous_command=torch.zeros((count,2),device=self.device)
        self.previous_acceleration=torch.zeros(count,device=self.device)
        if self.config.clearance_recovery_protocol in ('persistent_gap_ramp_v2','feasible_seed_envelope_v3'):
            self.margin_reference=torch.zeros(count,device=self.device)
            self.margin_reference_valid=torch.zeros(count,dtype=torch.bool,device=self.device)

    def set_active_ids(self,ids):
        self.ids=ids

    def context_observation(self,count):
        if self.warm is None or len(self.warm)!=count:self.reset_context(count)
        scale=self.warm.new_tensor([self.vehicle.max_brake_mps2,self.vehicle.max_steer_rate_rad_s])
        return torch.cat(((self.warm/scale).flatten(1),self.warm_valid.float()[:,None]),-1)

    def state_dict(self):
        state={"warm":self.warm,"warm_valid":self.warm_valid,"algorithm":self.algorithm}
        if self.config.controller_version=='v7':
            state.update(previous_command=self.previous_command,previous_acceleration=self.previous_acceleration)
        if self.config.clearance_recovery_protocol in ('persistent_gap_ramp_v2','feasible_seed_envelope_v3'):
            state.update(clearance_recovery_protocol=self.config.clearance_recovery_protocol,
                clearance_recovery_rate_mps=self.config.clearance_recovery_rate_mps,
                planner_seed_protocol=self.config.planner_seed_protocol,
                margin_reference=self.margin_reference,margin_reference_valid=self.margin_reference_valid)
        return state

    def load_state_dict(self,state):
        if state["algorithm"]!=self.algorithm:raise ValueError("Planner algorithm changed.")
        expected=self.config.clearance_recovery_protocol if self.config.clearance_recovery_protocol in ('persistent_gap_ramp_v2','feasible_seed_envelope_v3') else 'legacy'
        if state.get('clearance_recovery_protocol','legacy') != expected:
            raise ValueError('Planner recovery memory protocol changed.')
        if expected != 'legacy':
            if state.get('clearance_recovery_rate_mps') != self.config.clearance_recovery_rate_mps:
                raise ValueError('Planner recovery rate changed.')
            if state.get('planner_seed_protocol') != self.config.planner_seed_protocol:
                raise ValueError('Planner seed memory protocol changed.')
            self.margin_reference=state['margin_reference'].to(self.device)
            self.margin_reference_valid=state['margin_reference_valid'].to(self.device)
        self.warm=state["warm"].to(self.device);self.warm_valid=state["warm_valid"].to(self.device)
        self.previous_command=state.get("previous_command",torch.zeros((len(self.warm),2))).to(self.device)
        self.previous_acceleration=state.get("previous_acceleration",torch.zeros(len(self.warm))).to(self.device)

    def recovery_observation(self,count):
        """Expose persistent memory after the complete legacy observation prefix."""
        if self.config.clearance_recovery_protocol not in ('persistent_gap_ramp_v2','feasible_seed_envelope_v3'):
            return torch.empty((count,0),device=self.device)
        if self.margin_reference is None or len(self.margin_reference)!=count:self.reset_context(count)
        return torch.stack((self.margin_reference/2.,self.margin_reference_valid.float()),-1)

    def _rollout(self,name,ego,other,predictions,probabilities,parameters,controls=None,
                 perturb=None,seed_kind=None,record=False,warm_controls=None,warm_mask=None):
        b,c=(ego.shape[0],controls.shape[1] if controls is not None else perturb.shape[1])
        x=ego[:,None,None].expand(b,c,2,5).clone()
        z=other[:,None,None].expand_as(x).clone()
        branches=torch.arange(2,device=self.device).view(1,1,2)
        cost=x.new_zeros((b,c,2));violation=cost.clone()
        sat_violation=cost.clone();yield_violation=cost.clone()
        road_violation=cost.clone()
        min_exact=torch.full_like(cost,float("inf"));max_gap=cost.clone()
        margin=self.config.safety_margin_m+.05/parameters[:,None,None,10].clamp_min(.01)
        persistent = self.config.clearance_recovery_protocol in ('persistent_gap_ramp_v2','feasible_seed_envelope_v3')
        recovery = self.config.clearance_recovery_protocol != 'constant_v1'
        margin_start = (torch.minimum(margin, smooth_clearance(ego,other,self.vehicle).clamp_min(1.)[:,None,None])
                        if recovery else margin)
        required_endpoint = margin
        nominal_gate = self.config.clearance_recovery_protocol == 'nominal_gate_objective_v6'
        observed_envelope = self.config.clearance_recovery_protocol in ('observed_gap_envelope_v4', 'observed_gap_objective_v5', 'nominal_gate_objective_v6')
        if nominal_gate:
            nominal_margin = self.config.safety_margin_m + .05/max(self.config.base_parameters[10],.01)
            nominal_start = torch.minimum(torch.full_like(margin,nominal_margin),
                smooth_clearance(ego,other,self.vehicle).clamp_min(1.)[:,None,None])
            objective_violation = cost.clone()
        if observed_envelope:
            # This experimental ceiling uses only the present conservative
            # separation. Only the explicitly versioned objective variant
            # below also changes the soft scoring residual when capped.
            ceiling = (smooth_clearance(ego,other,self.vehicle) + .1).clamp_min(1.5)
            required_endpoint = torch.minimum(margin, ceiling[:,None,None])
            margin_start = torch.minimum(margin_start, required_endpoint)
        desired_violation = cost.clone()
        us=[];xs=[];zs=[];refs=[]
        envelope=self.config.clearance_recovery_protocol == 'feasible_seed_envelope_v3'
        smooth_profiles=[];gate_profiles=[]
        ids = self.ids if self.ids is not None and len(self.ids)==b else torch.arange(b,device=self.device)
        if persistent:
            saved=self.margin_reference[ids][:,None,None]
            available=self.margin_reference_valid[ids][:,None,None]
            margin_start=torch.where(available,torch.minimum(margin,saved.clamp_min(1.)),margin_start)
        jerk_coefficient = self.config.jerk_weight * parameters[:,None,None,9] / self.config.base_parameters[9] if self.config.jerk_weight else None
        previous = self.previous_acceleration[ids][:,None,None].expand(b,c,2) if jerk_coefficient is not None else None
        previous_command = self.previous_command[ids][:,None,None,0].expand(b,c,2) if jerk_coefficient is not None else None
        for k in range(self.config.horizon):
            u,ref=feedback(name,x,z,branches,self.vehicle,self.config.release_reference,self.config.controller_version)
            lane_steering = u[...,1].clone()
            if controls is not None:u=controls[:,:,:,k]
            if perturb is not None:
                knot=min(2,3*k//self.config.horizon)
                u=u+perturb[:,:,:,knot]
                if self.config.explicit_seeds:
                    braking=safe_brake_action(x,self.vehicle)
                    passing=torch.stack(((8.-x[...,3]).clamp(-4.,2.),u[...,1]),-1)
                    u=torch.where((seed_kind==1)[...,None],braking,u)
                    u=torch.where((seed_kind==2)[...,None],passing,u)
            if warm_controls is not None:
                u[:,3]=torch.where(warm_mask[:,None,None],warm_controls[:,:,k],u[:,3])
            if ((name == "intersection" and self.config.controller_version == "v6") or self.config.controller_version=='v7') and perturb is not None:
                # Search longitudinal branch controls while tracking each
                # branch's lateral reference with calibrated steering feedback;
                # random lateral offsets cannot trade lane drift for margin.
                u[...,1] = lane_steering
            u=clip_action(u,self.vehicle)
            if self.config.controller_version == 'v7':
                # Smooth candidate accelerations; emergency fallback remains
                # unrestricted and is still logged independently.
                u[...,0] = previous_command + (u[...,0]-previous_command).clamp(-.8,.8)
            if k==0:
                shared=(u*probabilities[:,None,:,None]).sum(2)
                if controls is not None:shared=controls[:,:,0,0]
                u=shared[:,:,None].expand(b,c,2,2)
            if self.config.interaction_version in ("v4", "v5", "v6"):
                z=step_vehicle(z,other_action(name,z,branches,self.vehicle,ego=x,interaction_version=self.config.interaction_version),self.vehicle)
            else:z=predictions[:,None,:,k].expand_as(x)
            previous_speed=x[...,3].clone()
            x=step_vehicle(x,u,self.vehicle)
            observed_acceleration=(x[...,3]-previous_speed)/self.vehicle.dt_s
            exact=clearance(x,z,self.vehicle);smooth=smooth_clearance(x,z,self.vehicle)
            # The desired margin still enters the objective. Only its
            # feasibility schedule recovers from observed initial geometry;
            # the conservative prediction floor never falls below 1 m.
            gate_margin = (margin-(margin-margin_start)*(.9**(k+1))).clamp_min(1.) if recovery else margin
            if observed_envelope:
                gate_margin = (required_endpoint-(required_endpoint-margin_start)*(.9**(k+1))).clamp_min(1.)
            if persistent:
                gate_margin=torch.minimum(margin,margin_start+(k+1)*self.vehicle.dt_s*self.config.clearance_recovery_rate_mps).clamp_min(1.)
            if envelope:
                smooth_profiles.append(smooth)
                gate_profiles.append(gate_margin)
            if nominal_gate:
                objective_residual=(gate_margin-smooth).clamp_min(0.)
                gate_margin=(nominal_margin-(nominal_margin-nominal_start)*(.9**(k+1))).clamp_min(1.)
            residual=(gate_margin-smooth).clamp_min(0.)
            desired_residual=(margin-smooth).clamp_min(0.)
            desired_violation=torch.maximum(desired_violation,desired_residual)
            sat_violation=torch.maximum(sat_violation,residual)
            if self.config.road_corridor:
                outside=(-road_clearance(x,name,self.vehicle,smooth=True)).clamp_min(0.)
                road_violation=torch.maximum(road_violation,outside)
                residual=torch.maximum(residual,outside)
            if name == "intersection" and self.config.controller_version in ("v6", "v7"):
                from .scenarios import intersection_stop_distance
                ego_front = (x[...,0]+self.vehicle.lr_m*x[...,2].cos()
                    + .5*self.vehicle.body_length_m*x[...,2].cos().abs()
                    + .5*self.vehicle.body_width_m*x[...,2].sin().abs())
                distance = (-1.75-INTERSECTION_RESERVE_M-ego_front).clamp_min(0.)
                waiting = (branches==0) & ~release_allowed(x,z,self.vehicle,"v6")
                viability = (intersection_stop_distance(x[...,3],self.vehicle)-distance).clamp_min(0.)
                # Finite-horizon clearance alone can admit entry into a state
                # from which later braking cannot keep the EV outside conflict.
                waiting_violation=torch.where(waiting,viability,torch.zeros_like(viability))
                yield_violation=torch.maximum(yield_violation,waiting_violation)
                residual=torch.maximum(residual,waiting_violation)
            violation=torch.maximum(violation,residual);min_exact=torch.minimum(min_exact,exact)
            max_gap=torch.maximum(max_gap,exact-smooth)
            _,ref=feedback(name,x,z,branches,self.vehicle,self.config.release_reference,self.config.controller_version)
            y,heading,speed=ref
            error=torch.stack((x[...,1]-y,x[...,2]-heading,x[...,3]-speed,x[...,4]),-1)
            stage=(error.square()*parameters[:,None,None,4:8]).sum(-1)+(u.square()*parameters[:,None,None,8:10]).sum(-1)
            if jerk_coefficient is not None:
                stage += jerk_coefficient * ((observed_acceleration-previous)/(5.*self.vehicle.dt_s)).square()
                previous = observed_acceleration
                previous_command = u[...,0]
            penalty_residual = torch.maximum(residual,desired_residual) if recovery else residual
            if nominal_gate:
                # Score the original adaptive objective on identical candidate
                # controls, while auditing the unchanged nominal feasibility
                # schedule, road residual and yielding viability separately.
                objective_residual=torch.maximum(objective_residual,residual)
                objective_violation=torch.maximum(objective_violation,objective_residual)
                penalty_residual=torch.where(margin > required_endpoint,objective_residual,
                    torch.maximum(objective_residual,desired_residual))
            if self.config.clearance_recovery_protocol == 'observed_gap_objective_v5':
                # Match the soft objective to this declared feasibility gate.
                # The learner's episode-level clearance objective is separate.
                penalty_residual = torch.where(margin > required_endpoint, residual, penalty_residual)
            cost+=self.vehicle.dt_s*stage+self.config.constraint_penalty*penalty_residual.square()
            if k==self.config.horizon-1:cost+=(error.square()*parameters[:,None,None,:4]).sum(-1)
            us.append(u);xs.append(x);zs.append(z);refs.append(torch.stack(ref,-1))
        objective,q=cvar_value(cost,probabilities[:,None],self.config.confidence,validate_probability=False)
        envelope_info={}
        if envelope:
            # A single common envelope is witnessed by one original candidate,
            # never by a pointwise maximum assembled from different controls.
            smooth_history=torch.stack(smooth_profiles,-1)
            desired_gate=torch.stack(gate_profiles,-1).expand(b,c,2,self.config.horizon)
            floor_sat=(1.-smooth_history).clamp_min(0.).amax(-1)
            physical=torch.maximum(floor_sat,torch.maximum(road_violation,yield_violation)).amax(-1)
            floor_ok=torch.isfinite(objective)&torch.isfinite(physical)&(physical<=self.config.fallback_violation_m)
            minimum=smooth_history.amin(dim=(-1,-2))
            largest=minimum.masked_fill(~floor_ok,-float('inf')).amax(-1,keepdim=True)
            pool=floor_ok&(minimum==largest)
            reference=objective.masked_fill(~pool,float('inf')).argmin(-1)
            row=torch.arange(b,device=self.device)
            witness=smooth_history[row,reference].amin(-2)[:,None,None,:]
            common=torch.minimum(desired_gate,witness).clamp_min(1.)
            available=floor_ok.any(-1)
            common=torch.where(available[:,None,None,None],common,desired_gate)
            sat_violation=(common-smooth_history).clamp_min(0.).amax(-1)
            violation=torch.maximum(sat_violation,torch.maximum(road_violation,yield_violation))
            envelope_info={'required_margin_profile_m':common[:,:,0,:],
                'reference_seed_available':available[:,None].expand(b,c),
                'reference_seed_index':reference[:,None].expand(b,c)}
        return {"objective":objective,"q":q,"branch_cost":cost,"max_violation":violation.amax(-1),
            "sat_violation_m":sat_violation.amax(-1),"yield_viability_violation_m":yield_violation.amax(-1),
            "road_violation_m":road_violation.amax(-1),
            "desired_margin_violation_m":desired_violation.amax(-1),
            "branch_max_violation":violation,"minimum_exact_clearance":min_exact.amin(-1),
            "smooth_exact_gap":max_gap.amax(-1),"controls":torch.stack(us,3),
            **envelope_info,
            **({'objective_constraint_violation_m':objective_violation.amax(-1)} if nominal_gate else {}),
            **({"states":torch.stack(xs,3),"others":torch.stack(zs,3),"references":torch.stack(refs,3)} if record else {})}

    def _record_plan(self,name,ego,other,predictions,controls):
        """Render the audited controls without repeating objective/SAT scoring."""
        x=ego[0].expand(2,5).clone();z=other[0].expand(2,5).clone()
        branches=torch.arange(2,device=self.device)
        states=[];neighbors=[];references=[]
        for k in range(self.config.horizon):
            if self.config.interaction_version in ("v4", "v5", "v6"):
                z=step_vehicle(z,other_action(name,z,branches,self.vehicle,ego=x,
                    interaction_version=self.config.interaction_version),self.vehicle)
            else:z=predictions[0,:,k]
            x=step_vehicle(x,controls[0,:,k],self.vehicle)
            _,ref=feedback(name,x,z,branches,self.vehicle,self.config.release_reference,self.config.controller_version)
            states.append(x);neighbors.append(z);references.append(torch.stack(ref,-1))
        return dict(planned_ego=torch.stack(states,1),planned_other=torch.stack(neighbors,1),
                    branch_reference=torch.stack(references,1))

    @torch.no_grad()
    def solve(self,name,ego,other,predictions,probabilities,parameters,record_plan=False):
        b=len(ego);c=self.config.candidates;row=torch.arange(b,device=self.device)
        if self.warm is None:self.reset_context(b)
        ids=self.ids if self.ids is not None and len(self.ids)==b else row
        center=torch.zeros((b,2,3,2),device=self.device)
        scale=center.new_tensor([.75,.08] if self.config.controller_version == 'v7' else [1.8,.28])
        best=None;best_index=None;acceptable_total=torch.zeros(b,device=self.device)
        replaced=torch.zeros(b,dtype=torch.bool,device=self.device);weight_ess=torch.zeros(b,device=self.device)
        shifted=None;warm_mask=None
        if self.config.warm_start:
            shifted=torch.cat((self.warm[ids,:,1:],self.warm[ids,:,-1:]),2)
            shifted[:,:,0]=(shifted[:,:,0]*probabilities[:,:,None]).sum(1)[:,None]
            warm_mask=self.warm_valid[ids]
        for iteration in range(self.config.cem_iterations):
            offsets=center[:,None]+torch.randn((b,c,2,3,2),device=self.device,generator=self.generator)*scale
            offsets[:,0]=0.;kind=torch.zeros((b,c,2),dtype=torch.long,device=self.device)
            if self.config.explicit_seeds:kind[:,1]=1;kind[:,2]=2
            result=self._rollout(name,ego,other,predictions,probabilities,parameters,perturb=offsets,
                                seed_kind=kind,warm_controls=shifted,warm_mask=warm_mask)
            index,acceptable=select_candidate(result["objective"],result["max_violation"],self.config.fallback_violation_m,self.config.feasibility_first)
            acceptable_total+=acceptable.sum(-1)
            replaced|=index!=result["objective"].argmin(-1)
            chosen={k:v[row,index] for k,v in result.items()}
            if best is None:best=chosen;best_index=index
            else:
                old_ok=best["max_violation"]<=self.config.fallback_violation_m
                new_ok=chosen["max_violation"]<=self.config.fallback_violation_m
                improve=((new_ok&~old_ok)|((new_ok==old_ok)&torch.where(new_ok,
                    chosen["objective"]<best["objective"],(chosen["max_violation"]<best["max_violation"])|
                    ((chosen["max_violation"]==best["max_violation"])&(chosen["objective"]<best["objective"])))))
                best={k:torch.where(improve.view(b,*([1]*(v.ndim-1))),v,best[k]) for k,v in chosen.items()}
                best_index=torch.where(improve,index,best_index)
            if self.algorithm=="cem":
                ranking=result["objective"]+torch.where(acceptable.any(-1,keepdim=True)&~acceptable,float("inf"),0.)
                fitting=torch.ones((b,c),dtype=torch.bool,device=self.device)
                if self.config.explicit_seeds:fitting[:,1:3]=False
                if warm_mask is not None:fitting[:,3]&=~warm_mask
                ranking=ranking.masked_fill(~fitting,float('inf'))
                elite=ranking.topk(max(2,c//5),largest=False).indices
                selected=offsets.gather(1,elite[...,None,None,None].expand(-1,-1,2,3,2))
                elite_valid=torch.isfinite(ranking.gather(1,elite))
                w=elite_valid.float()/elite_valid.sum(1,keepdim=True).clamp_min(1)
                new_center=(selected*w[...,None,None,None]).sum(1)
                any_elite=elite_valid.any(1)
                new_scale=((selected-new_center[:,None]).square()*w[...,None,None,None]).sum(1).sqrt().clamp_min(.03)[:,None]
                center=torch.where(any_elite[:,None,None,None],new_center,center)
                scale=torch.where(any_elite[:,None,None,None,None],new_scale,scale*.7)
            elif self.algorithm=="mppi":
                admissible=torch.where(acceptable.any(-1,keepdim=True),acceptable,torch.isfinite(result["objective"]))
                score=result["objective"].masked_fill(~admissible,float("inf"))
                no_finite=~torch.isfinite(score).any(-1,keepdim=True)
                score=torch.where(no_finite,torch.zeros_like(score),score)
                weights=torch.softmax(-(score-score.amin(-1,keepdim=True))/3.,-1)
                weight_ess=1./weights.square().sum(-1)
                averaged=(weights[...,None,None,None]*result["controls"]).sum(1)
                audit=self._rollout(name,ego,other,predictions,probabilities,parameters,controls=averaged[:,None])
                # Averaging feasible controls is not a feasibility-preserving operation.
                avg_ok=torch.isfinite(audit["objective"][:,0])&(audit["max_violation"][:,0]<=self.config.fallback_violation_m)
                improve=avg_ok & ((best["max_violation"]>self.config.fallback_violation_m)|(audit["objective"][:,0]<best["objective"]))
                best={k:torch.where(improve.view(b,*([1]*(v.ndim-1))),audit[k][:,0],v) for k,v in best.items()}
                best_index=torch.where(improve,torch.full_like(best_index,-1),best_index)
                center=(weights[...,None,None,None]*offsets).sum(1);scale=scale*.7
            else:
                center=offsets[row,index]
                explicit=(index==1)|(index==2) if self.config.explicit_seeds else torch.zeros_like(index,dtype=torch.bool)
                if warm_mask is not None:explicit|=(index==3)&warm_mask
                center=torch.where(explicit[:,None,None,None],torch.zeros_like(center),center)
                scale=scale*.5
        failed=(~torch.isfinite(best["objective"]))|(best["max_violation"]>self.config.fallback_violation_m)
        proposed=best["controls"][:,0,0].clone()
        action=torch.where(failed[:,None],safe_brake_action(ego,self.vehicle),proposed)
        if self.config.planner_seed_protocol == 'retained_seed_v1':
            self.warm[ids]=torch.where(failed[:,None,None,None],shifted,best['controls'])
            self.warm_valid[ids]=torch.where(failed,warm_mask,torch.ones_like(failed))
        else:
            self.warm[ids]=torch.where(failed[:,None,None,None],torch.zeros_like(best["controls"]),best["controls"])
            self.warm_valid[ids]=~failed
        if self.config.clearance_recovery_protocol in ('persistent_gap_ramp_v2','feasible_seed_envelope_v3'):
            requested=self.config.safety_margin_m+.05/parameters[:,10].clamp_min(.01)
            initial=torch.minimum(requested,smooth_clearance(ego,other,self.vehicle).clamp_min(1.))
            start=torch.where(self.margin_reference_valid[ids],
                torch.minimum(requested,self.margin_reference[ids].clamp_min(1.)),initial)
            next_reference=torch.minimum(requested,start+self.vehicle.dt_s*self.config.clearance_recovery_rate_mps)
            if self.config.clearance_recovery_protocol == 'feasible_seed_envelope_v3':
                next_reference=best['required_margin_profile_m'][:,0]
            self.margin_reference[ids]=torch.where(failed,start,next_reference)
            self.margin_reference_valid[ids]=True
        answer={k:v for k,v in best.items() if k!="controls"}
        answer.update(action=action,failed=failed,action_before_fallback=proposed,
            acceptable_count=acceptable_total,soft_choice_replaced=replaced,candidate_source=best_index,
            candidate_coverage_failure=failed,weight_ess=weight_ess,
            shared_first_action_error=(best["controls"][:,0,0]-best["controls"][:,1,0]).abs().amax(-1))
        if self.config.clearance_recovery_protocol in ('observed_gap_envelope_v4', 'observed_gap_objective_v5', 'nominal_gate_objective_v6'):
            requested = self.config.safety_margin_m + .05/parameters[:,10].clamp_min(.01)
            endpoint = torch.minimum(requested, (smooth_clearance(ego,other,self.vehicle)+.1).clamp_min(1.5))
            answer.update(requested_margin_m=requested, required_endpoint_margin_m=endpoint,
                observed_margin_cap_reduction_m=requested-endpoint)
            if self.config.clearance_recovery_protocol == 'nominal_gate_objective_v6':
                answer['nominal_feasibility_margin_m']=torch.full_like(requested,
                    self.config.safety_margin_m + .05/max(self.config.base_parameters[10],.01))
        if self.config.clearance_recovery_protocol in ('persistent_gap_ramp_v2','feasible_seed_envelope_v3'):
            answer.update(requested_margin_m=requested,recovery_reference_m=self.margin_reference[ids].clone(),
                recovery_frozen_on_fallback=failed.clone(),retained_seed_available=self.warm_valid[ids].clone())
        if record_plan:
            actual_controls=best["controls"][:1].clone()
            actual_controls[:,:,0]=action[:1,None]
            if bool(failed[0]):actual_controls[:]=safe_brake_action(ego[:1],self.vehicle)[:,None,None]
            answer.update(self._record_plan(name,ego[:1],other[:1],predictions[:1],actual_controls))
        self.previous_command[ids] = action
        return answer
