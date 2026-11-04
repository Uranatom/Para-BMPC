"""Werling-style Frenet polynomial planning with decoupled PID tracking.

Polynomial evaluators and the reference spline are executed from the pinned
PythonRobotics source. Its circular/static collision check is replaced by the
benchmark's oriented vehicle geometry and observable reactive hypotheses.
This is a declared common-plant adaptation, not the authors' original vehicle.
"""
from dataclasses import dataclass, asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from review_bmpc.planner_v5 import SequenceBranchMPC
from review_bmpc.model import step_vehicle, clip_action, safe_brake_action, clearance, road_clearance
from review_bmpc.scenarios import other_action

ROOT=Path(__file__).resolve().parents[3]


def reference_module():
    """Verify all imported upstream dependencies before executing their source."""
    lock=json.loads((ROOT/'configs/pythonrobotics.lock.json').read_text(encoding='utf-8'))
    base=ROOT/'third_party/PythonRobotics'
    for name,digest in lock['source_sha256'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Pinned Frenet reference changed: '+name)
    name='_rl_bmpc_upstream_frenet'
    if name not in sys.modules:
        sys.path.insert(0,str(base));sys.path.insert(0,str(base/'PathPlanning'))
        spec=importlib.util.spec_from_file_location(name,base/'PathPlanning/FrenetOptimalTrajectory/frenet_optimal_trajectory.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]


@dataclass(frozen=True)
class FrenetSettings:
    target_speed:float=8.
    jerk_weight:float=.1
    time_weight:float=.1
    lateral_weight:float=1.
    speed_weight:float=1.
    clearance_margin:float=1.
    speed_kp:float=1.
    speed_ki:float=.05
    speed_kd:float=.05
    lateral_kp:float=.03
    lateral_ki:float=.0005
    heading_kp:float=.5
    steering_gain:float=6.
    times:tuple=(3.,4.,5.)
    speed_samples:tuple=(0.,2.,4.,6.,8.,9.)


class FrenetPIDController(SequenceBranchMPC):
    """One nonbranching Frenet plan, sampled polynomials, then PID feedback."""
    def __init__(self,config,vehicle,*,settings=None,seed=1,device='cpu'):
        if str(device)!='cpu':raise ValueError('The classic reference runs on CPU.')
        super().__init__(config,vehicle,seed=seed,device=device,algorithm='frenet_pid_r1')
        self.settings=settings or FrenetSettings();self.native=reference_module();self.routes={};self.last_solver_records=[]
        self.integrals={};self.previous_errors={}

    def state_dict(self):
        return {**super().state_dict(),'frenet_settings':asdict(self.settings),
                'frenet_integrals':self.integrals,'frenet_previous_errors':self.previous_errors}

    def load_state_dict(self,state):
        if state.get('frenet_settings')!=asdict(self.settings):raise ValueError('Frenet semantics changed.')
        super().load_state_dict(state);self.integrals=state['frenet_integrals'];self.previous_errors=state['frenet_previous_errors']

    def route(self,name):
        if name not in self.routes:
            x=np.r_[-100.,-50.,0.,np.arange(8.,49.,5.),80.,150.]
            s=np.clip((x-8.)/40.,0,1);y=3.5*(10*s**3-15*s**4+6*s**5) if name=='merge' else np.zeros_like(x)
            self.routes[name]=self.native.cubic_spline_planner.CubicSpline2D(x,y)
        return self.routes[name]

    def candidates(self,name,ego,acceleration,stopping_allowed=True):
        """Retain quartic speed keeping, quintic stopping/lateral and squared jerk."""
        s=self.settings;v=self.vehicle;csp=self.route(name)
        if not np.isfinite(np.r_[ego,acceleration]).all() or max(0.,ego[0]+90.)>=min(csp.s[-1]-.01,ego[0]+110.):return []
        projection=minimize_scalar(lambda q:sum((np.array(csp.calc_position(q))-ego[:2])**2),
            bounds=(max(0.,ego[0]+90.),min(csp.s[-1]-.01,ego[0]+110.)),method='bounded')
        arc=float(projection.x);rx,ry=csp.calc_position(arc);yaw=csp.calc_yaw(arc)
        longitudinal_state,lateral_state=self.native.CartesianFrenetConverter.cartesian_to_frenet(
            arc,rx,ry,yaw,csp.calc_curvature(arc),csp.calc_curvature_rate(arc),
            ego[0],ego[1],max(0.,ego[3]),acceleration,ego[2],np.tan(ego[4])/v.wheelbase_m)
        offset=lateral_state[0];long_speed=max(0.,longitudinal_state[1]);acceleration=longitudinal_state[2]
        strategy=(self.native.HighSpeedLateralMovementStrategy() if long_speed>=2.
            else self.native.LowSpeedLateralMovementStrategy())
        result=[]
        for duration in s.times:
            t=np.arange(0.,duration+v.dt_s/2,v.dt_s)
            longitudinal=[('velocity',self.native.QuarticPolynomial(arc,long_speed,acceleration,tv,0.,duration))
                for tv in sorted(set((*s.speed_samples,s.target_speed)))]
            if name=='intersection' and stopping_allowed:
                stop_x=-1.75-.5-v.lr_m-v.body_length_m/2
                stop_arc=arc+max(0.,stop_x-ego[0])
                longitudinal.append(('stopping',self.native.QuinticPolynomial(arc,long_speed,acceleration,stop_arc,0.,0.,duration)))
            for kind,lon in longitudinal:
                positions=lon.calc_point(t);vel=lon.calc_first_derivative(t);acc=lon.calc_second_derivative(t)
                if np.min(vel)<-.01 or np.max(vel)>v.max_speed_mps or np.max(acc)>v.max_accel_mps2 or np.min(acc)<-v.max_brake_mps2:continue
                if long_speed<2. and positions[-1]-positions[0]<1e-4:continue
                # The native spline has no continuation outside its domain.
                # Reject the whole path before any endpoint yaw/curvature query.
                if not np.isfinite(np.r_[positions,vel,acc]).all() or np.any(positions<csp.s[0]) or np.any(positions>=csp.s[-1]):continue
                fp=self.native.FrenetPath();fp.t=t.tolist();fp.s=positions.tolist();fp.s_d=vel.tolist();fp.s_dd=acc.tolist();fp.s_ddd=lon.calc_third_derivative(t).tolist()
                if name=='merge':
                    end_x,end_y=csp.calc_position(float(positions[-1]));end_yaw=csp.calc_yaw(float(positions[-1]))
                    keep_lane=(ego[1]-end_y)/max(.5,np.cos(end_yaw))
                    lateral_samples=(-3.5,-2.5,-1.5,-.5,0.,.5,float(keep_lane),float(np.clip(offset,-3.5,3.5)))
                else:lateral_samples=(-.2,0.,.2,float(np.clip(offset,-.5,.5)))
                for dest in sorted(set(lateral_samples)):
                    candidate=strategy.calc_lateral_trajectory(fp,dest,*lateral_state,duration)
                    candidate=strategy.calc_cartesian_parameters(candidate,csp)
                    if len(candidate.x)!=len(t):continue
                    xy=np.column_stack((candidate.x,candidate.y));path_heading=np.array(candidate.yaw);curvature=np.array(candidate.c)
                    if not np.isfinite(np.r_[xy.ravel(),curvature,candidate.v,candidate.a]).all():continue
                    if np.max(np.abs(curvature))>np.tan(v.max_steer_rad)/v.wheelbase_m:continue
                    if np.max(candidate.v)>v.max_speed_mps or np.min(candidate.v)<-.01 or np.max(candidate.a)>v.max_accel_mps2 or np.min(candidate.a)<-v.max_brake_mps2:continue
                    # This is the upstream sampled squared-jerk cost, without a
                    # path-integral reinterpretation or controller reward terms.
                    destination=(s.target_speed-vel[-1])**2 if kind=='velocity' else (positions[-1]-stop_arc)**2
                    cost=(s.jerk_weight*(np.square(candidate.d_ddd).sum()+np.square(candidate.s_ddd).sum())
                        +2*s.time_weight*duration+s.lateral_weight*candidate.d[-1]**2+s.speed_weight*destination)
                    result.append({'xy':xy,'yaw':path_heading,'v':np.array(candidate.v),'a':np.array(candidate.a),'curvature':curvature,
                        'cost':float(cost),'kind':kind,'duration':duration,'lateral_domain':'time' if long_speed>=2. else 'arc_length'})
        return result

    @torch.no_grad()
    def solve(self,name,ego,other,predictions,probabilities,parameters,record_plan=False):
        began=time.perf_counter()
        if self.warm is None:self.reset_context(len(ego))
        if len(ego)!=1:raise ValueError('FOT-PID qualification uses independent single-vehicle processes.')
        ids=self.ids if self.ids is not None else torch.arange(len(ego));idx=int(ids[0]);s=self.settings;v=self.vehicle
        from review_bmpc.planner_v5 import release_allowed
        stopping_allowed=name=='intersection' and not bool(release_allowed(ego,other,v,'v7')[0]) and float(ego[0,0])<-6.
        paths=self.candidates(name,ego[0].numpy(),float(self.previous_acceleration[idx]),stopping_allowed);native_count=len(paths)
        valid=[];commands=[];integral0=np.array(self.integrals.get(idx,[0.,0.]));error0=np.array(self.previous_errors.get(idx,[0.,0.]))
        # Vectorize independent candidate screening; each retains two reactive
        # hypotheses. Padding is masked, so shorter paths gain no extra checks.
        eligible=[]
        if paths:
            lengths=torch.tensor([len(p['v']) for p in paths]);maximum=int(lengths.max())
            states=torch.tensor(np.array([np.pad(np.column_stack((p['xy'],p['yaw'],p['v'],np.zeros(len(p['v'])))),
                ((0,maximum-len(p['v'])),(0,0)),mode='edge') for p in paths]),dtype=ego.dtype)
            z=other[0].expand(len(paths),2,5).clone();ok=torch.ones(len(paths),dtype=torch.bool)
            branches=torch.arange(2)[None]
            for k in range(1,maximum):
                z=step_vehicle(z,other_action(name,z,branches,v,ego=states[:,k-1,None].expand(-1,2,-1),interaction_version='v6'),v)
                safe=(clearance(states[:,k,None].expand(-1,2,-1),z,v).amin(-1)>=s.clearance_margin)&(road_clearance(states[:,k],name,v,smooth=True)>=0)
                ok &= (k>=lengths)|safe
            eligible=[path for path,keep in zip(paths,ok.tolist()) if keep]
        for path in eligible:
            x=ego[0].clone();previous=self.previous_command[idx].clone();integral=integral0.copy();old_error=error0.copy();controls=[]
            for k in range(self.config.horizon):
                j=min(k+1,len(path['v'])-1);yaw=float(x[2]);refyaw=path['yaw'][j]
                lateral=-(float(x[0])-path['xy'][j,0])*np.sin(refyaw)+(float(x[1])-path['xy'][j,1])*np.cos(refyaw)
                error=np.array([path['v'][j]-float(x[3]),-lateral]);integral=np.clip(integral+error*v.dt_s,[-5.,-1.],[5.,1.]);derivative=(error-old_error)/v.dt_s
                acceleration=path['a'][j]+s.speed_kp*error[0]+s.speed_ki*integral[0]+s.speed_kd*derivative[0]
                angle=np.arctan2(np.sin(refyaw-yaw),np.cos(refyaw-yaw))
                steer=np.clip(np.arctan(v.wheelbase_m*path['curvature'][j])+s.lateral_kp*error[1]+s.lateral_ki*integral[1]+s.heading_kp*angle,-v.max_steer_rad,v.max_steer_rad)
                u=clip_action(x.new_tensor([acceleration,s.steering_gain*(steer-float(x[4]))]),v)
                u[0]=previous[0]+(u[0]-previous[0]).clamp(-.8,.8);controls.append(u);previous=u;x=step_vehicle(x,u,v);old_error=error
                if k==0:path['next_integral']=integral.copy();path['next_error']=error.copy()
            commands.append(torch.stack(controls)[None].expand(2,-1,-1));valid.append(path)
        if commands:
            raw=torch.stack(commands)[None];audit=self._rollout(name,ego,other,predictions,probabilities,parameters,controls=raw)
            acceptable=torch.isfinite(audit['objective'][0])&(audit['max_violation'][0]<=self.config.fallback_violation_m)
            choices=[i for i in range(len(valid)) if acceptable[i]]
        else:choices=[]
        failed=not choices
        if failed:
            raw=safe_brake_action(ego,v)[:,None,None,None].expand(1,1,2,self.config.horizon,2).clone()
            audit=self._rollout(name,ego,other,predictions,probabilities,parameters,controls=raw);choice=0
        else:
            choice=min(choices,key=lambda i:(valid[i]['cost'],-i));path=valid[choice]
            self.integrals[idx]=path['next_integral'].tolist();self.previous_errors[idx]=path['next_error'].tolist()
        answer={key:value[:,choice] for key,value in audit.items() if key not in ('controls','states','others','references')}
        controls=audit['controls'][:,choice];action=safe_brake_action(ego,v) if failed else controls[:,0,0]
        self.warm[ids]=controls;self.warm_valid[ids]=not failed
        answer.update(action=action,failed=torch.tensor([failed]),solver_failed=torch.tensor([failed]),
            action_before_fallback=controls[:,0,0],controls=controls,raw_controls=raw[:,choice],q=probabilities,
            shared_first_action_error=(controls[:,0,0]-controls[:,1,0]).abs().amax(-1))
        self.last_solver_records=[{'method':'fot_pid','finite':bool(torch.isfinite(action).all()),'native_candidates':native_count,
            'q_role':'Observed hypothesis weights only; no CVaR optimizer or risk forecast.',
            'full_path_feasible_candidates':len(valid),'common_audit_accepted_candidates':len(choices),'fallback':failed,
            'selected_polynomial_cost':None if failed else path['cost'],'selected_mode':None if failed else path['kind'],
            'post_audit_violation_m':float(answer['max_violation'][0]),'convergence_certificate':None,
            'history':[],'wall_seconds':time.perf_counter()-began,'settings':asdict(s)}]
        if record_plan:answer.update(self._record_plan(name,ego,other,predictions,controls))
        self.last_command_time=time.perf_counter();return answer
