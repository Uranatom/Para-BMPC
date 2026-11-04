"""Published-algorithm adapters with the unchanged R7 nonlinear plan audit."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np
import torch

from ..planner_v5 import SequenceBranchMPC, feedback
from ..model import step_vehicle, safe_brake_action
from ..scenarios import other_action
from .common_model import CommonModel
from .solvers import TreeILQR, BranchSOCP, SolverSettings, risk_vertices


class PublishedController(SequenceBranchMPC):
    """Use paper-derived optimization, retaining only the R7 audit/cache shell."""
    def __init__(self,config,vehicle,*,settings:SolverSettings,seed=1,device='cpu'):
        if str(device)!='cpu':raise ValueError('Published adapters currently declare single-process CPU solving only.')
        super().__init__(config,vehicle,seed=seed,device=device,algorithm=f'sota_r1_{settings.method}')
        self.settings=settings;self.models={};self.solvers={};self.last_solver_records=[]

    def state_dict(self):
        result=super().state_dict()
        result['sota_settings']=asdict(self.settings)
        result['sota_vehicle']=self.vehicle.to_dict()
        result['sota_controller_config']=asdict(self.config)
        return result

    def load_state_dict(self,state):
        if (state.get('sota_settings')!=asdict(self.settings) or state.get('sota_vehicle')!=self.vehicle.to_dict()
            or state.get('sota_controller_config')!=asdict(self.config)):
            raise ValueError('Published solver, vehicle or controller semantics changed on evaluation resume.')
        super().load_state_dict(state)

    def model(self,name):
        if name not in self.models:
            self.models[name]=CommonModel(name,self.vehicle,self.config,self.settings.weight_scale,self.settings)
            kind=TreeILQR if self.settings.method=='era' else BranchSOCP
            self.solvers[name]=kind(self.models[name],self.settings)
        return self.models[name],self.solvers[name]

    @torch.no_grad()
    def initial_controls(self,name,ego,other,probabilities,ids):
        # The published Era method uses a sampled behavior initializer. Adapt
        # that mechanism to the common plant, rather than relying on one guess.
        count=len(ego);N=self.config.horizon;seeds=6
        x=ego[:,None,None].expand(count,seeds,2,5).clone()
        z=other[:,None,None].expand_as(x).clone()
        previous=self.previous_command[ids][:,None,None].expand(count,seeds,2,2).clone()
        branches=torch.arange(2).view(1,1,2)
        target=ego.new_tensor([0.,0.,2.,4.,6.,8.]).view(1,seeds,1)
        plans=[]
        for k in range(N):
            u,refs=feedback(name,x,z,branches,self.vehicle,True,'v7')
            desired=torch.minimum(target,refs[2])
            acceleration=(desired-x[...,3]).clamp(-4.,2.)
            acceleration[:,0]=u[:,0,:,0]
            u[...,0]=previous[...,0]+(acceleration-previous[...,0]).clamp(-.8,.8)
            if k<self.settings.shared_prefix:
                shared=(u*probabilities[:,None,:,None]).sum(2)
                u=shared[:,:,None].expand(count,seeds,2,2).clone()
            plans.append(u)
            z=step_vehicle(z,other_action(name,z,branches,self.vehicle,ego=x,interaction_version='v6'),self.vehicle)
            x=step_vehicle(x,u,self.vehicle);previous=u
        controls=torch.stack(plans,dim=3)
        warm=torch.cat((self.warm[ids,:,1:],self.warm[ids,:,-1:]),dim=2).clone()
        common=(warm[:,:,:self.settings.shared_prefix]*probabilities[:,:,None,None]).sum(1)
        warm[:,:,:self.settings.shared_prefix]=common[:,None]
        controls=torch.cat((controls,warm[:,None]),dim=1)
        parameters=torch.as_tensor(self.config.base_parameters,dtype=ego.dtype).expand(count,-1)
        prediction=other[:,None,None].expand(count,2,N,5)
        audit=self._rollout(name,ego,other,prediction,probabilities,parameters,controls=controls)
        value=audit['objective'].clone();violation=audit['max_violation'].clone()
        value[:,-1]=torch.where(self.warm_valid[ids],value[:,-1],float('inf'))
        violation[:,-1]=torch.where(self.warm_valid[ids],violation[:,-1],float('inf'))
        from ..planner_v5 import select_candidate
        index,_=select_candidate(value,violation,self.config.fallback_violation_m,True)
        self.initializer_indices=index.tolist()
        return audit['controls'][torch.arange(count),index].double().numpy()

    @torch.no_grad()
    def solve(self,name,ego,other,predictions,probabilities,parameters,record_plan=False):
        decision_started=time.perf_counter()
        if self.warm is None:self.reset_context(len(ego))
        ids=self.ids if self.ids is not None and len(self.ids)==len(ego) else torch.arange(len(ego))
        model,solver=self.model(name)
        model_ready=time.perf_counter()
        initial_controls=self.initial_controls(name,ego,other,probabilities,ids)
        initializer_ready=time.perf_counter()
        proposals=[];records=[]
        for index in range(len(ego)):
            initial=model.initial(ego[index].numpy(),other[index].numpy(),
                self.previous_command[ids[index]].numpy(),float(self.previous_acceleration[ids[index]]))
            p=probabilities[index].double().numpy();p=p/p.sum()
            if self.settings.probability_model=='native_safety':
                # Original source's softsat/softmax design on the adapted plant.
                x=initial.copy();minimum=np.full(2,np.inf)
                for k in range(self.config.horizon):
                    x=np.asarray(model.f(x,initial_controls[index,0,k])).ravel()
                    from .common_model import smooth_clearance
                    for b in range(2):minimum[b]=min(minimum[b],float(smooth_clearance(x[:5],x[5+5*b:10+5*b],self.vehicle)))
                safety=1/(1+np.exp(-np.clip(minimum,-50,50)))
                p=np.exp(2*safety);p/=p.sum()
            started=time.perf_counter()
            try:
                controls,diagnostic=solver.solve(initial,initial_controls[index],p)
            except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
                controls=initial_controls[index]
                diagnostic={'solver_exception':repr(exc),'finite':False,'subproblem_accepted':False}
            diagnostic.update(wall_seconds=time.perf_counter()-started,planning_p=p.tolist(),
                method=self.settings.method,shared_prefix=self.settings.shared_prefix,
                initializer_index=self.initializer_indices[index],initializer_protocol='native_style_speed_bank_v1')
            proposals.append(controls);records.append(diagnostic)
        raw=torch.tensor(np.array(proposals),dtype=ego.dtype)
        optimizer_ready=time.perf_counter()
        # Evaluate exactly the controls proposed by this solver, never substitute
        # shooting candidates or average complete branch plans after the audit.
        audit=self._rollout(name,ego,other,predictions,probabilities,parameters,
            controls=raw[:,None],record=record_plan)
        answer={key:value[:,0] for key,value in audit.items() if key not in ('controls','states','others','references')}
        controls=audit['controls'][:,0]
        solver_failed=torch.tensor([not r.get('finite',False) or
            (self.settings.method=='branch' and not r.get('subproblem_accepted',False)) for r in records])
        failed=(~torch.isfinite(answer['objective']))|(answer['max_violation']>self.config.fallback_violation_m)|solver_failed
        proposed=controls[:,0,0].clone()
        action=torch.where(failed[:,None],safe_brake_action(ego,self.vehicle),proposed)
        self.warm[ids]=torch.where(failed[:,None,None,None],torch.zeros_like(controls),controls)
        self.warm_valid[ids]=~failed
        answer.update(action=action,failed=failed,action_before_fallback=proposed,
            raw_controls=raw,controls=controls,solver_failed=solver_failed,
            shared_first_action_error=(controls[:,0,0]-controls[:,1,0]).abs().amax(-1))
        for index,record in enumerate(records):
            record.update(post_audit_violation_m=float(answer['max_violation'][index]),
                fallback=bool(failed[index]),raw_first_action=raw[index,0,0].tolist(),
                actual_first_action=action[index].tolist(),audit_q=answer['q'][index].tolist(),
                control_audit_change=float((raw[index]-controls[index]).abs().max()))
        self.last_solver_records=records
        if record_plan:
            executed=controls[:1].clone();executed[:,:,0]=action[0]
            if failed[0]:executed[:]=action[0]
            answer.update(self._record_plan(name,ego[:1],other[:1],predictions[:1],executed))
        self.last_command_time=time.perf_counter()
        for record in records:
            record['stage_seconds']={'model_construction':model_ready-decision_started,
                'initializer':initializer_ready-model_ready,'optimizer_batch':optimizer_ready-initializer_ready,
                'audit_fallback_command':self.last_command_time-optimizer_ready,
                'adapter_total':self.last_command_time-decision_started}
        return answer
