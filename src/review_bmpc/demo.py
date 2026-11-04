"""Run baseline-only CPU examples on the shared response-augmented model.

These examples are not CarSim validation and do not reproduce the withheld RL
policies. Realized SV modes belong to the environment, never the controller API.
"""
from pathlib import Path
from unittest.mock import patch
import argparse,json
import numpy as np
import torch
import osqp
from review_bmpc.model import VehicleConfig
from review_bmpc.controller import ControllerConfig
from review_bmpc.planner_v5 import SequenceBranchMPC
from review_bmpc.ltv_mpc import EffectiveLTVBranchMPC
from review_bmpc import ltv_mpc
from review_bmpc.safe_env import SafeTrafficBatch
from review_bmpc.initial_manifest import apply_manifest
from review_bmpc.comparators.adapters import PublishedController
from review_bmpc.comparators.solvers import SolverSettings
from review_bmpc.comparators.classical import FrenetPIDController,FrenetSettings

ROOT=Path(__file__).resolve().parents[2]
METHODS=['fixed_nominal','fixed_tuned','fot_pid','era','branch','shooting','cem','mppi','ltv']
def read(name):return json.loads((ROOT/'configs'/name).read_text())
def evaluation_config(values,horizon):
    """Use the frozen fixed-controller horizon extension without changing guards."""
    if horizon not in (10,15,20,30):raise ValueError('Supported study horizons: 10,15,20,30.')
    config=ControllerConfig(**{**values,'horizon':10})
    if horizon!=10:object.__setattr__(config,'horizon',horizon)
    return config
class FixedIntervalOSQP(osqp.OSQP):
    def setup(self,**settings):
        settings['adaptive_rho_interval']=25
        return super().setup(**settings)
def build(scenario,method,seed=1,horizon=10):
    if scenario not in ('merge','intersection') or method not in METHODS:raise ValueError('Unknown scenario or method.')
    if method in ('fixed_tuned','fot_pid','era','branch') and horizon!=10:raise ValueError('The selected baseline settings are declared at N10.')
    common=read('common.json');vehicle=VehicleConfig(**common['vehicle']);config=evaluation_config(common['controllers'][scenario],horizon)
    if method in ('era','branch'):
        controller=PublishedController(config,vehicle,settings=SolverSettings(**read(f'{method}_{scenario}.json')),seed=seed,device='cpu')
    elif method=='fot_pid':
        settings=read('classical_selection.json')['selected'][scenario][0]['settings']
        controller=FrenetPIDController(config,vehicle,settings=FrenetSettings(**settings),seed=seed,device='cpu')
    elif method=='ltv':
        controller=EffectiveLTVBranchMPC(config,vehicle,seed=seed,device='cpu');original=controller._qp
        def qp(*args,**kwargs):
            with patch.object(ltv_mpc.osqp,'OSQP',FixedIntervalOSQP):return original(*args,**kwargs)
        controller._qp=qp
    else:
        controller=SequenceBranchMPC(config,vehicle,seed=seed,device='cpu',algorithm=method if method in ('cem','mppi') else 'shooting')
    if method=='fixed_tuned':
        parameters=common['tuned_fixed'][scenario]['constant_parameters'];original=controller.solve
        def solve(name,ego,other,predictions,probabilities,unused_parameters,record_plan=False):
            fixed=ego.new_tensor(parameters).expand(len(ego),-1)
            return original(name,ego,other,predictions,probabilities,fixed,record_plan=record_plan)
        controller.solve=solve
    return controller,common,vehicle
def run(scenario,method,*,index=0,boundary=False,seed=1,horizon=10,steps=5):
    if not 0<=index<128 or not 1<=steps<=180:raise ValueError('index must be 0..127 and steps 1..180.')
    torch.set_num_threads(1)
    controller,common,vehicle=build(scenario,method,seed,horizon)
    env=SafeTrafficBatch(scenario,1,controller,seed=seed,vehicle=vehicle,max_steps=180,
        reward_discount=common['reward_discount'],intersection_shield=False,stress_fraction=0.,
        interaction_version='v6',assert_probability=common['assert_probability'],reward_version=common['reward_version'])
    manifest=read(f"starts/{scenario}_{'stress' if boundary else 'nominal'}.json")
    sample=manifest['starts'][0]
    # apply_manifest expects a complete group; select the requested paired start and rehash it.
    from review_bmpc.initial_manifest import manifest_digest
    manifest['starts']=[{key:[value[index]] for key,value in sample.items()}]
    manifest['sha256']=manifest_digest(manifest);apply_manifest(env,manifest,0)
    trace=[]
    for step in range(steps):
        _,reward,cost,done,info=env.step(None,'fixed')
        trace.append({'step':step,'ego':env.ego[0].tolist(),'other':env.other[0].tolist(),
            'reward':float(reward[0]),'collision_cost':float(cost[0]),'finished':bool(done[0]),
            'running_minimum_m':float(env.running_minimum[0])})
        if bool(env.finished[0]):break
    return trace
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=METHODS,default='fixed_nominal');p.add_argument('--scenario',choices=['merge','intersection'],default='merge');p.add_argument('--index',type=int,default=0);p.add_argument('--boundary',action='store_true');p.add_argument('--seed',type=int,default=1);p.add_argument('--horizon',type=int,default=10);p.add_argument('--steps',type=int,default=5);p.add_argument('--output',type=Path);a=p.parse_args()
    result={'backend':'shared CPU model (not CarSim)','method':a.method,'scenario':a.scenario,'index':a.index,'boundary':a.boundary,'seed':a.seed,'horizon':a.horizon,'trace':run(a.scenario,a.method,index=a.index,boundary=a.boundary,seed=a.seed,horizon=a.horizon,steps=a.steps)}
    text=json.dumps(result,indent=2)
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text+'\n')
    else:print(text)
if __name__=='__main__':main()
