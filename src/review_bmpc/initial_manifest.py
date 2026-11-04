"""Explicit physical starts for paired CPU/CUDA and independent solver tests."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .model import VehicleConfig, clearance


def manifest_digest(manifest):
    content={k:v for k,v in manifest.items() if k!='sha256'}
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def export_manifest(environments):
    first=environments[0]
    if any(e.name!=first.name or e.vehicle!=first.vehicle for e in environments):
        raise ValueError('A manifest must use one scenario and vehicle configuration.')
    manifest={'schema':'physical_starts_v1','scenario':first.name,
        'vehicle':first.vehicle.to_dict(),'interaction_version':first.interaction_version,
        'assert_probability':first.assert_probability,'starts':[]}
    for env in environments:
        manifest['starts'].append({name:getattr(env,name).cpu().tolist() for name in
            ('ego_initial','other_initial','probabilities','mode','stress_mask')})
    manifest['sha256']=manifest_digest(manifest)
    return manifest


def apply_manifest(environment,manifest,group):
    if manifest.get('schema')!='physical_starts_v1' or manifest.get('sha256')!=manifest_digest(manifest):
        raise ValueError('Invalid initial-state manifest or digest.')
    if (manifest['scenario']!=environment.name or VehicleConfig(**manifest['vehicle']).to_dict()!=environment.vehicle.to_dict()
            or manifest['interaction_version']!=environment.interaction_version
            or manifest['assert_probability']!=environment.assert_probability):
        raise ValueError('Initial-state manifest scenario, vehicle or interaction semantics changed.')
    values=manifest['starts'][group]
    for name,source in (('ego','ego_initial'),('other','other_initial'),('probabilities','probabilities'),
                        ('mode','mode'),('stress_mask','stress_mask')):
        tensor=torch.as_tensor(values[source],device=environment.device,dtype=getattr(environment,name).dtype)
        if tensor.shape!=getattr(environment,name).shape or not torch.isfinite(tensor).all():
            raise ValueError('Manifest state dimensions or values are invalid.')
        setattr(environment,name,tensor.clone())
    if not torch.allclose(environment.probabilities.sum(-1),torch.ones(environment.count,device=environment.device)):
        raise ValueError('Manifest branch probabilities must sum to one.')
    if bool(((environment.probabilities < 0) | (environment.probabilities > 1)).any()):
        raise ValueError('Manifest probabilities must lie in [0,1].')
    if bool(((environment.mode != 0) & (environment.mode != 1)).any()):
        raise ValueError('Manifest modes must identify the two declared hypotheses.')
    if bool((clearance(environment.ego,environment.other,environment.vehicle)<=0).any()):
        raise ValueError('Manifest contains initial geometric contact or overlap.')
    environment.ego_initial=environment.ego.clone();environment.other_initial=environment.other.clone()
    environment.previous_clearance=clearance(environment.ego,environment.other,environment.vehicle)
    environment.running_minimum=environment.previous_clearance.clone()
    environment.last_q=environment.probabilities.clone()
    environment.initial_feasibility_margin_m=environment._stress_feasibility_margin()
    if bool((environment.stress_mask & (environment.initial_feasibility_margin_m<0)).any()):
        raise ValueError('Manifest contains physically unscreened stress starts.')
    return environment.observation()
