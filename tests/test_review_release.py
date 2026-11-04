"""Meaningful numerical and import-boundary checks for the review artifact."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import torch
from review_bmpc import demo
from review_bmpc.model import clearance,smooth_clearance,step_vehicle,VehicleConfig
from review_bmpc.risk import cvar_value
ROOT=Path(__file__).resolve().parents[1]

def test_conservative_rectangle_bound():
    generator=torch.Generator().manual_seed(41)
    x=torch.randn((128,5),generator=generator);z=torch.randn((128,5),generator=generator)
    assert bool((smooth_clearance(x,z)<=clearance(x,z)+1e-6).all())

def test_cvar_confidence_and_small_probability_cap():
    loss=torch.tensor([[1.,10.],[1.,10.]])
    probability=torch.tensor([[.7,.3],[.99,.01]])
    expectation,_=cvar_value(loss,probability,0.)
    tail,weights=cvar_value(loss,probability,.9)
    torch.testing.assert_close(expectation,torch.tensor([3.7,1.09]))
    torch.testing.assert_close(tail,torch.tensor([10.,1.9]))
    assert float(weights[1,1])<=.100001

def test_all_baselines_remain_finite_and_do_not_import_learning():
    import sys
    for method in demo.METHODS:
        trace=demo.run('merge',method,steps=2)
        assert len(trace)==2
        assert np.isfinite(np.array([r['ego'] for r in trace])).all()
    assert not any(name.startswith(('rl_bmpc','review_bmpc.learning','review_bmpc.training')) for name in sys.modules)

def test_frozen_results_recompute(tmp_path):
    spec=importlib.util.spec_from_file_location('review_reproduction',ROOT/'analysis/reproduce.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.controller_comparison(tmp_path);module.ablations(tmp_path);module.solver_tables(tmp_path);module.deployment(tmp_path);module.fixed_selection(tmp_path)
    assert len(json.loads((tmp_path/'table_III.json').read_text()))==32

def test_manifest_hashes():
    import hashlib
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
