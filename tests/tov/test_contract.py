"""Regression checks for the uploaded specification's scientific contract."""
from copy import deepcopy
import json
import numpy as np
from tov_sim.config import ROOT,read_spec,stable_hash,digest
from tov_sim.validate import validate,record
from tov_sim.simulator import Simulator
from tov_sim.metrics import bootstrap_weights,aggregate,ratio
from tov_sim.plan import plan
from tov_sim.workload import routes_for


def test_analytic_numerical_and_resource_contract():
    result=validate(read_spec())
    assert result['passed'] and len(result['checks'])==9


def test_short_decode_requests_exit_without_padding():
    c=read_spec();m=c['models'][1]
    rows=[record(m,rank,index) for rank,length in [(0,2),(1,5)] for index in range(length)]
    for mode in ['layer','replay']:
        result=Simulator(c,m,c['primary_proposal'],mode,'decode',rows,0).run()
        assert [r['tokens'] for r in result['rows']]==[2,2,1,1,1]
        assert result['tokens']==7
        assert sum(r['latency_ps'] for r in result['rows'])==result['total_ps']


def test_all_local_replay_has_zero_communication_without_infinite_goodput():
    c=read_spec();m=deepcopy(c['models'][1]);m['k']=1
    rows=[record(m,0,i,[0]) for i in range(9)]
    for method in c['methods']:
        result=Simulator(c,m,method,'replay','prefill',rows,0).run()
        assert result['tokens']==9 and result['total_ps']==0 and result['useful_bytes']==0
        assert result['packet_hops']==0


def test_bootstrap_cohorts_are_paired_and_ratios_follow_raw_sums():
    c=read_spec();weights=bootstrap_weights(c)
    for (phase,B),w in weights.items():
        assert w.shape==(2000,3*32//B)
        np.testing.assert_array_equal(w.sum(1),np.full(2000,3*32//B))
        a=np.tile([1000,1,2,4,3000,800],(w.shape[1],1))
        b=a.copy();b[:,0]*=2;b[:,2]*=3;b[:,4]*=4
        r=ratio(aggregate(w@a),aggregate(w@b))
        np.testing.assert_allclose(r['ST'],2);np.testing.assert_allclose(r['SG'],4);np.testing.assert_allclose(r['SE_proxy'],3)
    observations=np.array([[100,1,1,1,20,10],[900,3,3,3,80,10]],float)
    assert aggregate(observations.sum(0))['T_ns']==.25


def test_complete_plan_and_justified_reuse():
    rows=plan(read_spec())
    assert sum(r['family']=='main' for r in rows)==1152
    assert sum(r['family']=='sensitivity' for r in rows)==336
    assert sum(not r['reference'] for r in rows)==1296
    for r in rows:
        if r['axis']=='memory_efficiency' and r['measurement']=='replay':assert r['reference']


def test_frozen_trace_matches_document_rng_and_checksums():
    c=read_spec();inp=json.loads((ROOT/'Results/input_manifest.json').read_text());catalog=json.loads((ROOT/'Results/route_manifest.json').read_text())
    for entry in catalog:assert digest(ROOT/'Results/routes'/entry['file'])==entry['sha256']
    for model in c['models']:
        requests=[r for r in inp['requests'] if r['model']==model['id'] and r['seed']==11]
        for phase in ['prefill','decode']:
            for profile in ['uniform','hot_shift']:
                observed=routes_for(c,model,phase,profile,11,requests)
                entry=next(r for r in catalog if r['model']==model['id'] and r['seed']==11 and r['phase']==phase and r['profile']==profile)
                assert stable_hash(observed)==entry['route_hash']
                for row in observed:
                    assert len(set(row['expert_ids']))==model['k']
                    assert abs(sum(row['gate_weights'])-1)<1e-12
