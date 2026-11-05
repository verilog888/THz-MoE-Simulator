import copy
import math
import numpy as np
import pytest
from vdie_moe_sim.config import validate,capacity,vector_bytes
from vdie_moe_sim.compute import expected_matrix_ops,tile_ps
from vdie_moe_sim.memory import burst_ps
from vdie_moe_sim.network import control_ps
from vdie_moe_sim.routing import route,rng_for
from vdie_moe_sim.placement import contributions
from vdie_moe_sim.packets import wire_bytes
from vdie_moe_sim.energy import energy

@pytest.mark.parametrize('mid,expert,router,total',[
 ('M1',24898464,92288,54083520),('M2',4719744,262208,42214464),('M3',11799360,164176,63355280)])
def test_capacity(config,mid,expert,router,total):
    assert capacity(config['models'][mid],config)==dict(expert_storage_bytes=expert,router_storage_bytes=router,resident_bytes_per_die=total)

def test_reject_mismatched_derived_constants(config):
    config['models']['M3']['resident_bytes_per_die']+=1
    with pytest.raises(ValueError,match='resident'):validate(config)

def test_units(config):
    assert config['hardware']['peak_ops_s']/config['hardware']['clock_hz']==5000
    assert tile_ps(1,128,128)==35000
    assert tile_ps(16,128,128)==525000
    assert burst_ps(1024)==10000
    assert burst_ps(1025)==15000
    assert burst_ps(1024,.5)==15000
    assert vector_bytes(2880,config)==2972
    assert wire_bytes(4097,config)==4169
    assert control_ps(config,17)==25000
    assert control_ps(config,16)==20000

@pytest.mark.parametrize('mid',['M1','M2','M3'])
def test_exact_op_convention(config,mid):
    m=config['models'][mid]
    assert expected_matrix_ops(m,1)==2*m['d_model']*m['routed_experts']+6*m['d_model']*m['d_ff']*(m['top_k']+m['shared_experts'])

def test_shared_parity_and_only_two(config):
    for owner in range(16):
        groups=contributions(config['models']['M3'],owner,list(range(6)))
        shared=[(d,e) for d,es in groups.items() for e in es if e>=64]
        assert sorted(e for _,e in shared)==[64,65]
        assert {d for d,_ in shared}=={owner,owner^1}
        assert all(d%2==e-64 for d,e in shared)

@pytest.mark.parametrize('mid',['M1','M2','M3'])
def test_router_determinism_and_reuse(config,mid):
    m=config['models'][mid]
    a=route(20261104,mid,7,'decode',0,m)
    assert a==route(20261104,mid,7,'decode',0,m)
    assert len(set(a[0]))==m['top_k']
    assert math.isclose(sum(a[1]),1,abs_tol=1e-14)
    assert route(20261104,mid,7,'decode',1,m,reuse_probability=1,previous=a)==a
    assert route(20261104,mid,7,'prefill',1,m,reuse_probability=1,previous=a)!=a

def test_uniform_router_has_no_placement_hotspot(config):
    m=config['models']['M2'];counts=np.zeros(128)
    for t in range(2000):
        ids,_=route(20261104,'M2',0,'prefill',t,m)
        counts[ids]+=1
    assert counts.min()>80 and counts.max()<175

def test_refresh_even_for_subperiod_and_energy_closure(config):
    l=dict(duration_ps=100000,matrix_ops=0,vector_ops=0,read_bytes=0,write_bytes=0,
           tx_ps=0,rx_ps=0,control_tx_ps=0,control_rx_ps=0,ofdma_ps=0)
    a=energy(l,config)
    assert math.isclose(a['refresh'],16*64e6*8*.065e-12/11e-6*1e-7,rel_tol=1e-12)
    assert a['refresh']>0
    parts=[v for k,v in a.items() if k not in ['frontend_proxy','baseband_stress']]
    assert math.isclose(sum(parts),a['baseband_stress'],rel_tol=1e-14)
    l['duration_ps']*=2
    assert math.isclose(energy(l,config)['frontend_proxy'],2*a['frontend_proxy'],rel_tol=1e-14)

def test_lo_shared_across_groups_and_original_calibration(config):
    e=config['energy']
    assert math.isclose((2*e['lo_w_per_node']+2*e['tx_increment_w_per_group']+2*e['rx_increment_w_per_group'])/264e9*1e12,.0879/264e9*1e12)
    l=dict(duration_ps=1000000,matrix_ops=0,vector_ops=0,read_bytes=0,write_bytes=0,
           tx_ps=2000000,rx_ps=4000000,control_tx_ps=0,control_rx_ps=0,ofdma_ps=6000000)
    a=energy(l,config)
    assert math.isclose(a['rf_lo'],17*.032*1e-6)
    assert math.isclose(a['ofdma_extra'],e['ofdma_extra_w_per_active_group']*6e-6)

def test_memory_energy_scale_does_not_scale_leak(config):
    l=dict(duration_ps=100000,matrix_ops=5,vector_ops=7,read_bytes=128,write_bytes=32,
           tx_ps=0,rx_ps=0,control_tx_ps=0,control_rx_ps=0,ofdma_ps=0)
    a=energy(l,config);config['memory']['energy_scale']=4;b=energy(l,config)
    for k in ['edram_read','edram_write','refresh']:assert math.isclose(b[k],4*a[k])
    assert b['memory_leak']==a['memory_leak']
