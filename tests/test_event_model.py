import copy
import numpy as np
import pytest
from vdie_moe_sim.simulator import simulate
from vdie_moe_sim.compute import expected_matrix_ops
from vdie_moe_sim.packets import expected_bytes
from vdie_moe_sim.energy import energy

BASELINES=['T_P_N','T_B_N','T_P_L','T_B_L','O_P_N','O_B_N','O_P_L','O_B_L','TW_P_N','TW_B_L']

def descriptors(c,mid,kind,nreq=4,npos=3):
    m=c['models'][mid];rows=[]
    for req in range(nreq):
        for pos in range(npos):
            if kind=='one_die':ids=[(req+1)%16+16*i for i in range(m['top_k'])]
            elif kind=='local':ids=[req%16+16*i for i in range(m['top_k'])]
            else:ids=[(req+pos+i)%m['routed_experts'] for i in range(m['top_k'])]
            if max(ids)>=m['routed_experts']:ids=list(range(m['top_k']))
            rows.append([req,pos,100+req,42,*ids])
    return np.asarray(rows,np.int32)

def audit(c,mid,a,baseline,r):
    assert len(r['tokens'])==len(a)
    assert sum(x['matrix_ops'] for x in r['cohorts'])==expected_matrix_ops(c['models'][mid],len(a))
    for key,value in expected_bytes(c,mid,a,baseline).items():
        assert sum(x[key] for x in r['cohorts'])==value,(key,value)
    for x in r['cohorts']:
        assert sum(v for k,v in x.items() if k.startswith('critical_'))==x['duration_ps']
        assert x['ru_ps']<=x['available_ru_ps']
        assert max(x['peak_partial_bytes'])<=c['hardware']['buffer_partition_bytes'][3]-c['hardware']['partial_owner_reserve_bytes']
        assert max(x['peak_rx_bytes'])<=c['hardware']['buffer_partition_bytes'][1]
        assert max(x['peak_tx_bytes'])<=c['hardware']['buffer_partition_bytes'][2]
        assert max(x['die_compute_ps'])<=x['duration_ps']
        assert max(x['die_dma_ps'])<=x['duration_ps']
        e=energy(x,c)
        assert e['baseband_stress']>=e['frontend_proxy']>0

@pytest.mark.parametrize('mid',['M1','M2','M3'])
@pytest.mark.parametrize('baseline',BASELINES)
@pytest.mark.parametrize('mode',['prefill','decode'])
def test_full_layer_conservation(tiny,mid,baseline,mode):
    a=descriptors(tiny,mid,'spread')
    r=simulate(tiny,mid,a,4,mode,baseline);audit(tiny,mid,a,baseline,r)
    if mode=='decode':
        assert len(r['rounds'])==3
        assert all(row[2]==4 for row in r['rounds'])
        for step in range(1,3):
            current=[t for t in r['tokens'] if t[2]==step]
            previous=[t for t in r['tokens'] if t[2]==step-1]
            assert {t[6] for t in current}=={max(t[7] for t in previous)}

@pytest.mark.parametrize('baseline',BASELINES)
@pytest.mark.parametrize('kind',['local','one_die'])
def test_local_or_same_destination_dedup(tiny,baseline,kind):
    a=descriptors(tiny,'M2',kind,1,1)
    r=simulate(tiny,'M2',a,1,'prefill',baseline);audit(tiny,'M2',a,baseline,r)
    if kind=='local':
        assert r['cohorts'][0]['air_dispatch_bytes']==0
        assert r['cohorts'][0]['air_return_bytes']==0
    else:
        assert r['cohorts'][0]['air_dispatch_bytes']==168
        assert r['cohorts'][0]['air_return_bytes']==(120 if baseline.endswith('_L') else 960)

def test_short_final_cohort_and_decode_early_end(tiny):
    a=descriptors(tiny,'M3','spread',5,3)
    a=a[~((a[:,0]==0)&(a[:,1]>0))]
    r=simulate(tiny,'M3',a,4,'decode','O_B_L');audit(tiny,'M3',a,'O_B_L',r)
    assert [x[2] for x in r['rounds'] if x[0]==0]==[4,3,3]
    assert [x[2] for x in r['rounds'] if x[0]==1]==[1,1,1]

@pytest.mark.parametrize('baseline',['T_P_N','O_B_L','TW_B_L'])
def test_fragmentation_and_tight_buffer_backpressure(tiny,baseline):
    tiny['mac']['mtu_payload_bytes']=64
    hw=tiny['hardware'];hw['buffer_partition_bytes'][1]=4096;hw['buffer_partition_bytes'][2]=4096
    hw['rx_return_reserve_bytes']=2048;hw['tx_return_reserve_bytes']=2048
    a=descriptors(tiny,'M2','one_die',16,20)
    r=simulate(tiny,'M2',a,16,'prefill',baseline);audit(tiny,'M2',a,baseline,r)
    l=r['cohorts'][0]
    assert l['packets']>len(a)*4
    assert l['rx_credit_blocks']+l['tx_credit_blocks']>0

def test_all_receivers_and_physical_half_duplex(tiny):
    a=descriptors(tiny,'M2','spread',1,1)
    r=simulate(tiny,'M2',a,1,'prefill','O_B_L')
    tiny['radio']['multicast_receivers']='all_non_source'
    other=simulate(tiny,'M2',a,1,'prefill','O_B_L');audit(tiny,'M2',a,'O_B_L',other)
    assert other['cohorts'][0]['rx_ps']>r['cohorts'][0]['rx_ps']

def test_deterministic_repetition(tiny):
    a=descriptors(tiny,'M3','spread',8,5)
    assert simulate(tiny,'M3',a,8,'prefill','O_B_L')==simulate(tiny,'M3',a,8,'prefill','O_B_L')

def test_corrupt_trace_rejected(tiny):
    a=descriptors(tiny,'M2','spread',1,1);a[0,5]=a[0,4]
    with pytest.raises(RuntimeError,match='Duplicate'):simulate(tiny,'M2',a,1,'prefill','T_P_N')

@pytest.mark.parametrize('mid',['M1','M2','M3'])
def test_real_dimensions_and_ragged_matrix_edges(config,mid):
    a=descriptors(config,mid,'spread',1,1)
    r=simulate(config,mid,a,1,'prefill','O_B_L');audit(config,mid,a,'O_B_L',r)

@pytest.mark.parametrize('mid',['M1','M2','M3'])
@pytest.mark.parametrize('baseline',['T_P_N','O_B_L'])
@pytest.mark.parametrize('axis,value',[
 ('radio.rate_scale',.25),('radio.rate_scale',.5),('radio.broadcast_raw_bps',124000000000),
 ('hardware.compute_utilization',.5),('hardware.compute_utilization',.75),
 ('memory.bandwidth_scale',.5),('memory.bandwidth_scale',2.),
 ('mac.frame_cycles',10),('mac.frame_cycles',80),('radio.multicast_receivers','all_non_source')])
def test_timing_sensitivity_invariants(tiny,mid,baseline,axis,value):
    from vdie_moe_sim.config import override
    c=override(tiny,{axis:value});a=descriptors(c,mid,'spread',4,3)
    r=simulate(c,mid,a,4,'prefill',baseline);audit(c,mid,a,baseline,r)

@pytest.mark.parametrize('mid',['M1','M2','M3'])
def test_full_128_request_decode_cohort(config,mid):
    a=descriptors(config,mid,'spread',128,2)
    r=simulate(config,mid,a,128,'decode','O_B_L');audit(config,mid,a,'O_B_L',r)
    assert [x[2] for x in r['rounds']]==[128,128]

@pytest.mark.parametrize('baseline',['T_P_N','O_B_L','TW_B_L'])
def test_all_tokens_one_expert_and_other_experts_empty(tiny,baseline):
    tiny['models']['M2']['top_k']=1
    a=np.asarray([[request,position,request+100,42,17]
                  for request in range(32) for position in range(10)],dtype=np.int32)
    r=simulate(tiny,'M2',a,32,'prefill',baseline)
    audit(tiny,'M2',a,baseline,r)
    assert sum(l['batch_tokens'] for l in r['cohorts'])==len(a)
    assert len(r['tokens'])==320

def test_packet_tdma_has_no_frame_tail_even_when_credit_blocked(tiny):
    from vdie_moe_sim.config import override
    tiny['mac']['mtu_payload_bytes']=64
    h=tiny['hardware'];h['buffer_partition_bytes'][1]=4096;h['buffer_partition_bytes'][2]=4096
    h['rx_return_reserve_bytes']=2048;h['tx_return_reserve_bytes']=2048
    a=descriptors(tiny,'M2','one_die',16,20)
    outputs=[simulate(override(tiny,{'mac.frame_cycles':frame}),'M2',a,16,'prefill','TW_B_L')
             for frame in [10,20,80]]
    for result in outputs:audit(tiny,'M2',a,'TW_B_L',result)
    assert outputs[0]['cohorts'][0]['rx_credit_blocks']>0
    assert outputs[0]['tokens']==outputs[1]['tokens']==outputs[2]['tokens']
    assert outputs[0]['cohorts']==outputs[1]['cohorts']==outputs[2]['cohorts']

@pytest.mark.parametrize('access',['T','O','TW'])
def test_local_only_workload_is_independent_of_dispatch_and_remote_combine(tiny,access):
    a=descriptors(tiny,'M2','local',1,5)
    outputs=[simulate(tiny,'M2',a,1,'prefill',f'{access}_{dispatch}_{combine}')
             for dispatch in ['P','B'] for combine in ['N','L']]
    assert all(result==outputs[0] for result in outputs)

@pytest.mark.parametrize('access',['T','O','TW'])
def test_single_remote_destination_broadcast_degenerates_to_unicast(tiny,access):
    a=descriptors(tiny,'M2','one_die',1,5)
    assert simulate(tiny,'M2',a,1,'prefill',access+'_P_N')==simulate(tiny,'M2',a,1,'prefill',access+'_B_N')
