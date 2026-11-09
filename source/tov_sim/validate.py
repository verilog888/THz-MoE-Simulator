"""Analytic, causal, numerical, negative-control, and reproducibility checks."""
from copy import deepcopy
import math
import numpy as np
from .config import BASE,derive,capacity,expert_bytes,router_bytes,qbytes,stable_hash
from .events import Intervals
from .kernel import Kernel
from .simulator import Simulator
from .numerics import LEVELS,e4m3fn,quantize_vector,quantize_weight,activation,error


def record(m,rank=0,index=0,experts=None):
    return dict(request_rank=rank,token_index=index,expert_ids=experts or list(range(m['k'])),shared_ids=list(range(m['shared'])),request_id=rank)


def validate(c):
    checks=[]
    def check(name,fn):
        try:details=fn();checks.append(dict(name=name,passed=True,details=details))
        except Exception as exc:checks.append(dict(name=name,passed=False,error=f'{type(exc).__name__}: {exc}'));raise
    def units():
        d=derive(c)
        for key,value in [('peak_ops_per_second',8.192e12),('hb_bytes_per_second',819.2e9),('control_base_ps',21000),('control_peer_ps',20000)]:assert d[key]==value,(key,d[key])
        assert d['kernel_bytes_per_second']+d['dma_bytes_per_second']<=d['hb_bytes_per_second']
        assert d['tsv_norm_port_gbps']*16==264 and d['tsv_high_port_gbps']*16==7168
        return d
    check('units_and_fixed_HB_partition',units)
    def storage():
        rows=[]
        for n in c['sensitivity']['die_counts']:
            cc=deepcopy(c);cc['hardware']['dies']=n
            for m in c['models']:
                r=capacity(cc,m);assert all(a['passed'] for a in r);rows.extend(r)
                assert expert_bytes(c,m)>derive(c)['sram_bytes_per_die']
        return rows
    check('capacity_scales_bias_shared_SRAM',storage)
    def kernels():
        rows=[]
        for m in c['models']:
            kernel=Kernel(c,m)
            for batch in [1,4,8]:
                duration,mac,hb,scalar=kernel.matrix('expert',batch)
                assert mac==3*m['d']*m['h']*batch
                assert hb==expert_bytes(c,m)+batch*(qbytes(c,m['d'])+4*m['d'])
                assert duration>0 and scalar>0
                rows.append(dict(model=m['id'],batch=batch,MAC=mac,HB_bytes=hb,time_ps=duration))
        return rows
    check('batch_weight_read_once_and_MAC',kernels)
    def serialization():
        exact=4096*8/132;assert abs(exact-248.24242424242425)<1e-12
        m=c['models'][1];sim=Simulator(c,m,'TSV_NORM_MC_LC','replay','prefill',[],0);received=[];size=2048
        sim.network.send(0,(1,),'DISPATCH',size,lambda node:received.append((node,sim.ev.now)))
        sim.ev.run(lambda:bool(received));rate=264/16*.9/1000
        expected=2*math.ceil(size*8/rate)+25000+2*math.ceil(size/102.4e9*1e12)+2*math.ceil(size/1024e9*1e12)
        assert sim.ev.now==expected,(sim.ev.now,expected)
        assert not any(sim.buffers)
        # Independent Base ports overlap, despite one shared serial Base DMA.
        par=Simulator(c,m,'TSV_NORM_MC_LC','replay','prefill',[],0);got=[]
        for dst in [0,1]:par.network.send(BASE,(dst,),'INPUT',size,lambda node:got.append(par.ev.now))
        par.ev.run(lambda:len(got)==2)
        assert max(got)-min(got)==math.ceil(size/1024e9*1e12)
        return dict(virtual_4096B_one_hop_ns=exact,virtual_two_hops_ns=2*exact,integer_one_hop_ps=math.ceil(exact*1000),actual_two_hop_ps=sim.ev.now,parallel_port_completion_ps=got)
    check('analytic_serialization_store_forward_parallel_ports',serialization)
    def flow_checks():
        m=c['models'][1];rows=[record(m,r,i) for r in range(3) for i in range(3)];out=[]
        for mode in ['layer','replay']:
            useful=[]
            for method in c['methods']:
                result=Simulator(c,m,method,mode,'prefill',rows,0).run()
                assert result['tokens']==9 and result['contributor_checks']==9
                assert result['max_owner_active']<=4 and result['max_ru']<=16
                useful.append(result['useful_bytes'])
                out.append(dict(mode=mode,method=method,time_ps=result['total_ps'],physical_tx_bits=result['ledger']['physical_tx_bits'],peak_buffer_bytes=result['peak_buffer_bytes']))
            assert len(set(useful))==1
        # A broadcast transmits once but writes every destination staging buffer.
        sim=Simulator(c,m,'TOV_OFDMA_BCAST_LC','replay','prefill',[],0);got=[]
        sim.network.send(0,(1,2,3),'DISPATCH',100,lambda node:got.append(node));sim.ev.run(lambda:len(got)==3)
        assert sim.ledger.count['domain_1_data_bits']==800 and sim.ledger.count['data_rx_bits']==2400
        assert sim.ledger.count['DMA_bytes']==400
        mc=Simulator(c,m,'TSV_NORM_MC_LC','replay','prefill',[],0);got=[]
        mc.network.send(0,(1,2,3),'DISPATCH',120,lambda node:got.append(node),{1:100,2:100,3:100})
        mc.ev.run(lambda:len(got)==3)
        assert mc.ledger.count['tsv_bits']==(120+300)*8 and mc.ledger.count['DMA_bytes']==2*(120+300)
        return out
    check('resource_causality_contributors_multicast_conservation',flow_checks)
    def negative():
        m=deepcopy(c['models'][1]);m.update(k=1);rows=[record(m,experts=[1])]
        outcomes={method:Simulator(c,m,method,'replay','prefill',rows,0).run() for method in c['methods']}
        for a,b in [('TSV_NORM_P2P_LC','TSV_NORM_MC_LC'),('THZ_TDMA_P2P_LC','THZ_TDMA_BCAST_LC'),('THZ_OFDMA_P2P_LC','TOV_OFDMA_BCAST_LC'),('THZ_OFDMA_BCAST_NO_LC','TOV_OFDMA_BCAST_LC')]:
            assert outcomes[a]['ledger']['physical_tx_bits']==outcomes[b]['ledger']['physical_tx_bits']
            assert outcomes[a]['total_ps']==outcomes[b]['total_ps']
        fast=deepcopy(c);fast['tsv'].update(normalized_total_gbps=1e15,endpoint_pipeline_ns=0,turnaround_ns=0,base_route_ns=0)
        fast['hardware'].update(hb_pin_gbps=4e8,base_dma_GBps=1e12)
        slow=Simulator(fast,m,c['primary_baseline'],'layer','prefill',rows,0).run()
        assert slow['total_ps']>slow['ledger']['nmp_busy_ps']/fast['hardware']['dies']>0
        default=Simulator(c,m,c['primary_baseline'],'layer','prefill',rows,0).run()
        compute=deepcopy(c);compute['hardware']['clock_hz']*=100;compute['hardware']['hb_pin_gbps']*=100
        quick=Simulator(compute,m,c['primary_baseline'],'layer','prefill',rows,0).run()
        assert quick['total_ps']<default['total_ps']
        assert quick['ledger']['nmp_busy_ps']/quick['total_ps']<default['ledger']['nmp_busy_ps']/default['total_ps']
        return dict(single_destination_zero_savings=True,fast_network_nonzero_layer_ps=slow['total_ps'],original_ps=default['total_ps'],faster_compute_ps=quick['total_ps'],nmp_busy_fraction_before=default['ledger']['nmp_busy_ps']/default['total_ps'],nmp_busy_fraction_after=quick['ledger']['nmp_busy_ps']/quick['total_ps'])
    check('negative_controls_and_bottleneck_migration',negative)
    def intervals():
        iv=Intervals();iv.add(0,110,125);iv.add(0,100,105);iv.add(0,103,111)
        assert iv.through(115)==15 and iv.through(200)==10
        iv.add(0,0,20);iv.add(0,5,15);iv.add(1,0,20);assert iv.through(200)==40
        return dict(overlapping_RX_RUs_counted_once_per_domain=True,future_pipeline_tail_clipping=True)
    check('energy_activity_interval_union',intervals)
    def math_check():
        rng=np.random.Generator(np.random.PCG64(2026));x=rng.normal(0,.2,(16,32)).astype('float32')
        weights=rng.normal(0,.2,(6,3,32,16)).astype('float32');bias=rng.normal(0,.1,(6,3,32)).astype('float32')
        reports=[]
        for rule in ['gpt_oss_swiglu','silu_gate']:
            full=[];quant=[]
            for e in range(6):
                gate=x@weights[e,0];up=x@weights[e,1]
                if rule=='gpt_oss_swiglu':gate+=bias[e,0,:16];up+=bias[e,1,:16]
                y=activation(gate,up,rule)@weights[e,2].T
                if rule=='gpt_oss_swiglu':y+=bias[e,2]
                full.append(y)
                qx=np.stack([quantize_vector(v)[0] for v in x]);qw=[quantize_weight(w) for w in weights[e]]
                qg=qx@qw[0];qu=qx@qw[1]
                if rule=='gpt_oss_swiglu':qg+=quantize_vector(bias[e,0,:16])[0];qu+=quantize_vector(bias[e,1,:16])[0]
                qy=activation(qg,qu,rule)@qw[2].T
                if rule=='gpt_oss_swiglu':qy+=quantize_vector(bias[e,2])[0]
                quant.append(qy)
            gates=rng.random((16,6)).astype('float32');gates/=gates.sum(1,keepdims=True)
            contributions=[full[e]*gates[:,e,None] for e in range(6)];qcon=[quant[e]*gates[:,e,None] for e in range(6)]
            reference=sum(contributions);groups=[[0,2,4],[1,3,5]]
            local=sum(sum(contributions[e] for e in group) for group in groups)
            for method in c['methods']:np.testing.assert_allclose(local if not method.endswith('NO_LC') else sum(contributions),reference,rtol=1e-5,atol=1e-6)
            lc=sum(np.stack([quantize_vector(v)[0] for v in sum(qcon[e] for e in group)]) for group in groups)
            no_lc=sum(np.stack([quantize_vector(v)[0] for v in contribution]) for contribution in qcon)
            reports.append(dict(rule=rule,fp32_equivalent=True,fp8_LC=error(lc,reference),fp8_NO_LC=error(no_lc,reference)))
        # Nonlinear branches fuse on the intermediate dimension, never sum pre-activation.
        g1=x@weights[0,0];g2=x@weights[1,0];u1=x@weights[0,1];u2=x@weights[1,1]
        split=activation(g1,u1,'silu_gate')@weights[0,2].T+activation(g2,u2,'silu_gate')@weights[1,2].T
        fused=activation(np.concatenate([g1,g2],1),np.concatenate([u1,u2],1),'silu_gate')@np.concatenate([weights[0,2].T,weights[1,2].T],0)
        np.testing.assert_allclose(fused,split,rtol=1e-5,atol=1e-6)
        np.testing.assert_array_equal(e4m3fn([1.0625,1.1875,500,-500]),[1,1.25,448,-448])
        zero,scales=quantize_vector(np.zeros(32));assert scales==[1.0] and not zero.any()
        np.testing.assert_array_equal(e4m3fn(LEVELS),LEVELS)
        g=np.array([-10,8.]);u=np.array([-9,10.]);expected=np.array([-10/(1+np.exp(17.02))*(-6),7/(1+np.exp(-1.702*7))*8])
        np.testing.assert_allclose(activation(g,u,'gpt_oss_swiglu'),expected)
        return dict(tests=reports,ERNIE_fused_split=True,nearest_even=True,finite_max=448,zero_scale=1,scope='random d=32 h=16 tokens=16; not pretrained-model accuracy')
    check('fp32_math_FP8_reference_shared_GPT_activation',math_check)
    def determinism():
        m=c['models'][2];rows=[record(m,rank,i) for rank in range(2) for i in range(3)]
        a=Simulator(c,m,c['primary_proposal'],'layer','decode',rows,0).run();b=Simulator(c,m,c['primary_proposal'],'layer','decode',rows,0).run()
        assert stable_hash(a)==stable_hash(b)
        assert sum(r['tokens'] for r in a['rows'])==6
        return dict(result_sha256=stable_hash(a),exact_repeat=True,decode_rounds=len(a['rows']))
    check('deterministic_replay_and_decode_round_accounting',determinism)
    return dict(passed=all(x['passed'] for x in checks),checks=checks)
