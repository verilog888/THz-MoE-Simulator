"""Read the uploaded JSON as the only source of adjustable parameters."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = 16  # Endpoint identifier, independent of the number of V-Die nodes.


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read_spec(path=None):
    path=Path(path or ROOT/'ToV_MoE_MINIMAL_SIM_SPEC.md')
    text=path.read_text(encoding='utf-8')
    blocks=re.findall(r'^~~~json\n(.*?)^~~~',text,re.M|re.S)
    if len(blocks)!=1:raise ValueError('Expected exactly one JSON configuration fence')
    config=json.loads(blocks[0])
    assert config['schema_version']=='tov-minimal-1.0'
    return config


def ceildiv(a,b):return (a+b-1)//b


def qbytes(c,n):return n+c['precision']['scale_bytes']*ceildiv(n,c['precision']['activation_block'])


def expert_bytes(c,m):
    d,h=m['d'],m['h'];b=c['precision']['weight_block'];s=c['precision']['scale_bytes']
    return 3*d*h+3*s*ceildiv(d,b)*ceildiv(h,b)+(qbytes(c,h)*2+qbytes(c,d) if m['expert_bias'] else 0)


def router_bytes(c,m):
    return m['d']*m['E']+c['precision']['scale_bytes']*ceildiv(m['d'],c['precision']['weight_block'])*ceildiv(m['E'],c['precision']['weight_block'])+(qbytes(c,m['E']) if m['router_bias'] else 0)


def control_ps(c,domain):
    r=c['radio'];participants=c['hardware']['dies']+(domain==0)
    ns=8*participants*r['report_bytes']/(r['unicast_gbps']*r['rate_scale'])
    ns+=8*(r['grant_base_bytes']+r['rus']*r['grant_bytes_per_ru'])/(r['broadcast_gbps']*r['rate_scale'])
    return math.ceil(ns+r['controller_ns'])*1000


def derive(c):
    h=c['hardware'];peak=2*h['pe_per_die']*h['mac_rows']*h['mac_cols']*h['clock_hz']
    hb=h['banks_per_die']*h['hb_pins_per_bank']*h['hb_pin_gbps']*1e9/8
    return dict(peak_ops_per_second=peak,system_peak_ops_per_second=peak*h['dies'],hb_bytes_per_second=hb,
                kernel_bytes_per_second=hb*(1-h['dma_bandwidth_fraction'])*h['kernel_memory_efficiency'],
                dma_bytes_per_second=hb*h['dma_bandwidth_fraction'],effective_ops_per_second=peak*h['compute_efficiency'],
                scalar_ops_per_second=h['vector_elements_per_cycle']*h['clock_hz'],
                ofdm_efficiency=c['radio']['data_bins']/(c['radio']['fft_bins']+c['radio']['cp_bins']),
                control_base_ps=control_ps(c,0),control_peer_ps=control_ps(c,1),
                tsv_norm_port_gbps=c['tsv']['normalized_total_gbps']/h['dies'],
                tsv_high_port_gbps=8*c['tsv']['high_bw_total_GBps']/h['dies'],
                sram_bytes_per_die=16*(32+16)*1024+128*1024)


def capacity(c,m):
    n=c['hardware']['dies'];out=[]
    for node in range(n):
        routed=len(range(node,m['E'],n));shared=int(m['shared']>0)
        weights=(routed+shared)*expert_bytes(c,m)+router_bytes(c,m)
        total=weights+c['hardware']['staging_bytes_per_die']
        out.append(dict(model=m['id'],dies=n,node=node,routed_experts=routed,shared_replicas=shared,
                        resident_weight_bytes=weights,staging_bytes=c['hardware']['staging_bytes_per_die'],
                        allocated_bytes=total,capacity_bytes=c['hardware']['dram_bytes_per_die'],
                        passed=total<=c['hardware']['dram_bytes_per_die']))
    return out


def resolve(c,overrides):
    result=deepcopy(c)
    for key,value in overrides.items():
        parts=key.split('.');d=result
        for p in parts[:-1]:d=d[p]
        d[parts[-1]]=value
    return result


def placement_hash(c,m):
    n=c['hardware']['dies']
    return stable_hash(dict(nodes=n,routed=[e%n for e in range(m['E'])],
                            owners=[r%n for r in range(c['workload']['requests_per_seed'])],
                            shared='even shared-0; odd shared-1; owner and owner xor 1' if m['shared'] else 'none'))
