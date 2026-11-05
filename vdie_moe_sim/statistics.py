"""Pair whole cohorts within sampled seeds, preserving cross-baseline pairing."""
import numpy as np
from .routing import rng_for

def paired_effects(left,right,seed_key,resamples=1000,confidence=.95):
    merged=left.merge(right,on=['seed','cohort'],suffixes=('_a','_b'),validate='one_to_one')
    if len(merged)!=len(left) or len(merged)!=len(right):
        raise ValueError('Unpaired cohorts')
    if not np.array_equal(merged.tokens_a,merged.tokens_b):
        raise ValueError('Different paired workload')
    cols=['request_layer_latency_s_a','request_layer_latency_s_b',
          'energy_frontend_proxy_j_a','energy_frontend_proxy_j_b',
          'energy_baseband_stress_j_a','energy_baseband_stress_j_b']
    groups=[g[cols].to_numpy() for _,g in merged.groupby('seed',sort=True)]
    def effect(mean):
        return np.asarray([mean[0]/mean[1],1-mean[3]/mean[2],1-mean[5]/mean[4]])
    mean=merged[cols].mean().to_numpy();point=effect(mean);samples=[]
    rng=rng_for(seed_key)
    for _ in range(resamples):
        selected=rng.integers(0,len(groups),size=len(groups));total=np.zeros(6);n=0
        for index in selected:
            g=groups[index];sample=g[rng.integers(0,len(g),size=len(g))]
            total+=sample.sum(axis=0);n+=len(sample)
        samples.append(effect(total/n))
    alpha=(1-confidence)/2;bounds=np.quantile(samples,[alpha,1-alpha],axis=0)
    out={}
    for j,name in enumerate(['speedup','frontend_saving','stress_saving']):
        out[name]=float(point[j]);out[name+'_low']=float(bounds[0,j]);out[name+'_high']=float(bounds[1,j])
    out.update(paired_cohorts=len(merged),paired_seeds=len(groups))
    return out
