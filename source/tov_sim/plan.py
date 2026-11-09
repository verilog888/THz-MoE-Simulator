"""All 1,152 main configurations and one-axis sensitivities, with reuse."""
import itertools
from .config import stable_hash


def plan(c):
    rows=[];main={}
    def add(family,measurement,model,phase,B,profile,seed,method,axis='',value='',overrides=None,reference=''):
        row=dict(run_id=f'{family[:1]}{len(rows):04d}',family=family,measurement=measurement,model=model,phase=phase,B=B,
                 profile=profile,seed=seed,method=method,axis=axis,value=value,overrides=overrides or {},reference=reference,
                 status='planned',failure_reason='')
        rows.append(row);return row['run_id']
    w=c['workload']
    for measurement,m,phase,B,profile,seed,method in itertools.product(['layer','replay'],c['models'],w['phases'],w['batch_sizes'],w['routing_profiles'],w['seeds'],c['methods']):
        mid=m['id'];main[measurement,mid,phase,B,profile,seed,method]=add('main',measurement,mid,phase,B,profile,seed,method)
    assert len(rows)==1152
    s=c['sensitivity'];base=c['primary_baseline'];proposal=c['primary_proposal']
    axes=[('dies',s['die_counts'],'hardware.dies',c['hardware']['dies'],[base,proposal]),
          ('tsv_budget',s['tsv_budget_multipliers'],'tsv.normalized_total_gbps',1,[base,proposal]),
          ('thz_rate',s['thz_rate_scales'],'radio.rate_scale',1,[base,proposal]),
          ('memory_efficiency',s['memory_efficiencies'],'hardware.kernel_memory_efficiency',c['hardware']['kernel_memory_efficiency'],[base,proposal])]
    for axis,values,key,default,methods in axes:
        for value,measurement,phase,seed,method in itertools.product(values,['layer','replay'],w['phases'],w['seeds'],methods):
            reference=main[measurement,s['model'],phase,s['B'],s['routing_profile'],seed,method]
            changed=value!=default and not (axis=='tsv_budget' and method==proposal or axis=='thz_rate' and method==base)
            # With compute removed, kernel memory efficiency cannot change replay.
            if axis=='memory_efficiency' and measurement=='replay':changed=False
            overrides={key:value*c['tsv']['normalized_total_gbps'] if axis=='tsv_budget' else value} if changed else {}
            add('sensitivity',measurement,s['model'],phase,s['B'],s['routing_profile'],seed,method,axis,value,overrides,'' if changed else reference)
    return rows
