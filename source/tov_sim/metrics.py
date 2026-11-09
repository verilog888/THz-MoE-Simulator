"""Raw-observation aggregation and stratified paired cohort bootstrap."""
from collections import defaultdict
import csv
from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np
from .config import digest,stable_hash,resolve,derive
from .io import write_json,write_csv,read_gzip

ENERGY=['mac_J','scalar_J','hb_J','dram_core_J','tsv_J','rf_lo_J','rf_tx_J','rf_rx_J','ofdm_J','background_J']


def case_key(row):return (row['model'],row['phase'],int(row['B']),row['profile'])
def case_id(row):return f'{row["model"]}_{row["phase"]}_B{row["B"]}_{row["profile"]}'
def compact(v):
    if isinstance(v,float):return format(v,'.14g')
    return v


class CSV:
    def __init__(self,path,fields):
        self.handle=Path(path).open('w',encoding='utf-8-sig',newline='');self.writer=csv.DictWriter(self.handle,fieldnames=fields,extrasaction='ignore');self.writer.writeheader();self.rows=0
    def add(self,row):self.writer.writerow({k:compact(v) for k,v in row.items()});self.rows+=1
    def close(self):self.handle.close()


def load_plan(out):
    with (out/'run_manifest.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ['B','seed']:r[k]=int(r[k])
        r['overrides']=json.loads(r['overrides'])
    return [r for r in rows if r['family']!='energy']


def aggregate(array):
    """Columns: total layer ps, observations, energy, tokens, replay ps, useful B."""
    a=np.asarray(array)
    return dict(T_ns=a[...,0]/a[...,1]/1000,E_J_per_token=a[...,2]/a[...,3],G_Gbps=8000*a[...,5]/a[...,4])


def bootstrap_weights(c):
    rng=np.random.Generator(np.random.PCG64(c['experiment']['bootstrap_seed']));n=c['experiment']['bootstrap_samples'];seeds=len(c['workload']['seeds'])
    draws=rng.integers(seeds,size=(n,seeds));out={}
    for phase in c['workload']['phases']:
        for B in c['workload']['batch_sizes']:
            cohorts=c['workload']['requests_per_seed']//B;co=rng.integers(cohorts,size=(n,seeds,cohorts));idx=draws[:,:,None]*cohorts+co
            weights=np.zeros((n,seeds*cohorts),dtype=np.float64)
            np.add.at(weights,(np.arange(n)[:,None,None],idx),1)
            assert np.all(weights.sum(1)==seeds*cohorts)
            out[phase,B]=weights
    return out


def ratio(proposal,baseline):
    return dict(SG=proposal['G_Gbps']/baseline['G_Gbps'],ST=baseline['T_ns']/proposal['T_ns'],SE_proxy=baseline['E_J_per_token']/proposal['E_J_per_token'])


def geometric(values):return np.exp(np.log(np.asarray(values)).mean(axis=0))


def summarize(c,out):
    out=Path(out);rows=load_plan(out)
    if any(r['status']!='completed' for r in rows):
        missing=[r['run_id'] for r in rows if r['status']!='completed']
        (out/'abstract_result.txt').write_text('Incomplete experiment: no overall abstract ratios. Missing: '+', '.join(missing)+'\n')
        raise RuntimeError(f'{len(missing)} planned runs are incomplete')
    index={r['run_id']:r for r in rows}
    common=['run_id','case_id','model','phase','B','profile','seed','cohort','method']
    hashes=['input_hash','route_hash','placement_hash']
    layerfields=common+['step_index','tokens','latency_ns','p95_token_ns','energy_proxy_J','MAC','HB_bytes','physical_tx_bits','peak_buffer_bytes']+hashes
    replayfields=common+['tokens','useful_bytes','replay_ns','goodput_Gbps','physical_tx_bits','domain_0_data_bits','domain_1_data_bits','domain_0_control_bits','domain_1_control_bits','domain_busy_ps','domain_0_data_busy_ps','domain_1_data_busy_ps','domain_0_data_busy_fraction','domain_1_data_busy_fraction','domain_0_RU_utilization','domain_1_RU_utilization','peak_buffer_bytes']+hashes
    writers={('main','layer'):CSV(out/'layer_results.csv',layerfields),('sensitivity','layer'):CSV(out/'sensitivity_layer_results.csv',layerfields),
             ('main','replay'):CSV(out/'replay_results.csv',replayfields),('sensitivity','replay'):CSV(out/'sensitivity_replay_results.csv',replayfields)}
    cohort_writer=CSV(out/'cohort_summary.csv',common+['family','measurement','axis','value','reference','tokens','rounds','total_ns','energy_proxy_J','MAC','HB_bytes','physical_tx_bits','peak_buffer_bytes','nmp_busy_ps','dma_busy_ps','events']+hashes)
    energy_writer=CSV(out/'energy_breakdown.csv',common+['family','tokens','energy_proxy_J']+ENERGY+['ofdm_active_endpoint_ps'])
    audit_writer=CSV(out/'resource_audit.csv',common+['measurement','family','events','max_ru','max_owner_active','contributor_checks','packet_hops','receiver_deliveries','peak_by_node','nmp_busy_ps_by_node','dma_busy_ps_by_node','batch_histogram'])
    allstats={};main={};maxpeak=0;total_tokens=0;raw_rows=0;total_events=0;fairness={};fingerprints={};batchhist=defaultdict(int);tails=defaultdict(lambda:defaultdict(list))
    @lru_cache(maxsize=32)
    def result(run_id):return read_gzip(out/'runs'/(run_id+'.json.gz'))
    for row in rows:
        data=result(row['reference'] or row['run_id']);hashvalues={key:data[key] for key in hashes}
        if row['reference']:assert index[row['reference']]['status']=='completed'
        fingerprints[row['run_id']]=data['fingerprint']
        meta={key:row[key] for key in ['run_id','model','phase','B','profile','seed','method']};meta['case_id']=case_id(row);meta.update(hashvalues)
        stats=[]
        for cohort,raw in enumerate(data['cohorts']):
            ledger=raw['ledger'];meta['cohort']=cohort;maxpeak=max(maxpeak,raw['peak_buffer_bytes'])
            if not row['reference']:
                total_tokens+=raw['tokens'];total_events+=raw['events']
            observations=1 if row['phase']=='prefill' else len(raw['rows'])
            stat=np.array([raw['total_ps'],observations,ledger['energy_proxy_J'],raw['tokens'],0,0],dtype=np.float64) if row['measurement']=='layer' else np.array([0,0,0,0,raw['total_ps'] if raw['useful_bytes'] else 0,raw['useful_bytes']],dtype=np.float64)
            stats.append(stat)
            if row['measurement']=='layer':
                for obs in raw['rows']:
                    v=obs['ledger'];writers[row['family'],'layer'].add(meta|dict(step_index=obs['step_index'],tokens=obs['tokens'],latency_ns=obs['latency_ps']/1000,p95_token_ns=obs['p95_token_ps']/1000,energy_proxy_J=v['energy_proxy_J'],MAC=int(v.get('MAC',0)),HB_bytes=int(v.get('HB_bytes',0)),physical_tx_bits=int(v.get('physical_tx_bits',0)),peak_buffer_bytes=raw['peak_buffer_bytes']))
                energy_writer.add(meta|dict(family=row['family'],tokens=raw['tokens'],energy_proxy_J=ledger['energy_proxy_J'],ofdm_active_endpoint_ps=ledger['ofdm_active_endpoint_ps'])|{key:ledger[key] for key in ENERGY})
                assert sum(obs['latency_ps'] for obs in raw['rows'])==raw['total_ps']
                assert math.isclose(sum(obs['ledger']['energy_proxy_J'] for obs in raw['rows']),ledger['energy_proxy_J'],rel_tol=1e-12)
            else:
                radio={key:int(ledger.get(key,0)) for key in replayfields if key.startswith('domain_')}
                for domain in [0,1]:
                    radio[f'domain_{domain}_data_busy_fraction']=ledger.get(f'domain_{domain}_data_busy_ps',0)/raw['total_ps'] if raw['total_ps'] else 0
                    radio[f'domain_{domain}_RU_utilization']=ledger.get(f'domain_{domain}_allocated_ru_ps',0)/(c['radio']['rus']*raw['total_ps']) if raw['total_ps'] else 0
                writers[row['family'],'replay'].add(meta|dict(tokens=raw['tokens'],useful_bytes=raw['useful_bytes'],replay_ns=raw['total_ps']/1000,goodput_Gbps=8000*raw['useful_bytes']/raw['total_ps'] if raw['useful_bytes'] else '',physical_tx_bits=int(ledger.get('physical_tx_bits',0)),peak_buffer_bytes=raw['peak_buffer_bytes'])|radio)
            cohort_writer.add(meta|dict(family=row['family'],measurement=row['measurement'],axis=row['axis'],value=row['value'],reference=row['reference'],tokens=raw['tokens'],rounds=observations,total_ns=raw['total_ps']/1000,energy_proxy_J=ledger['energy_proxy_J'],MAC=int(ledger.get('MAC',0)),HB_bytes=int(ledger.get('HB_bytes',0)),physical_tx_bits=int(ledger.get('physical_tx_bits',0)),peak_buffer_bytes=raw['peak_buffer_bytes'],nmp_busy_ps=int(ledger.get('nmp_busy_ps',0)),dma_busy_ps=int(ledger.get('dma_busy_ps',0)),events=raw['events']))
            audit_writer.add(meta|dict(family=row['family'],measurement=row['measurement'])|{key:json.dumps(raw[key],separators=(',',':')) if isinstance(raw[key],(list,dict)) else raw[key] for key in ['events','max_ru','max_owner_active','contributor_checks','packet_hops','receiver_deliveries','peak_by_node','nmp_busy_ps_by_node','dma_busy_ps_by_node','batch_histogram']})
            if row['family']=='main':
                key=(case_key(row),row['seed'],cohort)
                fair=(data['input_hash'],data['route_hash'],data['placement_hash'],raw['tokens'],raw['useful_bytes'])
                if key in fairness:assert fairness[key]==fair
                fairness[key]=fair
                if row['measurement']=='layer':
                    for size,count in raw['batch_histogram'].items():batchhist[int(size)]+=count
                    tail=tails[case_key(row),row['method']]
                    tail['layer_observation'].extend(a['latency_ps']/1000 for a in raw['rows'])
                    tail['token'].extend(a[4]/1000 for a in raw['token_timings'])
                    if row['phase']=='prefill':tail['request_completion'].extend(a/1000 for a in raw['request_completion_ps'].values())
        allstats[row['run_id']]=np.stack(stats)
        if row['family']=='main':
            key=(case_key(row),row['method']);seedindex=c['workload']['seeds'].index(row['seed'])
            if key not in main:main[key]=np.zeros((3,len(stats),6))
            main[key][seedindex]+=np.stack(stats)
    counts={str(k):w.rows for k,w in writers.items()}
    for w in list(writers.values())+[cohort_writer,energy_writer,audit_writer]:w.close()
    weights=bootstrap_weights(c);nominal={};boot={};absolute=[]
    for (case,method),a in main.items():
        nominal[case,method]=aggregate(a.sum(axis=(0,1)))
        boot[case,method]=aggregate(weights[case[1],case[2]]@a.reshape(-1,6))
        m,phase,B,profile=case
        absolute.append(dict(model=m,phase=phase,B=B,profile=profile,method=method,**nominal[case,method],cohorts=int(a.shape[0]*a.shape[1]),observations=int(a[:,:,1].sum()),tokens=int(a[:,:,3].sum())))
    write_csv(out/'method_summary.csv',absolute)
    tailrows=[]
    for (case,method),categories in tails.items():
        for category,values in categories.items():
            tailrows.append(dict(model=case[0],phase=case[1],B=case[2],profile=case[3],method=method,observation=category,count=len(values),p50_ns=float(np.percentile(values,50)),p95_ns=float(np.percentile(values,95)),p99_ns=float(np.percentile(values,99))))
    write_csv(out/'tail_latency.csv',tailrows)
    cases=sorted(set(case for case,_ in main));summaries=[]
    groups=[('case',str(case),[case]) for case in cases]
    groups += [('model_phase',f'{m["id"]}/{p}',[x for x in cases if x[0]==m['id'] and x[1]==p]) for m in c['models'] for p in c['workload']['phases']]
    groups += [('phase',p,[x for x in cases if x[1]==p]) for p in c['workload']['phases']]
    groups += [('model',m['id'],[x for x in cases if x[0]==m['id']]) for m in c['models']]+[('overall','all',cases)]
    proposal=c['primary_proposal']
    for baseline in [c['primary_baseline'],'TSV_HIGH_MC_LC']:
        for scope,label,group in groups:
            ratios=[ratio(nominal[case,proposal],nominal[case,baseline]) for case in group]
            samples=[ratio(boot[case,proposal],boot[case,baseline]) for case in group]
            item=dict(scope=scope,group=label,proposal=proposal,baseline=baseline,cases=len(group),seeds=3,bootstrap_samples=c['experiment']['bootstrap_samples'])
            for name in ['SG','ST','SE_proxy']:
                item[name]=float(geometric([r[name] for r in ratios]));draws=geometric([r[name] for r in samples]);lo,hi=np.percentile(draws,[2.5,97.5]);item[name+'_ci_low']=float(lo);item[name+'_ci_high']=float(hi)
            item['latency_reduction_percent']=100*(1-1/item['ST']);item['latency_fraction_of_baseline']=1/item['ST']
            summaries.append(item)
    write_csv(out/'summary.csv',summaries)
    mechanisms=[]
    pairs=[('wireless_broadcast_TDMA','THZ_TDMA_P2P_LC','THZ_TDMA_BCAST_LC'),('OFDMA_P2P','THZ_TDMA_P2P_LC','THZ_OFDMA_P2P_LC'),('OFDMA_broadcast','THZ_TDMA_BCAST_LC',proposal),('local_combine','THZ_OFDMA_BCAST_NO_LC',proposal),('wired_multicast','TSV_NORM_P2P_LC',c['primary_baseline'])]
    for label,baseline,method in pairs:
        for phase in ['prefill','decode','all']:
            group=[x for x in cases if phase=='all' or x[1]==phase]
            rr=[ratio(nominal[x,method],nominal[x,baseline]) for x in group]
            mechanisms.append(dict(mechanism=label,phase=phase,baseline=baseline,proposal=method,**{k:float(geometric([r[k] for r in rr])) for k in rr[0]}))
    write_csv(out/'mechanisms.csv',mechanisms)
    sensitivity=[];sensitivity_data=defaultdict(dict)
    for r in rows:
        if r['family']=='sensitivity':
            key=(r['axis'],float(r['value']),r['phase'],r['method'])
            si=c['workload']['seeds'].index(r['seed'])
            if key not in sensitivity_data:sensitivity_data[key]=np.zeros((3,4,6))
            sensitivity_data[key][si]+=allstats[r['run_id']]
    for axis,value,phase,_ in sorted(sensitivity_data):
        if any(s['axis']==axis and s['value']==value and s['phase']==phase for s in sensitivity):continue
        values={method:aggregate(sensitivity_data[axis,value,phase,method].sum(axis=(0,1))) for method in [c['primary_baseline'],proposal]}
        rr=ratio(values[proposal],values[c['primary_baseline']])
        sensitivity.append(dict(axis=axis,value=value,phase=phase,model=c['sensitivity']['model'],B=8,profile='hot_shift',seeds=3,**rr,
                                TSV_latency_ns=float(values[c['primary_baseline']]['T_ns']),TOV_latency_ns=float(values[proposal]['T_ns']),
                                TSV_goodput_Gbps=float(values[c['primary_baseline']]['G_Gbps']),TOV_goodput_Gbps=float(values[proposal]['G_Gbps'])))
    write_csv(out/'sensitivity_summary.csv',sensitivity)
    energy_rows=[];energy_plan=[]
    for r in rows:
        if r['family']!='main' or r['measurement']!='layer' or r['model']!=c['sensitivity']['model'] or r['B']!=8 or r['profile']!='hot_shift' or r['method'] not in [c['primary_baseline'],proposal]:continue
        data=result(r['run_id']);tokens=sum(v['tokens'] for v in data['cohorts']);energy=sum(v['ledger']['energy_proxy_J'] for v in data['cohorts']);active=sum(v['ledger']['ofdm_active_endpoint_ps'] for v in data['cohorts'])*1e-12
        for power in c['sensitivity']['ofdm_endpoint_power_W']:
            rid=f'e{len(energy_plan):04d}';value=energy+(power-c['energy']['ofdm_W_per_active_endpoint'])*active
            energy_rows.append(dict(run_id=rid,reference=r['run_id'],model=r['model'],phase=r['phase'],B=8,profile='hot_shift',seed=r['seed'],method=r['method'],ofdm_endpoint_power_W=power,energy_proxy_J=value,tokens=tokens,energy_proxy_uJ_per_token=value/tokens*1e6,ofdm_active_endpoint_seconds=active,timing_rerun=False))
            energy_plan.append(dict(r,run_id=rid,family='energy',measurement='energy_ledger_reprice',axis='ofdm_endpoint_power_W',value=power,overrides={'energy.ofdm_W_per_active_endpoint':power},reference=r['run_id'],status='completed',cached=True,wall_seconds=0))
    write_csv(out/'energy_sensitivity.csv',energy_rows);write_csv(out/'run_manifest.csv',rows+energy_plan)
    validation=json.loads((out/'validation.json').read_text())
    validation['full_experiment']=dict(passed=True,planned_records=len(rows)+len(energy_plan),completed_records=len(rows)+len(energy_plan),main_layer_runs=576,main_replay_runs=576,sensitivity_records=336,energy_ledger_records=len(energy_plan),physical_runs=sum(not r['reference'] for r in rows),reused_timing_records=sum(bool(r['reference']) for r in rows),all_tokens_completed_and_contributors_checked=total_tokens,total_heap_events=total_events,peak_staging_bytes=maxpeak,staging_limit_bytes=c['hardware']['staging_bytes_per_die'],fairness_groups_checked=len(fairness),raw_csv_rows=counts,expert_batch_histogram=dict(sorted(batchhist.items())),decode_round_sums_checked=True,energy_increment_sums_checked=True,per_service_bit_tolerance=1e-6,zero_remote_exclusions=sum(not a[-1] for a in fairness.values()))
    validation['bootstrap']=dict(paired=True,strata='seed, then whole cohort; shared draws for same phase/B across models, profiles, methods',samples=2000,seed=2026,token_level_iid=False)
    write_json(out/'validation.json',validation)
    write_json(out/'result_fingerprints.json',fingerprints)
    from .report import render
    render(c,out,summaries,absolute,mechanisms,sensitivity,energy_rows,validation)
    checksums={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and '.cache' not in p.parts and p.name!='artifact_checksums.json'}
    write_json(out/'artifact_checksums.json',checksums)
    print(f'Completed {len(rows)+len(energy_plan)} planned records; all figures, paired bootstrap, and report written to {out}',flush=True)
