"""Offline integrity, workload fairness, ledger closure and source provenance."""
from collections import Counter
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import pandas as pd
from .config import ROOT,digest_file,stable_digest,override

def check(condition,message):
    if not condition:
        raise ValueError('Offline audit: '+message)

def numeric_diagnosis(c):
    import ml_dtypes
    from .precision import POSITIVE,encode_e4m3fn,decode_e4m3fn,numerical_validation
    rng=np.random.default_rng(20261104)
    mid=(POSITIVE[1:]+POSITIVE[:-1])/2
    values=np.concatenate([rng.uniform(-448,448,1000000),POSITIVE,-POSITIVE,mid,-mid]).astype(np.float32)
    encoding=bool(np.array_equal(encode_e4m3fn(values),values.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)))
    codes=np.arange(256,dtype=np.uint8)
    decoding=bool(np.allclose(decode_e4m3fn(codes),codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32),equal_nan=True))
    check(encoding and decoding,'Independent FP8 encoding/decoding mismatch')
    return dict(independent_reference='ml_dtypes.float8_e4m3fn',reference_version=importlib.metadata.version('ml_dtypes'),
                encode_comparisons=len(values),encoding_bit_exact=encoding,all_256_decodings_match=decoding,
                end_to_end_fp8_checks=numerical_validation(c))

def audit_results(out,c,raw,manifest):
    started=time.monotonic();out=Path(out)
    plan={r['plan_id']:r for r in manifest['experiments']}
    check(len(plan)==4020 and len(raw)==4020 and raw.plan_id.nunique()==4020,'Plan completeness')
    check(set(plan)==set(raw.plan_id),'Unexpected or missing run IDs')
    treatments=Counter(r['treatment'] for r in plan.values())
    check(treatments=={'timing':2280,'energy_replay':1680,'equivalent_prefill_control':60},'Treatment counts')
    check(int((raw.family=='main').sum())==840,'Main experiment count')
    check(int((raw.family=='strong_tdma').sum())==60,'Strong TDMA count')
    check((raw.requests==128).all(),'Each run must contain 128 requests')
    main=raw[raw.family=='main']
    for _,g in main.groupby(['model','mode','batch','seed']):
        check(len(g)==8 and g.baseline.nunique()==8,'Eight-way ablation incomplete')
        check(g.trace_sha256.nunique()==1 and g.tokens.nunique()==1,'Cross-baseline workload differs')
    index=json.loads((out/'traces/index.json').read_text());check(len(index)==75,'Frozen trace coverage')
    inputs=json.loads((out/'input_manifest.json').read_text())
    traces={};trace_tokens=0
    for name,entry in index.items():
        path=out/'traces'/f'{name}.jsonl.gz';binary=out/'traces'/f'{name}.npz'
        check(digest_file(path)==entry['trace_sha256'],'Text trace checksum '+name)
        check(digest_file(binary)==entry['binary_sha256'],'Binary trace checksum '+name)
        desc=np.load(binary)['descriptors'];traces[name]=desc
        check(len(desc)==entry['tokens'],'Trace token count')
        mid,seed,mode,*_=name.split('-');model=c['models'][mid]
        check(len(set(desc[:,0]))==128,'Trace request count')
        selected=inputs['samples'][seed]
        check(len(set(selected))==128,'Request sampling without replacement')
        for request in range(128):
            rows=desc[desc[:,0]==request]
            check(np.all(rows[:,2]==selected[request]),'Cross-model sampled row identity')
            check(np.array_equal(rows[:,1],np.arange(len(rows))),'Trace token sequence')
        with gzip.open(path,'rt') as f:
            count=0
            for record,values in zip(f,desc,strict=True):
                r=json.loads(record);count+=1
                expected=[r['request_ordinal'],r['step'],r['row_id'],r['token_id'],*r['expert_ids']]
                check(np.array_equal(values,expected),'Text/binary trace discrepancy')
                check(len(set(r['expert_ids']))==model['top_k'],'Duplicate routed contribution')
                check(all(0<=e<model['routed_experts'] for e in r['expert_ids']),'Expert ID range')
                check(all(w>0 for w in r['gate_weights']) and math.isclose(sum(r['gate_weights']),1,abs_tol=1e-12),'Gate coefficients')
                check(r['input_trace_mode']=='dataset_tokens_synthetic_router','Trace provenance label')
        trace_tokens+=count
    from .run_all import engine_hash,calculation_hash,native_hash,verify_approval
    verify_approval(ROOT/'VDIE_MOE_SIM_SPEC.md')
    check(engine_hash()==manifest['engine_hash'],'Event model changed during experiments')
    lock=json.loads((out/'implementation_lock.json').read_text())
    check(lock['engine_hash']==engine_hash() and lock['calculation_hash']==calculation_hash()
          and lock['native_sha256']==native_hash(),'Implementation or compiled kernel changed during experiments')
    manifest['implementation_lock']=lock
    check(digest_file(ROOT/'VDIE_MOE_SIM_SPEC.md')==manifest['spec_sha256'],'Specification changed')
    records={};ledgers={};file_checks=0;timing_tokens=0;events=0
    for row in raw.to_dict('records'):
        rid=row['plan_id'];p=plan[rid];record=json.loads((out/'runs'/rid/'record.json').read_text());records[rid]=record
        check(record['plan_id']==rid and row['status']==p['status']==record['status'],'Run identity and status')
        check(record['engine_hash']==manifest['engine_hash'],'Mixed event-model versions')
        check(record['trace_sha256']==row['trace_sha256']==index[record['trace_name']]['trace_sha256'],'Run trace identity')
        for path,sha in record['files'].items():
            check(digest_file(out/path)==sha,'Output checksum '+path);file_checks+=1
        effective=override(c,p['overrides'])
        if p['treatment']=='timing':
            check(record['status']=='passed','Timing run did not pass audits')
            expected=stable_digest(dict(config=effective,engine=manifest['engine_hash'],
                         trace=record['trace_sha256'],model=row['model'],mode=row['mode'],batch=row['batch'],baseline=row['baseline']))
            check(record['signature']==expected,'Configuration/result signature')
            l=pd.read_parquet(out/'runs'/rid/'ledger.parquet');ledgers[rid]=l
            tokens=pd.read_parquet(out/'token_metrics.parquet'/f'{rid}.parquet')
            desc=traces[record['trace_name']]
            order=tokens.sort_values(['request_ordinal','step'])
            check(np.array_equal(order[['request_ordinal','step','row_id','token_id']].to_numpy(),desc[:,:4]),'Output token identity')
            check(np.array_equal(tokens.owner,tokens.request_ordinal%16),'Stable owner placement')
            check((tokens.completed_ps>tokens.ready_ps).all(),'Nonpositive token delay')
            check(len(tokens)==row['tokens'],'Output token conservation')
            check(l.resource_audit_passed.all() and l.token_conservation_passed.all(),'Internal resource/token audit')
            check(np.array_equal(l[[x for x in l if x.startswith('critical_')]].sum(axis=1),l.duration_ps),'Dependency path closure')
            for column in ['die_compute_ps','die_dma_ps']:
                check(all(max(a)<=duration for a,duration in zip(l[column],l.duration_ps)),'Resource busy-time exceeds measured duration')
            for column,limit in [('peak_rx_bytes',1048576),('peak_tx_bytes',1048576),('peak_partial_bytes',786432),
                                 ('peak_owner_bytes',262144),('peak_tile_bytes',524288),('peak_control_bytes',524288)]:
                check(all(max(a)<=limit for a in l[column]),'Workspace partition '+column)
            if row['mode']=='decode':
                rounds=pd.read_parquet(out/'runs'/rid/'rounds.parquet')
                expected=tokens.groupby(['cohort','step']).agg(active_requests=('request_ordinal','nunique'),
                         ready=('ready_ps','min'),last=('completed_ps','max'),ready_max=('ready_ps','max')).reset_index()
                check((expected.ready==expected.ready_max).all(),'Decode barrier release')
                expected['duration_ps']=expected['last']-expected.ready
                joined=expected.merge(rounds,on=['cohort','step'],suffixes=('_a','_b'),validate='one_to_one')
                check(len(joined)==len(expected)==len(rounds),'Decode round count')
                check(np.array_equal(joined.active_requests_a,joined.active_requests_b),'Actual active B')
                check(np.array_equal(joined.duration_ps_a,joined.duration_ps_b),'Decode duration')
                for _,rr in expected.groupby('cohort'):
                    check(np.array_equal(rr.ready.iloc[1:].to_numpy(),rr['last'].iloc[:-1].to_numpy()),'Decode next round causality')
            timing_tokens+=len(tokens);events+=int(l.events.sum())
        else:
            ref=p['reference_main_id'];base=records[ref]
            check(all(p[key]==plan[ref][key] for key in ['model','mode','batch','seed','baseline']),'Replay mismatched reference configuration')
            check(record['temporal_reference']==ref and row['temporal_run_id']==ref,'Replay points to wrong temporal run')
            check(record['trace_sha256']==base['trace_sha256'],'Replay changed workload')
            check(record['signature']==stable_digest([base['signature'],p['overrides'],p['treatment']]),'Replay signature')
            for name in ['mean_request_layer_s','mean_token_layer_s','sum_cohort_s','matrix_ops','read_bytes','write_bytes']:
                check(record['summary'][name]==base['summary'][name],'Energy-only replay changed timing or service')
            l=ledgers[ref]
        # Independently close both accounts from the aggregate components and duration.
        s=record['summary'];front=['compute','vector','edram_read','edram_write','refresh','memory_leak','rf_lo','rf_tx','rf_rx','control_idle']
        for key,value in s.items():
            actual=float(row[key])  # Optional Decode columns are strings after CSV reload with empty Prefill cells.
            equal=int(actual)==value and actual.is_integer() if isinstance(value,int) else math.isclose(actual,value,rel_tol=1e-12,abs_tol=1e-15)
            check(equal,'Run table differs from preserved record: '+key)
        frontend=math.fsum(s['energy_'+key+'_j'] for key in front)
        stress=frontend+s['energy_online_eq_j']+s['energy_ofdma_extra_j']
        check(math.isclose(frontend,s['energy_frontend_proxy_j'],rel_tol=1e-12),'Frontend account closure')
        check(math.isclose(stress,s['energy_baseband_stress_j'],rel_tol=1e-12),'Stress account closure')
        m=effective['memory'];seconds=int(l.duration_ps.sum())*1e-12
        refresh=16*64000000*8*m['refresh_pj_bit']/m['refresh_period_ps']*m['energy_scale']*seconds
        check(math.isclose(refresh,s['energy_refresh_j'],rel_tol=1e-12),'Full-capacity continuous refresh')
        check(math.isclose(s['frontend_proxy_uj_token'],frontend*1e6/s['tokens'],rel_tol=1e-12),'Actual-token energy denominator')
    source_paths=[ROOT/name for name in ['VDIE_MOE_SIM_SPEC.md','pyproject.toml','requirements.lock','setup.py','README.md',
                                       '.gitignore','.gitattributes','.github/workflows/ci.yml','pytest.ini']]
    for directory in ['vdie_moe_sim','tests','scripts','configs','docs/approval']:
        source_paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.so','.pyc'})
    source_hashes={str(p.relative_to(ROOT)):digest_file(p) for p in sorted(set(source_paths))}
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0]
    dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,text=True).strip())
    provenance=dict(source_files=source_hashes,source_sha256=stable_digest(source_hashes),
                    base_git_commit=git_head,working_tree_modified=dirty,compiler=compiler,python=platform.python_version(),
                    note='Source file hashes identify the current implementation; base commit alone does not include these modifications.')
    (out/'source_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    diagnosis=numeric_diagnosis(c)
    (out/'numeric_diagnosis.json').write_text(json.dumps(diagnosis,ensure_ascii=False,indent=2)+'\n')
    result=dict(passed=True,planned_records=len(plan),treatments=dict(treatments),frozen_traces=len(index),
                frozen_trace_tokens=trace_tokens,simulated_tokens=timing_tokens,processed_events=events,
                checked_result_files=file_checks,source_sha256=provenance['source_sha256'],numeric_diagnosis=diagnosis,
                acceptance_all_passed=manifest['validation']['all_acceptance_passed'],elapsed_wall_s=time.monotonic()-started)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result
