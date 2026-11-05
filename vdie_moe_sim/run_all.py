"""Single command for approved, frozen-input, resumable complete experiments."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import yaml
from .config import ROOT,load_spec,override,digest_file,stable_digest
from .workload import prepare_data,prepare_traces,trace_name
from .metrics import tables,summarize
from .precision import numerical_validation

TRACE_CACHE={}

def atomic_json(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def engine_hash():
    paths=[ROOT/'vdie_moe_sim'/x for x in ['native/core.cpp','simulator.py','config.py','packets.py','compute.py','memory.py','network.py']]
    return stable_digest({str(p.relative_to(ROOT)):digest_file(p) for p in paths})

def calculation_hash():
    names=['energy.py','metrics.py','workload.py','routing.py','placement.py','precision.py','statistics.py']
    paths=[ROOT/'vdie_moe_sim'/name for name in names]
    paths.extend([ROOT/'docs/approval/experiment_plan.csv',ROOT/'configs/review_options.yaml'])
    return stable_digest({str(p.relative_to(ROOT)):digest_file(p) for p in paths})

def native_hash():
    from . import _native
    return digest_file(_native.__file__)

def verify_approval(spec):
    approval=yaml.safe_load((ROOT/'configs/review_options.yaml').read_text())
    if not approval['simulation_allowed'] or approval['status']!='approved':
        raise PermissionError('User parameter approval required')
    if digest_file(spec)!=approval['original_spec_sha256']:
        raise PermissionError('Specification differs from the user-approved parameter version')
    review=json.loads((ROOT/'docs/approval/review_manifest.json').read_text())
    if approval['approval_id']!=review['approval_id']:
        raise PermissionError('Approval version differs from the parameter review')
    for name,sha in review['files'].items():
        if digest_file(ROOT/'docs/approval'/name)!=sha:
            raise PermissionError('Approved review artifact was modified: '+name)
    return approval

def read_plan():
    with (ROOT/'docs/approval/experiment_plan.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    for r in rows:
        r['batch']=int(r['batch']);r['seed']=int(r['seed'])
        r['overrides']=json.loads(r.pop('overrides_json'));r['status']='not_run'
    if len(rows)!=4020 or len({r['plan_id'] for r in rows})!=4020:
        raise ValueError('Approved experiment plan must contain 4020 distinct records')
    return rows

def execute_one(row,c,out,code_hash,trace_index):
    from .simulator import simulate
    out=Path(out);run=out/'runs'/row['plan_id'];run.mkdir(parents=True,exist_ok=True)
    effective=override(c,row['overrides'])
    profile=effective['workload']['routing_profile']
    reuse=effective['workload']['decode_reuse_probability'] if row['mode']=='decode' else 0.
    name=trace_name(row['model'],row['seed'],row['mode'],profile,reuse)
    signature=stable_digest(dict(config=effective,engine=code_hash,trace=trace_index[name]['trace_sha256'],
                                 model=row['model'],mode=row['mode'],batch=row['batch'],baseline=row['baseline']))
    record=run/'record.json'
    if record.exists():
        old=json.loads(record.read_text())
        if old['signature']!=signature:raise ValueError(f"Stale or incompatible result: {row['plan_id']}")
        for rel,sha in old['files'].items():
            if digest_file(out/rel)!=sha:raise ValueError('Existing result checksum mismatch')
        return old
    if name not in TRACE_CACHE:
        path=out/'traces'/f'{name}.npz'
        if digest_file(path)!=trace_index[name]['binary_sha256']:raise ValueError('Frozen binary trace changed')
        TRACE_CACHE[name]=np.load(path)['descriptors']
    desc=TRACE_CACHE[name];start=time.monotonic()
    result=simulate(effective,row['model'],desc,row['batch'],row['mode'],row['baseline'])
    ledgers,tokens,rounds=tables(result,effective,row['model'],desc,row['baseline'])
    summary,units,requests=summarize(ledgers,tokens,rounds,effective,row['mode'])
    token_path=out/'token_metrics.parquet'/f"{row['plan_id']}.parquet"
    tokens.insert(0,'temporal_run_id',row['plan_id']);tokens.to_parquet(token_path,index=False,compression='zstd')
    ledgers.to_parquet(run/'ledger.parquet',index=False,compression='zstd')
    rounds.to_parquet(run/'rounds.parquet',index=False,compression='zstd')
    units.to_parquet(run/'cohorts.parquet',index=False,compression='zstd')
    requests.to_parquet(run/'requests.parquet',index=False,compression='zstd')
    paths=[token_path,*[run/x for x in ['ledger.parquet','rounds.parquet','cohorts.parquet','requests.parquet']]]
    record_data=dict(plan_id=row['plan_id'],signature=signature,engine_hash=code_hash,
                     trace_sha256=trace_index[name]['trace_sha256'],trace_name=name,
                     elapsed_wall_s=time.monotonic()-start,summary=summary,
                     files={str(p.relative_to(out)):digest_file(p) for p in paths},status='passed')
    atomic_json(record,record_data)
    return record_data

def reprice(row,c,out,code_hash,trace_index):
    out=Path(out);ref=row['reference_main_id'];source=out/'runs'/ref
    base=json.loads((source/'record.json').read_text())
    if base['engine_hash']!=code_hash:raise ValueError('Energy replay references a different engine')
    effective=override(c,row['overrides'])
    ledgers=pd.read_parquet(source/'ledger.parquet')
    tokens=pd.read_parquet(out/'token_metrics.parquet'/f'{ref}.parquet')
    rounds=pd.read_parquet(source/'rounds.parquet')
    summary,units,_=summarize(ledgers,tokens,rounds,effective,row['mode'])
    run=out/'runs'/row['plan_id'];run.mkdir(exist_ok=True)
    units.to_parquet(run/'cohorts.parquet',index=False,compression='zstd')
    record=dict(plan_id=row['plan_id'],temporal_reference=ref,engine_hash=code_hash,
                trace_sha256=base['trace_sha256'],trace_name=base['trace_name'],
                signature=stable_digest([base['signature'],row['overrides'],row['treatment']]),
                status='energy_recomputed' if row['treatment']=='energy_replay' else 'equivalent_control_reused',
                summary=summary,files={str((run/'cohorts.parquet').relative_to(out)):digest_file(run/'cohorts.parquet')})
    atomic_json(run/'record.json',record);return record

def validate_tests(out):
    out=Path(out)
    results={}
    for label,marker in [('structural','not numeric_gate'),('numeric_gate','numeric_gate')]:
        log=out/f'{label}_tests.log';xml=out/f'{label}_tests.xml'
        with log.open('w') as f:
            proc=subprocess.run([sys.executable,'-m','pytest','-q','-m',marker,f'--junitxml={xml}'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        suite=ET.parse(xml).getroot().find('testsuite')
        results[label]=dict(exit_code=proc.returncode,**{k:int(suite.get(k,0)) for k in ['tests','failures','errors','skipped']})
        if label=='structural' and proc.returncode:raise RuntimeError(f'Structural tests failed: {log}')
    results['fp8_cases']=numerical_validation(load_spec())
    results['all_acceptance_passed']=results['numeric_gate']['exit_code']==0
    atomic_json(out/'validation.json',results)
    return results

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',default=str(ROOT/'VDIE_MOE_SIM_SPEC.md'))
    parser.add_argument('--output',default='results')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--require-numeric-gate',action='store_true',
                        help='Stop before timing experiments if the independently reported numeric acceptance fails.')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv);out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    approval=verify_approval(args.spec)
    c=load_spec(args.spec);plan=read_plan();code_hash=engine_hash()
    for d in ['runs','traces','figures','token_metrics.parquet']: (out/d).mkdir(exist_ok=True)
    if (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        if not args.resume:raise ValueError('Output already has a manifest; use --resume')
        if old['engine_hash']!=code_hash or old['spec_sha256']!=digest_file(args.spec):
            raise ValueError('Existing results use different physics code or specification')
        lock=old.get('implementation_lock')
        if lock is None and (out/'implementation_lock.json').exists():
            lock=json.loads((out/'implementation_lock.json').read_text())
        if lock is None or lock['calculation_hash']!=calculation_hash() or lock['native_sha256']!=native_hash():
            raise ValueError('Existing calculation code, approved plan or compiled kernel differs')
    if not (out/'traces/index.json').exists():
        encoded,inputs=prepare_data(c,out);traces=prepare_traces(c,out,encoded,inputs)
    else:
        traces=json.loads((out/'traces/index.json').read_text())
        for name,record in traces.items():
            if digest_file(out/'traces'/f'{name}.jsonl.gz')!=record['trace_sha256']:
                raise ValueError('Frozen routing trace hash mismatch')
    validation=validate_tests(out)
    manifest=dict(spec_sha256=digest_file(args.spec),config_sha256=stable_digest(c),engine_hash=code_hash,
                  approval=approval,complete=False,experiments_complete=False,validation=validation,
                  planned_records=len(plan),expected_time_simulations=2280,expected_energy_replays=1680,expected_equivalent_controls=60,
                  dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
                  input_manifest=json.loads((out/'input_manifest.json').read_text()),
                  seed_list=c['simulation']['seed_list'],input_trace_mode='dataset_tokens_synthetic_router',
                  experiments=plan,status='prepared')
    manifest['implementation_lock']=dict(calculation_hash=calculation_hash(),native_sha256=native_hash(),engine_hash=code_hash)
    atomic_json(out/'implementation_lock.json',manifest['implementation_lock'])
    atomic_json(out/'manifest.json',manifest)
    (out/'config_resolved.yaml').write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False))
    pd.read_csv(ROOT/'docs/approval/parameters.csv').assign(status='approved').to_csv(out/'parameters.csv',index=False)
    if args.prepare_only:return 0
    if not validation['all_acceptance_passed'] and args.require_numeric_gate:
        raise RuntimeError('FP8 0.05 NRMSE acceptance failed; preserved in validation.json. Awaiting user decision.')
    if not validation['all_acceptance_passed']:
        manifest['numeric_failure_handling']='Retain approved 0.05 threshold and failed assertions; continue already authorized timing and energy experiments; complete remains false.'
        print('NUMERIC GATE FAILED: retaining failures and 0.05 threshold; timing/energy experiments continue, complete=false.',flush=True)
    timing=[r for r in plan if r['treatment']=='timing']
    completed={};manifest['status']='running';atomic_json(out/'manifest.json',manifest)
    with ProcessPoolExecutor(max_workers=min(max(1,args.workers),4)) as pool:
        futures={pool.submit(execute_one,r,c,str(out),code_hash,traces):r for r in timing}
        for future in as_completed(futures):
            row=futures[future]
            try:record=future.result()
            except Exception as error:
                row['status']='failed';row['error']=str(error);manifest['status']='failed'
                atomic_json(out/'manifest.json',manifest)
                for pending in futures:pending.cancel()
                raise
            row['status']=record['status'];row['trace_sha256']=record['trace_sha256'];completed[row['plan_id']]=record
            print(f"TIME {len(completed)}/{len(timing)} {row['plan_id']} {row['model']} {row['mode']} B{row['batch']} {row['baseline']}",flush=True)
            if len(completed)%10==0:atomic_json(out/'manifest.json',manifest)
    for row in plan:
        if row['treatment']!='timing':
            record=reprice(row,c,out,code_hash,traces);row['status']=record['status'];row['trace_sha256']=record['trace_sha256'];completed[row['plan_id']]=record
    rows=[];units=[]
    for row in plan:
        record=completed[row['plan_id']]
        rows.append({**{k:v for k,v in row.items() if k!='overrides'},**record['summary'],
                     'temporal_run_id':record.get('temporal_reference',row['plan_id']),
                     'trace_sha256':record['trace_sha256']})
        u=pd.read_parquet(out/'runs'/row['plan_id']/'cohorts.parquet')
        for k in ['plan_id','family','model','mode','batch','seed','baseline','axis','value']:u[k]=row[k]
        units.append(u)
    raw=pd.DataFrame(rows);raw.to_csv(out/'raw_runs.csv',index=False)
    cohort=pd.concat(units,ignore_index=True);cohort.to_parquet(out/'cohort_metrics.parquet',index=False,compression='zstd')
    energy_cols=[k for k in raw if k.startswith('energy_')]
    raw[['plan_id','model','mode','batch','seed','baseline','tokens',*energy_cols]].to_csv(out/'energy_breakdown.csv',index=False)
    manifest['experiments_complete']=True;manifest['complete']=validation['all_acceptance_passed']
    manifest['status']='experiments_complete' if manifest['complete'] else 'experiments_complete_with_numeric_failure'
    atomic_json(out/'manifest.json',manifest)
    from .report import build_report
    build_report(out,c,raw,cohort,manifest)
    print('All planned experiment records resolved. See manifest and simulation_report.docx.',flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
