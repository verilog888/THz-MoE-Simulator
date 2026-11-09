"""One command: frozen inputs, validation, every run, paired statistics, report."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
from functools import lru_cache
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from zoneinfo import ZoneInfo

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')

from .config import ROOT,read_spec,derive,resolve,stable_hash,digest,placement_hash,capacity
from .io import write_json,write_gzip,read_gzip,write_csv
from .plan import plan
from .simulator import Simulator
from .validate import validate


def timestamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def engine_hash():
    return stable_hash({name:digest(Path(__file__).with_name(name+'.py')) for name in ['config','kernel','events','network','simulator']})


@lru_cache(maxsize=36)
def load_routes(path):
    with gzip.open(path,'rt',encoding='utf-8') as handle:return [json.loads(line) for line in handle]


def execute(c,row,out,trace,input_hash,source_hash):
    started=time.monotonic();out=Path(out);cc=resolve(c,row['overrides']);model=next(m for m in cc['models'] if m['id']==row['model'])
    fingerprint=stable_hash(dict(config=cc,row={k:row[k] for k in ['measurement','model','phase','B','profile','seed','method']},input=input_hash,route=trace['route_hash'],engine=source_hash))
    target=out/'runs'/(row['run_id']+'.json.gz')
    if target.exists():
        old=read_gzip(target)
        if old.get('fingerprint')==fingerprint:return dict(run_id=row['run_id'],status='completed',failure_reason='',cached=True,wall_seconds=0)
    records=load_routes(str(out/'routes'/trace['file']));cohorts=[]
    for start in range(0,cc['workload']['requests_per_seed'],row['B']):
        selected=[r for r in records if start<=r['request_rank']<start+row['B']]
        cohorts.append(Simulator(cc,model,row['method'],row['measurement'],row['phase'],selected,start//row['B']).run())
    result=dict(run=row,fingerprint=fingerprint,engine_hash=source_hash,input_hash=input_hash,route_hash=trace['route_hash'],
                route_file_sha256=trace['sha256'],placement_hash=placement_hash(cc,model),config_hash=stable_hash(cc),
                cohorts=cohorts)
    assert sum(a['tokens'] for a in cohorts)==trace['tokens']
    write_gzip(target,result)
    return dict(run_id=row['run_id'],status='completed',failure_reason='',cached=False,wall_seconds=round(time.monotonic()-started,6))


def experiment(c,out,workers):
    out=Path(out);manifest=json.loads((out/'input_manifest.json').read_text());traces=json.loads((out/'route_manifest.json').read_text())
    catalog={(t['model'],t['phase'],t['profile'],t['seed']):t for t in traces}
    source_hash=engine_hash();rows=plan(c);index={r['run_id']:r for r in rows};start=time.monotonic();started_at=timestamp()
    for r in rows:r.update(cached=False,wall_seconds=0)
    write_csv(out/'run_manifest.csv',rows)
    unique=[r for r in rows if not r['reference']];completed=0;lastlog=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(execute,c,r,str(out),catalog[r['model'],r['phase'],r['profile'],r['seed']],manifest['input_hash'],source_hash):r for r in unique}
        for future in as_completed(futures):
            row=futures[future]
            try:row.update(future.result())
            except Exception as exc:
                row.update(status='failed',failure_reason=f'{type(exc).__name__}: {exc}')
                (out/'failures').mkdir(exist_ok=True)
                (out/'failures'/(row['run_id']+'.txt')).write_text(traceback.format_exc())
            completed+=1
            if time.monotonic()-lastlog>20 or completed==len(unique):
                write_csv(out/'run_manifest.csv',rows);lastlog=time.monotonic()
                print(f'{timestamp()} completed {completed}/{len(unique)} physical runs; failed {sum(r["status"]=="failed" for r in rows)}; elapsed {time.monotonic()-start:.1f}s',flush=True)
    for row in rows:
        if row['reference']:
            ref=index[row['reference']];row.update(status=ref['status'],failure_reason=ref['failure_reason'],cached=True)
    write_csv(out/'run_manifest.csv',rows)
    write_json(out/'execution.json',dict(started_at=started_at,finished_at=timestamp(),wall_seconds=time.monotonic()-start,workers=workers,
                                       physical_runs=len(unique),plan_records=len(rows),engine_hash=source_hash,
                                       completed=sum(r['status']=='completed' for r in rows),failed=sum(r['status']=='failed' for r in rows)))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',default=str(ROOT/'ToV_MoE_MINIMAL_SIM_SPEC.md'))
    parser.add_argument('--output',default=str(ROOT/'Results'))
    parser.add_argument('--workers',type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    args=parser.parse_args();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True);c=read_spec(args.spec)
    versions={name:importlib.metadata.version(name) for name in ['numpy','matplotlib','huggingface_hub','transformers','tokenizers','sentencepiece','protobuf','python-docx']}
    versions['python']=sys.version
    write_json(out/'config_resolved.json',dict(configuration=c,derived=derive(c),capacity=[r for m in c['models'] for r in capacity(c,m)],software=versions,
                                               specification_sha256=digest(args.spec),engine_hash=engine_hash(),routing_source='controlled_synthetic'))
    if not args.summarize_only:
        from .workload import prepare,freeze_routes
        manifest=prepare(c,out);freeze_routes(c,manifest,out)
        print(f'{timestamp()} official tokenizers and 36 controlled routing traces verified',flush=True)
        if args.prepare_only:return
        validation=validate(c);write_json(out/'validation.json',validation)
        print(f'{timestamp()} {len(validation["checks"])} preflight validation groups passed',flush=True)
        if args.validate_only:return
        experiment(c,out,args.workers)
    from .metrics import summarize
    from .audit import full_audit
    full_audit(c,out)
    summarize(c,out)


if __name__=='__main__':main()
