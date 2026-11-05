"""Pinned public files, official tokenization, and immutable synthetic route traces."""
import csv
import gzip
import io
import json
from pathlib import Path
import urllib.request
import numpy as np
from tokenizers import Tokenizer
from .config import ROOT, digest_file, stable_digest
from .routing import route

def download(url, target):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'VDie-MoE-Sim/1.0'}), timeout=60) as r:
        data = r.read()
    temp = target.with_suffix(target.suffix + '.tmp')
    temp.write_bytes(data); temp.replace(target)

def prepare_data(c, out):
    out = Path(out); cache = ROOT / '.cache' / 'frozen_inputs'
    lock = json.loads((ROOT/'docs/approval/source_lock.json').read_text())
    ds = next(x for x in lock['sources'] if x['label']=='dataset')
    csv_path = cache / 'dataset' / ds['revision'] / 'train.csv'
    download(f"https://huggingface.co/datasets/{ds['repo']}/resolve/{ds['revision']}/train.csv", csv_path)
    if digest_file(csv_path) != lock['dataset_check']['sha256']:
        raise ValueError('Frozen dataset hash mismatch')
    raw = list(csv.DictReader(io.StringIO(csv_path.read_text(encoding='utf-8-sig'))))
    clean=[]; seen=set()
    for row_id, r in enumerate(raw):
        p,a = r['human_prompt'],r['chatgpt_response']
        if not p.strip() or not a.strip() or p in seen:
            continue
        seen.add(p); clean.append((row_id,p,a))
    if len(clean) < c['workload']['requests_per_seed']:
        raise ValueError('Not enough distinct requests')
    encoded = {}
    input_manifest = dict(dataset=ds, raw_rows=len(raw), clean_rows=len(clean),
                          dataset_sha256=digest_file(csv_path), models={}, samples={})
    for model_id in c['models']:
        s = next(x for x in lock['sources'] if x['label']==model_id)
        base = cache / model_id / s['revision']
        for name in ['config.json','tokenizer.json','tokenizer_config.json']:
            download(f"https://huggingface.co/{s['repo']}/resolve/{s['revision']}/{name}",base/name)
        if digest_file(base/'config.json') != s['config_sha256']:
            raise ValueError('Official configuration hash mismatch')
        tok = Tokenizer.from_file(str(base/'tokenizer.json'))
        prompts = tok.encode_batch([r[1] for r in clean], add_special_tokens=False)
        answers = tok.encode_batch([r[2] for r in clean], add_special_tokens=False)
        encoded[model_id] = [(r[0], p.ids, a.ids) for r,p,a in zip(clean,prompts,answers)]
        input_manifest['models'][model_id] = {
            'revision':s['revision'],'repo':s['repo'],
            'files':{p.name:digest_file(p) for p in base.iterdir() if p.is_file()},
            'prompt_lengths':[len(p.ids) for p in prompts],
            'response_lengths':[len(a.ids) for a in answers],
            'add_special_tokens':False,'chat_template':False}
    for seed in c['simulation']['seed_list']:
        indices = np.random.Generator(np.random.PCG64(seed)).choice(
            len(clean), c['workload']['requests_per_seed'], replace=False)
        input_manifest['samples'][str(seed)] = [clean[i][0] for i in indices]
    out.mkdir(parents=True,exist_ok=True)
    (out/'input_manifest.json').write_text(json.dumps(input_manifest,indent=2)+'\n')
    return encoded, input_manifest

def trace_name(mid, seed, mode, profile, reuse):
    return f'{mid}-{seed}-{mode}-{profile}-reuse{reuse:g}'

def prepare_traces(c, out, encoded, inputs):
    out=Path(out); target=out/'traces';target.mkdir(exist_ok=True)
    index={}
    for mid,m in c['models'].items():
        by_id={r[0]:r for r in encoded[mid]}
        for seed in c['simulation']['seed_list']:
            for mode in ['prefill','decode']:
                for profile,reuse in [('uniform',0.),('zipf',0.)]+([('uniform',.8)] if mode=='decode' else []):
                    name=trace_name(mid,seed,mode,profile,reuse)
                    path=target/f'{name}.jsonl.gz'; binary=target/f'{name}.npz'
                    desc=[]; records=[]; selected=inputs['samples'][str(seed)]
                    for request,row_id in enumerate(selected):
                        tokens=by_id[row_id][1 if mode=='prefill' else 2]
                        cap=c['workload']['prefill_length_cap'] if mode=='prefill' else c['workload']['decode_steps_cap']
                        previous=None
                        for position, token_id in enumerate(tokens[:cap]):
                            ids,weights=route(seed,mid,row_id,mode,position,m,profile,c['workload']['zipf_alpha'],reuse,previous)
                            previous=(ids,weights)
                            desc.append([request,position,row_id,token_id,*ids])
                            records.append(dict(row_id=row_id,request_ordinal=request,
                                                token_uid=f'{row_id}:{mode}:{position}',mode=mode,step=position,
                                                token_id=token_id,d_model=m['d_model'],expert_ids=ids,gate_weights=weights,
                                                payload_hash=stable_digest([seed,mid,row_id,mode,position,'synthetic_activation']),
                                                input_trace_mode='dataset_tokens_synthetic_router'))
                    with path.open('wb') as f:
                        with gzip.GzipFile(fileobj=f,mode='wb',mtime=0,filename='') as g:
                            for r in records:g.write((json.dumps(r,sort_keys=True,separators=(',',':'))+'\n').encode())
                    np.savez_compressed(binary,descriptors=np.asarray(desc,dtype=np.int32))
                    index[name]={'trace_sha256':digest_file(path),'binary_sha256':digest_file(binary),
                                 'tokens':len(desc),'requests':len(selected)}
                    print(f'Frozen trace {name}: {len(desc)} tokens',flush=True)
    (target/'index.json').write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
    return index
