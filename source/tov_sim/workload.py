"""Official tokenizers and deterministic, explicitly synthetic routing."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('USE_TORCH','0')
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
import numpy as np

from .config import ROOT,digest,stable_hash
from .io import write_json


def seed64(text):return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest()[:8],'little')


def freeze_sources(c,out):
    """Replay verified local files, or download only an explicit small-file list."""
    folder=Path(out)/'inputs';lock=folder/'source_lock.json'
    if lock.exists():
        sources=json.loads(lock.read_text())
        for source in sources:
            for name,sha in source['sha256'].items():
                if digest(folder/source['id']/name)!=sha:raise ValueError('Frozen input checksum mismatch: '+source['id']+'/'+name)
        return sources
    delivered=ROOT/'Results/inputs'
    if (delivered/'source_lock.json').exists() and delivered.resolve()!=folder.resolve():
        shutil.copytree(delivered,folder,dirs_exist_ok=True)
        return freeze_sources(c,out)
    from huggingface_hub import HfApi,hf_hub_download
    api=HfApi();sources=[]
    allowed={'config.json','tokenizer.json','tokenizer_config.json','tokenizer.model','special_tokens_map.json','added_tokens.json','vocab.json','merges.txt'}
    for model in c['models']:
        meta=api.model_info(model['repo'],files_metadata=True)
        files=sorted(s.rfilename for s in meta.siblings if s.rfilename in allowed)
        sources.append(dict(id=model['id'],repo=model['repo'],revision=meta.sha,files=files,repo_type='model'))
    meta=api.dataset_info(c['workload']['dataset'])
    sources.append(dict(id='dataset',repo=c['workload']['dataset'],revision=meta.sha,files=['train.csv'],repo_type='dataset'))
    for s in sources:
        s['sha256']={}
        for name in s['files']:
            path=hf_hub_download(s['repo'],name,repo_type=s['repo_type'],revision=s['revision'],local_dir=folder/s['id'],cache_dir=ROOT/'.cache/tov_hf')
            s['sha256'][name]=digest(path)
    write_json(lock,sources);return sources


def check_model(model,config):
    mid=model['id']
    if mid=='gpt_oss_20b':observed=dict(d=config['hidden_size'],h=config['intermediate_size'],E=config['num_local_experts'],k=config['num_experts_per_tok'],shared=0)
    elif mid=='qwen3_30b_a3b':observed=dict(d=config['hidden_size'],h=config['moe_intermediate_size'],E=config['num_experts'],k=config['num_experts_per_tok'],shared=0)
    else:observed=dict(d=config['hidden_size'],h=config['moe_intermediate_size'],E=config['moe_num_experts'],k=config['moe_k'],shared=config['moe_num_shared_experts'])
    for key,value in observed.items():
        if value!=model[key]:raise ValueError(f'{mid}.{key}: specification {model[key]}, official config {value}')
    assert 0<=model['layer_index']<config['num_hidden_layers']
    if mid=='ernie_21b_a3b':assert model['layer_index']>=config['moe_layer_start_index']
    return observed


def prepare(c,out):
    from transformers import AutoTokenizer
    out=Path(out);sources=freeze_sources(c,out);byid={s['id']:s for s in sources};folder=out/'inputs'
    with (folder/'dataset/train.csv').open(encoding='utf-8-sig',newline='') as f:original=list(csv.DictReader(f))
    rows=[];seen=set()
    for row_id,row in enumerate(original):
        prompt,response=row['human_prompt'],row['chatgpt_response']
        if not prompt.strip() or not response.strip() or prompt in seen:continue
        seen.add(prompt);rows.append((row_id,prompt,response))
    w=c['workload'];assert len(rows)>=w['requests_per_seed']
    samples={str(seed):[int(i) for i in np.random.Generator(np.random.PCG64(seed)).choice(len(rows),w['requests_per_seed'],replace=False)] for seed in w['seeds']}
    lengths=[];tokenizers={};structure={}
    for model in c['models']:
        mid=model['id'];path=folder/mid
        structure[mid]=check_model(model,json.loads((path/'config.json').read_text()))
        tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        tokenizers[mid]=dict(class_name=type(tokenizer).__name__,commit=byid[mid]['revision'],
                             add_special_tokens=False,chat_template_applied=False,files=byid[mid]['sha256'])
        encoded={}
        for seed,indices in samples.items():
            for rank,index in enumerate(indices):
                row_id,prompt,response=rows[index]
                if row_id not in encoded:encoded[row_id]=(tokenizer.encode(prompt,add_special_tokens=False),tokenizer.encode(response,add_special_tokens=False))
                pt,rt=encoded[row_id];assert len(pt)>0 and len(rt)>0
                lengths.append(dict(model=mid,seed=int(seed),request_rank=rank,row_id=row_id,
                                    prompt_tokens=len(pt),response_tokens=len(rt),
                                    prefill_tokens=min(len(pt),w['prefill_token_cap']),decode_tokens=min(len(rt),w['decode_step_cap']),
                                    prompt_token_ids_sha256=stable_hash(pt),response_token_ids_sha256=stable_hash(rt)))
    manifest=dict(dataset=byid['dataset'],original_rows=len(original),clean_rows=len(rows),
                  selected_rows={s:[rows[i][0] for i in ids] for s,ids in samples.items()},
                  tokenizers=tokenizers,verified_model_structure=structure,requests=lengths,
                  routing_source='controlled_synthetic_topk_not_pretrained_router',routing_semantics=dict(
                      rng='NumPy PCG64',derivation='SHA256 UTF8 first 8 bytes little-endian',
                      hot_shift_direction='positive numpy.roll of the fixed ranked expert permutation',
                      permutation_key='seed|model_id|perm',token_key='seed|model_id|phase|request_rank|token_index|route'))
    manifest['input_hash']=stable_hash(manifest);write_json(out/'input_manifest.json',manifest)
    return manifest


def routes_for(c,m,phase,profile,seed,requests):
    w=c['workload'];e,k=m['E'],m['k'];perm=np.random.Generator(np.random.PCG64(seed64(f'{seed}|{m["id"]}|perm'))).permutation(e)
    weights=(np.arange(e,dtype=np.float64)+1)**(-w['zipf_alpha']);weights/=weights.sum()
    distributions={};records=[];ordinal=0
    for request in sorted(requests,key=lambda r:r['request_rank']):
        for index in range(request['prefill_tokens' if phase=='prefill' else 'decode_tokens']):
            epoch=ordinal//w['prefill_hot_epoch_tokens'] if phase=='prefill' else index//w['decode_hot_epoch_steps']
            if profile=='uniform':p=np.full(e,1/e,dtype=np.float64)
            else:
                epoch%=4
                if epoch not in distributions:
                    p=np.empty(e,dtype=np.float64);p[np.roll(perm,epoch*(e//4))]=weights;distributions[epoch]=p
                p=distributions[epoch]
            key=f'{seed}|{m["id"]}|{phase}|{request["request_rank"]}|{index}|route'
            ids=np.random.Generator(np.random.PCG64(seed64(key))).choice(e,k,replace=False,p=p).tolist()
            selected=p[ids];selected=selected/selected.sum()
            records.append(dict(model_id=m['id'],seed=seed,profile=profile,phase=phase,request_id=request['row_id'],
                                request_rank=request['request_rank'],token_index=index,step_index=index if phase=='decode' else -1,
                                expert_ids=ids,gate_weights=selected.tolist(),shared_ids=list(range(m['shared'])),
                                routing_source='controlled_synthetic'))
            ordinal+=1
    return records


def freeze_routes(c,manifest,out):
    out=Path(out);folder=out/'routes';folder.mkdir(exist_ok=True);catalog=[]
    for m in c['models']:
        for seed in c['workload']['seeds']:
            requests=[r for r in manifest['requests'] if r['model']==m['id'] and r['seed']==seed]
            for phase in c['workload']['phases']:
                for profile in c['workload']['routing_profiles']:
                    records=routes_for(c,m,phase,profile,seed,requests)
                    name=f'{m["id"]}_{phase}_{profile}_{seed}.jsonl.gz'
                    with (folder/name).open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as zipped:
                        for record in records:zipped.write((json.dumps(record,separators=(',',':'),ensure_ascii=False)+'\n').encode())
                    catalog.append(dict(model=m['id'],phase=phase,profile=profile,seed=seed,file=name,sha256=digest(folder/name),tokens=len(records),route_hash=stable_hash(records)))
    write_json(out/'route_manifest.json',catalog);return catalog
