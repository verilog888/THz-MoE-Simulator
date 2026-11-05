import hashlib
import json
import numpy as np

def rng_for(parts):
    raw = json.dumps(parts, separators=(',', ':'), ensure_ascii=False).encode()
    return np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(raw).digest(), 'big')))

def route(seed, model_id, row_id, mode, position, model, profile='uniform',
          alpha=1.2, reuse_probability=0., previous=None):
    if mode == 'decode' and position and previous is not None:
        if rng_for([seed, model_id, row_id, 'decode_reuse', position]).random() < reuse_probability:
            return previous
    e = model['routed_experts']
    p = np.full(e, 1/e, dtype=np.float64)
    if profile == 'zipf':
        order = rng_for([seed, model_id, 'expert_rank']).permutation(e)
        p[order] = np.arange(1,e+1,dtype=float)**(-alpha)
        p /= p.sum()
    elif profile != 'uniform':
        raise ValueError(profile)
    rng = rng_for([seed, model_id, row_id, mode, position])
    u = rng.random(e)
    while np.any(u == 0):
        zero = u == 0
        u[zero] = rng.random(zero.sum())
    z = np.log(p)-np.log(-np.log(u))
    ids = np.lexsort((np.arange(e), -z))[:model['top_k']]
    w = np.exp(z[ids]-z[ids].max()); w /= w.sum()
    return ids.tolist(), w.tolist()
