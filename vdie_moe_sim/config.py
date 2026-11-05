"""Parse the unique specification YAML and verify its derived constants."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]

def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def load_spec(path=None):
    path = Path(path or ROOT / 'VDIE_MOE_SIM_SPEC.md')
    ticks=chr(96)*3
    blocks = re.findall(r'^'+ticks+r'yaml\n(.*?)^'+ticks, path.read_text(), re.M | re.S)
    if len(blocks) != 1:
        raise ValueError('Specification must contain exactly one YAML block')
    c = yaml.safe_load(blocks[0])
    validate(c)
    return c

def matrix_storage(n, k, c):
    bn, bk = c['precision']['weight_block']
    return n*k + c['precision']['scale_bytes'] * math.ceil(n/bn) * math.ceil(k/bk)

def vector_bytes(n, c):
    return n + c['precision']['scale_bytes'] * math.ceil(n/c['precision']['activation_block'])

def capacity(m, c):
    d, h, e = m['d_model'], m['d_ff'], m['routed_experts']
    expert = 2*matrix_storage(h,d,c)+matrix_storage(d,h,c)
    if m['expert_linear_bias']:
        expert += 2*vector_bytes(h,c)+vector_bytes(d,c)
    router = matrix_storage(e,d,c)
    if m['router_linear_bias']:
        router += vector_bytes(e,c)
    if m['router_correction_fp32']:
        router += 4*e
    resident = (e//c['hardware']['vdies'] + int(m['shared_experts'] > 0))*expert + router + c['hardware']['workspace_bytes']
    return dict(expert_storage_bytes=expert, router_storage_bytes=router,
                resident_bytes_per_die=resident)

def validate(c):
    h = c['hardware']
    if sum(h['buffer_partition_bytes']) != h['workspace_bytes']:
        raise ValueError('Workspace partitions do not sum to workspace size')
    if h['workspace_bytes'] >= h['edram_bytes']:
        raise ValueError('Workspace exceeds eDRAM')
    if h['ops_per_mac'] != 2 or h['clock_hz'] != 200000000:
        raise ValueError('This implementation requires the approved clock and OP convention')
    for mid,m in c['models'].items():
        for k,v in capacity(m,c).items():
            if m[k] != v:
                raise ValueError(f'{mid}.{k}: expected {v}, got {m[k]}')
        if m['resident_bytes_per_die'] > h['edram_bytes']:
            raise ValueError(f'{mid}: capacity exceeded')
        if m['expert_matrix_params'] != 3*m['d_model']*m['d_ff']:
            raise ValueError('Wrong expert parameter count')
        if m['routed_experts'] % h['vdies'] or m['top_k'] > m['routed_experts']:
            raise ValueError('Invalid expert placement')
    if not math.isclose(c['energy']['compute_pj_op'], (1/21.4)*(1.1/.9)**2, rel_tol=1e-12):
        raise ValueError('Wrong energy calibration')
    if not math.isclose(c['energy']['ofdma_extra_w_per_active_group'], .347*132/112, rel_tol=1e-12):
        raise ValueError('Wrong baseband calibration')
    return True

def override(c, values):
    c = copy.deepcopy(c)
    for key,value in values.items():
        node = c
        parts = key.split('.')
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise ValueError(f'Unknown configuration key: {key}')
        node[parts[-1]] = value
    return c
