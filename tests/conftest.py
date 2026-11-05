import copy
import pytest
from vdie_moe_sim.config import load_spec,capacity

@pytest.fixture
def config():
    return load_spec()

@pytest.fixture
def tiny(config):
    c=copy.deepcopy(config)
    for m in c['models'].values():
        m['d_model']=64;m['d_ff']=32
        if m['shared_experts']:m['shared_d_ff']=32
        m['expert_matrix_params']=3*64*32
        m['expert_params_including_bias']=3*64*32+(2*32+64 if m['expert_linear_bias'] else 0)
        m.update(capacity(m,c))
    return c
