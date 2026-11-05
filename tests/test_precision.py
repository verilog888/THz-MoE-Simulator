import numpy as np
import pytest
import ml_dtypes
from vdie_moe_sim.precision import POSITIVE,encode_e4m3fn,decode_e4m3fn,quantize,activation,numerical_validation

def test_exhaustive_fp8_and_rounding_against_independent_library():
    codes=np.arange(256,dtype=np.uint8)
    assert np.allclose(decode_e4m3fn(codes),codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32),equal_nan=True)
    rng=np.random.default_rng(20261104)
    mid=(POSITIVE[1:]+POSITIVE[:-1])/2
    values=np.concatenate([rng.uniform(-448,448,100000),POSITIVE,-POSITIVE,mid,-mid]).astype(np.float32)
    assert np.array_equal(encode_e4m3fn(values),values.astype(ml_dtypes.float8_e4m3fn).view(np.uint8))
    assert np.array_equal(decode_e4m3fn(encode_e4m3fn([1e5,-1e5])),[448,-448])
    assert POSITIVE[1]==2**-9

def test_block_scales_zero_and_ragged_edges():
    x=np.zeros((129,129),np.float32);x[0,0]=448;x[128,128]=224
    q,s=quantize(x,weight=True)
    assert np.array_equal(q,x)
    assert np.array_equal(s,[1,1,1,.5])

def test_gpt_oss_special_activation_and_clipping(config):
    g=np.array([8.,-8.],np.float32);u=np.array([-9.,9.],np.float32)
    expected=np.array([7.,-8.])/(1+np.exp(-1.702*np.array([7.,-8.])))*(np.array([-7.,7.])+1)
    assert np.allclose(activation(g,u,config['models']['M1']),expected,rtol=1e-6)
    assert not np.allclose(activation(g,u,config['models']['M2']),expected)

def test_shared_fused_split_mathematical_equivalence(config):
    rng=np.random.default_rng(20261104);x=rng.normal(size=(16,64)).astype(np.float32)
    gate=(rng.normal(size=(64,64))/8).astype(np.float32);up=(rng.normal(size=(64,64))/8).astype(np.float32)
    down=(rng.normal(size=(64,64))/8).astype(np.float32);m={'activation':'silu_glu'}
    full=activation(x@gate.T,x@up.T,m)@down.T
    split=sum(activation(x@gate[b:b+32].T,x@up[b:b+32].T,m)@down[:,b:b+32].T for b in [0,32])
    assert np.allclose(full,split,rtol=config['simulation']['fp32_rtol'],atol=config['simulation']['fp32_atol'])

def test_fp32_local_combine_equivalence(config):
    assert all(r['fp32_equivalence'] for r in numerical_validation(config))

@pytest.mark.numeric_gate
@pytest.mark.parametrize('index',range(8))
def test_approved_end_to_end_fp8_threshold(config,index):
    r=numerical_validation(config)[index]
    assert r['nrmse']<=config['simulation']['fp8_nrmse_limit'],r
    assert r['cosine']>=config['simulation']['fp8_cosine_min'],r
