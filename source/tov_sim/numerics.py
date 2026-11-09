"""Finite E4M3FN reference quantization, NumPy only (never full model weights)."""
import numpy as np


def fp8_table():
    codes=np.arange(127,dtype=np.int32);exponent=codes>>3;mantissa=codes&7
    values=np.where(exponent==0,mantissa/8*2.**-6,(1+mantissa/8)*2.**(exponent-7))
    return values.astype(np.float32),codes


LEVELS,CODES=fp8_table()


def e4m3fn(values):
    """Saturating nearest finite value; even low significand bit wins exact ties."""
    a=np.asarray(values,dtype=np.float32);x=np.minimum(np.abs(a),448.)
    hi=np.minimum(np.searchsorted(LEVELS,x,side='left'),len(LEVELS)-1);lo=np.maximum(hi-1,0)
    dl=x-LEVELS[lo];dh=LEVELS[hi]-x
    choice=np.where((dl<dh)|((dl==dh)&((CODES[lo]&1)==0)),lo,hi)
    return np.copysign(LEVELS[choice],a)


def quantize_vector(values,block=128):
    a=np.asarray(values,dtype=np.float32);out=np.empty_like(a);flat=a.reshape(-1);result=out.reshape(-1);scales=[]
    for start in range(0,flat.size,block):
        part=flat[start:start+block];maximum=float(np.max(np.abs(part)));scale=np.float32(maximum/448 if maximum else 1.)
        result[start:start+block]=e4m3fn(part/scale)*scale;scales.append(float(scale))
    return out,scales


def quantize_weight(values,block=128):
    a=np.asarray(values,dtype=np.float32);out=np.empty_like(a)
    for i in range(0,a.shape[0],block):
        for j in range(0,a.shape[1],block):out[i:i+block,j:j+block]=quantize_vector(a[i:i+block,j:j+block],block*block)[0]
    return out


def activation(gate,up,rule):
    if rule=='gpt_oss_swiglu':
        gate=np.minimum(gate,7);up=np.clip(up,-7,7)
        return gate/(1+np.exp(-1.702*gate))*(up+1)
    return gate/(1+np.exp(-gate))*up


def error(actual,reference):
    x=np.asarray(actual,dtype=np.float64).ravel();y=np.asarray(reference,dtype=np.float64).ravel()
    return dict(relative_l2=float(np.linalg.norm(x-y)/np.linalg.norm(y)),cosine=float(x@y/(np.linalg.norm(x)*np.linalg.norm(y))))
