"""IEEE-style E4M3FN encoding (not integer quantization), block scales, and oracles."""
import numpy as np

def _positive_values():
    codes=np.arange(127,dtype=np.int32);exponent=codes//8;mantissa=codes%8
    return np.where(exponent==0,mantissa*2.**-9,
                    (1+mantissa/8)*2.**(exponent-7)).astype(np.float32)

POSITIVE=_positive_values()

def encode_e4m3fn(values):
    x=np.asarray(values,dtype=np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError('Nonfinite input to FP8')
    a=np.minimum(np.abs(x),448)
    hi=np.minimum(np.searchsorted(POSITIVE,a),126);lo=np.maximum(hi-1,0)
    dlo=a-POSITIVE[lo];dhi=POSITIVE[hi]-a
    choose_hi=(dhi<dlo)|((dhi==dlo)&((hi%2)==0))
    code=np.where(choose_hi,hi,lo).astype(np.uint8)
    return code|(np.signbit(x).astype(np.uint8)*128)

def decode_e4m3fn(codes):
    codes=np.asarray(codes,dtype=np.uint8)
    table=np.append(POSITIVE,np.float32(np.nan))
    return np.where(codes&128,-1,1).astype(np.float32)*table[codes&127]

def quantize(values,weight=False):
    x=np.asarray(values,dtype=np.float32);y=np.empty_like(x);scales=[]
    if x.ndim==1:
        for start in range(0,x.size,128):
            v=x[start:start+128];scale=float(np.max(np.abs(v)))/448 or 1.
            y[start:start+128]=decode_e4m3fn(encode_e4m3fn(v/scale))*scale;scales.append(scale)
    elif weight:
        for i in range(0,x.shape[0],128):
            for j in range(0,x.shape[1],128):
                v=x[i:i+128,j:j+128];scale=float(np.max(np.abs(v)))/448 or 1.
                y[i:i+128,j:j+128]=decode_e4m3fn(encode_e4m3fn(v/scale))*scale;scales.append(scale)
    else:
        for i in range(x.shape[0]):
            y[i],s=quantize(x[i]);scales.extend(s)
    return y,np.asarray(scales,dtype=np.float32)

def activation(gate,up,model):
    if model['activation']=='gpt_oss_swiglu':
        gate=np.minimum(gate,model['swiglu_limit'])
        up=np.clip(up,-model['swiglu_limit'],model['swiglu_limit'])
        return gate/(1+np.exp(-model['swiglu_alpha']*gate))*(up+1)
    return gate/(1+np.exp(-gate))*up

def numerical_validation(c):
    from .placement import contributions
    from .routing import route
    rows=[];n=c['simulation']['fp8_numeric_test_tokens'];d=c['simulation']['numeric_test_d'];h=c['simulation']['numeric_test_h']
    seed=c['simulation']['seed_list'][0]
    for mid,m in c['models'].items():
        for nonzero_bias in ([False,True] if m['expert_linear_bias'] else [False]):
            rng=np.random.Generator(np.random.PCG64(seed))
            x=rng.normal(size=(n,d)).astype(np.float32)
            experts=[]
            for e in range(m['routed_experts']+m['shared_experts']):
                w=[(rng.normal(size=shape)/np.sqrt(d)).astype(np.float32) for shape in [(h,d),(h,d),(d,h)]]
                b=[(rng.normal(size=s)/np.sqrt(d)).astype(np.float32) if nonzero_bias else np.zeros(s,np.float32) for s in [h,h,d]]
                experts.append((w,b))
            outputs={False:[],True:[]};references=[]
            exact={False:[],True:[]}
            for t in range(n):
                ids,weights=route(seed,mid,t,'prefill',t,m);weights=dict(zip(ids,weights))
                groups=contributions(m,t%16,ids);unquant={};quant={}
                for e in [e for es in groups.values() for e in es]:
                    w,b=experts[e];g=x[t]@w[0].T+b[0];u=x[t]@w[1].T+b[1]
                    y=activation(g,u,m)@w[2].T+b[2]
                    coefficient=weights.get(e,1.)
                    unquant[e]=(y*coefficient).astype(np.float32)
                    qw=[quantize(z,weight=True)[0] for z in w]
                    qb=[quantize(z)[0] for z in b]
                    qx=quantize(x[t])[0]
                    qh=quantize(activation(qx@qw[0].T+qb[0],qx@qw[1].T+qb[1],m))[0]
                    quant[e]=((qh@qw[2].T+qb[2])*coefficient).astype(np.float32)
                reference=np.sum(np.stack([unquant[e] for e in sorted(unquant)]),axis=0,dtype=np.float32)
                references.append(reference)
                for lc in [False,True]:
                    parts=[];plain=[]
                    for die,es in sorted(groups.items()):
                        if lc and die!=t%16:
                            local=np.sum(np.stack([quant[e] for e in es]),axis=0,dtype=np.float32)
                            parts.append(quantize(local)[0])
                            plain.append(np.sum(np.stack([unquant[e] for e in es]),axis=0,dtype=np.float32))
                        else:
                            for e in es:
                                parts.append(quant[e] if die==t%16 else quantize(quant[e])[0])
                                plain.append(unquant[e])
                    exact[lc].append(np.sum(np.stack(plain),axis=0,dtype=np.float32))
                    outputs[lc].append(quantize(np.sum(np.stack(parts),axis=0,dtype=np.float32))[0])
            ref=np.asarray(references,dtype=np.float32).ravel()
            for lc in [False,True]:
                no_link=np.asarray(exact[lc])
                mathematical=bool(np.allclose(no_link,np.asarray(references),rtol=c['simulation']['fp32_rtol'],atol=c['simulation']['fp32_atol']))
                actual=np.asarray(outputs[lc],dtype=np.float32).ravel()
                error=float(np.linalg.norm(actual-ref)/max(np.linalg.norm(ref),1e-12))
                cosine=float(np.dot(actual,ref)/(np.linalg.norm(actual)*np.linalg.norm(ref)))
                rows.append(dict(model=mid,nonzero_bias=nonzero_bias,local_combine=lc,
                                 fp32_equivalence=mathematical,nrmse=error,cosine=cosine,
                                 passed=mathematical and error<=c['simulation']['fp8_nrmse_limit'] and cosine>=c['simulation']['fp8_cosine_min']))
    return rows
