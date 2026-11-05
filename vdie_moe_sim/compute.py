import math

def expected_matrix_ops(model,tokens):
    d,h,e,k,s=[model[x] for x in ['d_model','d_ff','routed_experts','top_k','shared_experts']]
    return tokens*(2*d*e+6*d*h*(k+s))

def tile_ps(m,n,k,utilization=1.):
    return math.ceil(2*m*n*k/(5000*utilization))*5000
