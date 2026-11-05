import math

def burst_ps(size,bandwidth_scale=1.):
    return 5000+math.ceil(size/(1024*bandwidth_scale))*5000
