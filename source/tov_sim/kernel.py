"""Operator-level roofline model; no expert weights or isolated-latency cache."""
from functools import lru_cache
import math
from .config import derive,expert_bytes,router_bytes,qbytes


class Kernel:
    def __init__(self,c,m):
        self.c=c;self.m=m;self.d=derive(c);self.start=c['hardware']['kernel_startup_ns']*1000

    def matrix(self,kind,batch):
        m=self.m;d,h,e,k=m['d'],m['h'],m['E'],m['k']
        if kind=='router':
            mac=d*e*batch;hb=router_bytes(self.c,m)+batch*(qbytes(self.c,d)+4*e)
            scalar=batch*(k*(e-1)+8*e+(e if m['router_bias'] else 0))
        else:
            mac=3*d*h*batch;hb=expert_bytes(self.c,m)+batch*(qbytes(self.c,d)+4*d)
            scalar=batch*(8*h+d+(2*h+d if m['expert_bias'] else 0))
        time=self.start+math.ceil((max(2*mac/self.d['effective_ops_per_second'],hb/self.d['kernel_bytes_per_second'])+scalar/self.d['scalar_ops_per_second'])*1e12)
        return time,mac,hb,scalar

    def vector(self,read,write,ops):
        time=self.start+math.ceil(max((read+write)/self.d['kernel_bytes_per_second'],ops/self.d['scalar_ops_per_second'])*1e12)
        return time,0,read+write,ops
