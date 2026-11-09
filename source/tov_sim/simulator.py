"""A single-layer dependency graph, finite staging, and one NMP per Die."""
from collections import defaultdict,deque
from dataclasses import dataclass,field
import math
import numpy as np
from .config import BASE,qbytes,ceildiv,derive
from .events import Calendar,Ledger
from .kernel import Kernel
from .network import Network


@dataclass(slots=True)
class Token:
    id:int
    record:dict
    owner:int
    destinations:dict
    expected:int
    start:int
    got:int=0
    computed:dict=field(default_factory=dict)
    outputs:dict=field(default_factory=dict)
    incoming:list=field(default_factory=list)
    lat:int=0


class Simulator:
    def __init__(self,c,model,method,measurement,phase,records,cohort):
        self.c=c;self.m=model;self.method=method;self.replay=measurement=='replay';self.phase=phase;self.cohort=cohort
        self.n=c['hardware']['dies'];self.d=model['d'];self.q=qbytes(c,self.d);self.fp=4*self.d
        self.header=c['packet']['header_bytes']+c['packet']['crc_bytes'];self.entry=c['packet']['expert_entry_bytes']
        self.return_size=self.header+self.q+ceildiv(model['E']+model['shared'],8)
        self.lc=not method.endswith('NO_LC');self.ev=Calendar();self.ledger=Ledger(c,not method.startswith('TSV'),'OFDMA' in method)
        self.kernel=Kernel(c,model);self.network=Network(self);self.derived=derive(c)
        self.buffers=[0]*17;self.peaks=[0]*17;self.dma_until=[0]*17;self.busy=[False]*self.n
        self.engine_last=[0]*self.n;self.kick_pending=[False]*self.n
        self.nmp_by_node=[0]*self.n;self.dma_by_node=[0]*17;self.token_timings=[]
        self.routers=[deque() for _ in range(self.n)];self.combines=[deque() for _ in range(self.n)]
        self.experts=[{} for _ in range(self.n)];self.jobid=0;self.matrix_batches=defaultdict(int)
        self.active=[0]*self.n;self.pending=[deque() for _ in range(self.n)]
        self.records=sorted(records,key=lambda r:(r['request_rank'],r['token_index']))
        self.by_request=defaultdict(list)
        for r in self.records:self.by_request[r['request_rank']].append(r)
        self.completed=0;self.round_completed=0;self.round_expected=0;self.step=0;self.round_start=0
        self.rows=[];self.cohort_ledger=defaultdict(float);self.useful_bytes=0;self.token_lats=[];self.request_done={}
        self.tokens={};self.tid=0;self.done=False;self.max_active=0;self.contributor_checks=0
    def alloc(self,node,size):
        self.buffers[node]+=size;self.peaks[node]=max(self.peaks[node],self.buffers[node])
        limit=self.c['hardware']['base_staging_bytes' if node==BASE else 'staging_bytes_per_die']
        if self.buffers[node]>limit:raise RuntimeError(f'Staging overflow node={node}: {self.buffers[node]} > {limit}')
    def free(self,node,size):
        self.buffers[node]-=size
        assert self.buffers[node]>=0,(node,self.buffers[node],size)
    def dma(self,node,size,callback,*args):
        rate=self.c['hardware']['base_dma_GBps']*1e9 if node==BASE else self.derived['dma_bytes_per_second']
        start=max(self.ev.now,self.dma_until[node]);end=start+math.ceil(size/rate*1e12)
        self.dma_until[node]=end;self.ledger.count['HB_bytes']+=size;self.ledger.count['DMA_bytes']+=size
        self.ledger.count['dma_busy_ps']+=end-start;self.ev.at(end,callback,*args)
        self.dma_by_node[node]+=end-start
    def job(self,node,kind,cost,callback,*args):
        self.jobid+=1;job=(self.ev.now,self.jobid,cost,callback,args)
        (self.routers if kind=='router' else self.combines)[node].append(job);self.kick(node)
    def kick(self,node):
        if not self.busy[node] and not self.kick_pending[node]:
            self.kick_pending[node]=True;self.ev.at(self.ev.now,self._engine,node,priority=2)
    def _engine(self,node):
        self.kick_pending[node]=False
        if self.busy[node]:return
        q=self.combines[node] or self.routers[node]
        if q:
            _,_,cost,callback,args=q.popleft();self._start(node,cost,callback,args);return
        eligible=[];wait=self.c['execution']['batch_wait_ns']*1000;maximum=self.c['execution']['expert_batch_max']
        for eid,q in self.experts[node].items():
            if q and (len(q)>=maximum or q[0][0]+wait<=self.ev.now):eligible.append((q[0][0],q[0][1],eid))
        if eligible:
            _,_,eid=min(eligible);q=self.experts[node][eid];batch=[q.popleft()[2] for _ in range(min(maximum,len(q)))]
            self.matrix_batches[len(batch)]+=1
            self._start(node,self.kernel.matrix('expert',len(batch)),self._expert_done,(node,eid,batch))
    def _start(self,node,cost,callback,args):
        duration,mac,hb,scalar=cost;assert self.ev.now>=self.engine_last[node]
        self.busy[node]=True;self.engine_last[node]=self.ev.now+duration
        self.ledger.count['MAC']+=mac;self.ledger.count['HB_bytes']+=hb;self.ledger.count['scalar_ops']+=scalar
        self.ledger.count['nmp_busy_ps']+=duration
        self.nmp_by_node[node]+=duration
        self.ev.after(duration,self._job_done,node,callback,args)
    def _job_done(self,node,callback,args):
        self.busy[node]=False;callback(*args);self.kick(node)
    def run(self):
        self._new_round();self.ev.run(lambda:self.done)
        assert self.completed==len(self.records),(self.completed,len(self.records))
        assert not any(self.buffers),self.buffers
        assert not any(self.active) and not any(self.busy)
        assert self.contributor_checks==len(self.records)
        assert self.network.sent==self.network.hops
        assert self.network.received==self.network.expected_receivers
        assert not any(self.network.flows) and not any(self.network.portq)
        assert sum(r['latency_ps'] for r in self.rows)==self.ev.now
        # Future events can only be idle batching timers or empty arbitration.
        return dict(rows=self.rows,total_ps=self.ev.now,tokens=self.completed,useful_bytes=self.useful_bytes,
                    ledger=dict(self.cohort_ledger),peak_buffer_bytes=max(self.peaks),peak_by_node=self.peaks,
                    events=self.ev.events,batch_histogram=dict(self.matrix_batches),max_ru=self.network.max_ru,
                    max_owner_active=self.max_active,contributor_checks=self.contributor_checks,
                    request_completion_ps=self.request_done,token_timings=self.token_timings,
                    nmp_busy_ps_by_node=self.nmp_by_node,dma_busy_ps_by_node=self.dma_by_node,
                    packet_hops=self.network.hops,receiver_deliveries=self.network.received)
    def _new_round(self):
        self.round_start=self.ev.now;self.round_completed=0;self.token_lats=[]
        if self.phase=='prefill':
            # Fixed round-robin admission across requests, then per-owner FIFO.
            rows=[]
            for index in range(max(len(v) for v in self.by_request.values())):
                rows.extend(v[index] for _,v in sorted(self.by_request.items()) if index<len(v))
        else:rows=[v[self.step] for _,v in sorted(self.by_request.items()) if self.step<len(v)]
        self.round_expected=len(rows)
        if not rows:self.done=True;return
        for r in rows:self.pending[r['request_rank']%self.n].append(r)
        for node in range(self.n):self._admit(node)
    def _admit(self,node):
        q=self.pending[node];window=self.c['execution']['owner_window']
        while q and self.active[node]<window:
            r=q.popleft();self.active[node]+=1;self.max_active=max(self.max_active,self.active[node]);self.tid+=1
            dest=defaultdict(list)
            for eid in r['expert_ids']:dest[eid%self.n].append(eid)
            if self.m['shared']:
                for sid in range(self.m['shared']):
                    snode=node if node%2==sid else node^1;dest[snode].append(self.m['E']+sid)
            expected=sum(1<<eid for ids in dest.values() for eid in ids)
            t=Token(self.tid,r,node,dict(dest),expected,self.ev.now);self.tokens[t.id]=t
            self.useful_bytes+=2*self.d*sum(dst!=node for dst in dest)
            if self.replay:
                self.alloc(node,self.q);self.ev.at(self.ev.now,self._dispatch,t,priority=1)
            else:self.network.send(BASE,(node,),'INPUT',self.header+self.q,lambda _,t=t:self._input(t))
    def _input(self,t):
        self.alloc(t.owner,self.q)
        self.job(t.owner,'router',self.kernel.matrix('router',1),self._dispatch,t)
    def _dispatch(self,t):
        remote={d:ids for d,ids in t.destinations.items() if d!=t.owner}
        if t.owner in t.destinations:self._expert_input(t,t.owner,already_allocated=True)
        else:self.free(t.owner,self.q)
        if remote:
            sizes={d:self.header+self.q+self.entry*len(ids) for d,ids in remote.items()}
            if 'BCAST' in self.method or '_MC_' in self.method:
                size=self.header+self.q+self.entry*sum(map(len,remote.values()))
                self.network.send(t.owner,tuple(remote),'DISPATCH',size,lambda node,t=t:self._expert_input(t,node),sizes)
            else:
                for node,size in sorted(sizes.items()):self.network.send(t.owner,(node,),'DISPATCH',size,lambda node,t=t:self._expert_input(t,node))
    def _expert_input(self,t,node,already_allocated=False):
        if not already_allocated:self.alloc(node,self.q)
        t.computed[node]=0;t.outputs[node]=[]
        if self.replay:
            for eid in t.destinations[node]:self._one_expert_done(t,node,eid)
            return
        for eid in t.destinations[node]:
            q=self.experts[node].setdefault(eid,deque());self.jobid+=1;q.append((self.ev.now,self.jobid,t))
            self.ev.after(self.c['execution']['batch_wait_ns']*1000,self.kick,node,priority=2)
        self.kick(node)
    def _expert_done(self,node,eid,batch):
        for t in batch:self._one_expert_done(t,node,eid)
    def _one_expert_done(self,t,node,eid):
        self.alloc(node,self.fp);t.computed[node]+=1;t.outputs[node].append(eid)
        complete=t.computed[node]==len(t.destinations[node])
        if complete:self.free(node,self.q)
        if node==t.owner:
            # Owner contributions already are weighted FP32; no D2D quantization.
            self._contribution(t,1<<eid,self.fp)
        elif not self.lc:self._return_prepare(t,node,[eid])
        elif complete:self._return_prepare(t,node,t.outputs[node][:])
    def _return_prepare(self,t,node,eids):
        count=len(eids);bits=sum(1<<e for e in eids)
        # Fused local sum + quantization: actual FP32 reads, one FP8 write.
        cost=self.kernel.vector(count*self.fp,self.q,(count-1)*self.d+2*self.d)
        if self.replay:self._return_ready(t,node,bits,count)
        else:self.job(node,'combine',cost,self._return_ready,t,node,bits,count)
    def _return_ready(self,t,node,bits,count):
        self.free(node,count*self.fp)
        self.network.send(node,(t.owner,),'RETURN',self.return_size,lambda _,t=t,bits=bits:self._remote_return(t,bits))
    def _remote_return(self,t,bits):
        # The DMA packet is consumed; retain only its quantized vector until Combine.
        self.alloc(t.owner,self.q);self._contribution(t,bits,self.q)
    def _contribution(self,t,bits,size):
        assert not(t.got&bits),(t.id,t.got,bits)
        assert bits&t.expected==bits
        t.got|=bits;t.incoming.append(size)
        if t.got==t.expected:
            self.contributor_checks+=1
            # Remote FP8 is dequantized into FP32 registers. No extra DRAM FP32 copy.
            cost=self.kernel.vector(sum(t.incoming),self.q,(len(t.incoming)-1)*self.d+2*self.d)
            if self.replay:self._combined(t)
            else:self.job(t.owner,'combine',cost,self._combined,t)
    def _combined(self,t):
        self.free(t.owner,sum(t.incoming))
        if self.replay:self._complete(t)
        else:self.network.send(t.owner,(BASE,),'OUTPUT',self.header+self.q,lambda _,t=t:self._complete(t))
    def _complete(self,t):
        assert t.got==t.expected
        self.completed+=1;self.round_completed+=1;self.active[t.owner]-=1
        # Prefill latency includes time waiting for admission (all inputs ready at 0).
        latency=self.ev.now-(self.round_start if self.phase=='decode' else 0)
        self.token_lats.append(latency);self.request_done[t.record['request_rank']]=self.ev.now
        self.token_timings.append([t.record['request_rank'],t.record['token_index'],t.start,self.ev.now,latency])
        del self.tokens[t.id]
        self._admit(t.owner)
        if self.round_completed==self.round_expected:
            ledger=self.ledger.delta(self.ev.now)
            for key,value in ledger.items():self.cohort_ledger[key]+=value
            self.rows.append(dict(cohort=self.cohort,step_index=self.step if self.phase=='decode' else -1,
                                  tokens=self.round_expected,latency_ps=self.ev.now-self.round_start,
                                  p95_token_ps=float(np.percentile(self.token_lats,95)),ledger=ledger))
            if self.phase=='prefill':self.done=True
            else:self.step+=1;self.ev.at(self.ev.now,self._new_round,priority=1)
