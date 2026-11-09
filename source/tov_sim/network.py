"""TSV store-and-forward and two shared wireless domains, without latency caches."""
from collections import defaultdict,deque
from dataclasses import dataclass
import heapq
import math
from .config import BASE,ceildiv,control_ps,derive


@dataclass(slots=True)
class Packet:
    id:int
    src:int
    dst:tuple
    kind:str
    size:int
    callback:object
    sizes:dict|None=None
    ready:int=0
    remaining:float=0


class Network:
    def __init__(self,sim):
        self.s=sim;self.c=sim.c;self.ev=sim.ev;self.l=sim.ledger;self.n=self.c['hardware']['dies']
        self.wireless=not sim.method.startswith('TSV');self.ofdm='OFDMA' in sim.method
        self.ids=0;self.flowids={};self.flows=[{},{}];self.pending=[False,False];self.nextframe=[0,0]
        self.portq=[[] for _ in range(self.n)];self.portbusy=[False]*self.n;self.direction=[None]*self.n
        self.portlast=[0]*self.n;self.portkick=[False]*self.n;self.max_ru=0;self.frames=[0,0];self.sent=0;self.received=0;self.hops=0;self.expected_receivers=0
        self.d=derive(self.c);self.pipe=(self.c['radio']['endpoint_pipeline_ns']+(self.c['radio']['ofdm_extra_endpoint_ns'] if self.ofdm else 0))*1000 if self.wireless else self.c['tsv']['endpoint_pipeline_ns']*1000
    def send(self,src,dst,kind,size,callback,sizes=None):
        assert dst and src not in dst
        self.ids+=1;p=Packet(self.ids,src,tuple(sorted(dst)),kind,int(size),callback,sizes)
        assert 0<p.size<4096,p.size
        if self.wireless:self._hop(p)
        elif src==BASE or p.dst==(BASE,):self._hop(p)
        else:
            # One ingress for MC; P2P callers already issue one packet per Die.
            target=p.dst;original=p.callback;size_map=sizes or {d:p.size for d in target}
            def forward(_):
                self.s.alloc(BASE,p.size)
                self.ev.after(self.c['tsv']['base_route_ns']*1000,self._forward,target,kind,size_map,original,p.size)
            p.dst=(BASE,);p.callback=forward;self._hop(p)
    def _forward(self,target,kind,sizes,callback,held):
        for dst in target:self.send(BASE,(dst,),kind,sizes[dst],callback)
        self.s.free(BASE,held)
    def _hop(self,p):
        self.hops+=1;self.expected_receivers+=len(p.dst)
        self.s.alloc(p.src,p.size)
        self.s.dma(p.src,p.size,self._tx_ready,p)
    def _tx_ready(self,p):
        start=self.ev.now
        if self.wireless:
            domain=0 if p.src==BASE or BASE in p.dst else 1
            self.l.radio(domain,p.src,'tx',start,start+self.pipe)
        self.ev.after(self.pipe,self._ready,p)
    @staticmethod
    def priority(p):return (0 if p.kind in ('RETURN','OUTPUT') else 1,p.ready,p.id)
    def _ready(self,p):
        p.ready=self.ev.now;p.remaining=p.size*8
        if self.wireless:
            domain=0 if p.src==BASE or BASE in p.dst else 1
            key=(p.src,p.dst,p.kind)
            if key not in self.flowids:self.flowids[key]=len(self.flowids)
            self.flows[domain].setdefault(key,deque()).append(p);self._schedule_frame(domain)
        else:
            port=p.dst[0] if p.src==BASE else p.src
            heapq.heappush(self.portq[port],(self.priority(p),p));self._schedule_port(port)
    def _schedule_port(self,port):
        if not self.portbusy[port] and not self.portkick[port]:
            self.portkick[port]=True;self.ev.at(self.ev.now,self._port,port,priority=2)
    def _port(self,port):
        self.portkick[port]=False
        if self.portbusy[port] or not self.portq[port]:return
        _,p=heapq.heappop(self.portq[port]);direction=p.src==BASE
        turn=self.c['tsv']['turnaround_ns']*1000 if self.direction[port] is not None and self.direction[port]!=direction else 0
        self.direction[port]=direction;self.portbusy[port]=True
        rate=(self.d['tsv_high_port_gbps'] if 'HIGH' in self.s.method else self.d['tsv_norm_port_gbps'])*self.c['tsv']['payload_efficiency']/1000
        a=self.ev.now+turn;b=a+math.ceil(p.size*8/rate)
        assert a>=self.portlast[port];self.portlast[port]=b
        self.l.count['tsv_busy_port_ps']+=b-a
        self.ev.at(b,self._port_done,port,p)
    def _port_done(self,port,p):
        self.portbusy[port]=False;bits=p.size*8
        self.l.count['tsv_bits']+=bits;self.l.count['physical_tx_bits']+=bits
        self._delivered(p,None);self._schedule_port(port)
    def _delivered(self,p,domain):
        self.sent+=1;self.s.free(p.src,p.size)
        self.l.count[p.kind+'_tx_bits']+=p.size*8
        self.l.count['data_rx_bits']+=p.size*8*len(p.dst)
        self.l.count['packet_hops']+=1;self.l.count['receiver_deliveries']+=len(p.dst)
        if len(p.dst)>1:
            self.l.count['broadcast_packets']+=1;self.l.count['broadcast_fanout_sum']+=len(p.dst)
        if domain is not None:
            bits=p.size*8;self.l.count['physical_tx_bits']+=bits;self.l.count[f'domain_{domain}_data_bits']+=bits
        for dst in p.dst:
            self.s.alloc(dst,p.size)
            if domain is not None:self.l.radio(domain,dst,'rx',self.ev.now,self.ev.now+self.pipe)
            self.ev.after(self.pipe,self._rx_dma,p,dst)
    def _rx_dma(self,p,dst):self.s.dma(dst,p.size,self._rx_done,p,dst)
    def _rx_done(self,p,dst):
        self.received+=1;self.s.free(dst,p.size);p.callback(dst)
    def _schedule_frame(self,domain):
        if self.pending[domain]:return
        frame=self.c['radio']['frame_ns']*1000
        when=max(self.nextframe[domain],ceildiv(self.ev.now,frame)*frame)
        self.pending[domain]=True;self.ev.at(when,self._frame,domain,priority=2)
    def _frame(self,domain):
        self.pending[domain]=False;flows=self.flows[domain]
        if not flows:return
        now=self.ev.now;r=self.c['radio'];frame=r['frame_ns']*1000;C=control_ps(self.c,domain)
        a=now+C+r['guard_ns']*1000;end=now+frame
        if a>=end:raise ValueError('Wireless control exhausts the frame')
        self.nextframe[domain]=end;self.frames[domain]+=1;self.l.count[f'domain_{domain}_frames']+=1
        participants=list(range(self.n))+([BASE] if domain==0 else [])
        controller=BASE if domain==0 else 0
        for node in participants:self.l.radio(domain,node,'tx' if node==controller else 'rx',now,now+C)
        bits=8*(len(participants)*r['report_bytes']+r['grant_base_bytes']+r['rus']*r['grant_bytes_per_ru'])
        self.l.count[f'domain_{domain}_control_bits']+=bits;self.l.count['physical_tx_bits']+=bits
        # Snapshot only already-ready packets. New arrivals wait for a later report.
        per_src={}
        for key,q in flows.items():
            p=q[0];priority=(0 if p.kind in ('RETURN','OUTPUT') else 1,p.ready,self.flowids[key])
            if p.src not in per_src or priority<per_src[p.src][0]:per_src[p.src]=(priority,key)
        candidates=[key for _,key in sorted(per_src.values())]
        snapshots={key:len(flows[key]) for key in candidates}
        if self.ofdm:self._ofdma(domain,candidates,snapshots,a,end)
        else:self._tdma(domain,candidates,snapshots,a,end)
        # Service is reserved now, but delivery occurs at its physical completion.
        for key in list(flows):
            if not flows[key]:del flows[key]
        if flows:self._schedule_frame(domain)
    def _rate(self,key):
        r=self.c['radio'];return (r['unicast_gbps'] if len(key[1])==1 else r['broadcast_gbps'])*r['rate_scale']/1000
    def _service(self,domain,key,count,a,end,rate,rus):
        q=self.flows[domain][key];t=a;src,dst,_=key;served=0
        while count and q and t<end:
            p=q[0];capacity=rate*(end-t)
            if p.remaining<=capacity+1e-9:
                assert p.remaining-capacity<=1e-6
                finish=min(end,t+math.ceil(p.remaining/rate));served+=p.remaining;p.remaining=0
                q.popleft();count-=1;self.ev.at(finish,self._delivered,p,domain);t=finish
            else:
                previous=p.remaining;p.remaining-=capacity
                assert abs((previous-p.remaining)-capacity)<=1e-6
                served+=capacity;t=end
        if t>a:
            self.l.radio(domain,src,'tx',a,t)
            for node in dst:self.l.radio(domain,node,'rx',a,t)
            self.l.air.add(domain,a,t)
            self.l.count['allocated_ru_ps']+=rus*(t-a)
            self.l.count[f'domain_{domain}_allocated_ru_ps']+=rus*(t-a)
        return t
    def _tdma(self,domain,keys,snapshots,a,end):
        directions={};t=a
        for key in keys:
            src,dst,_=key
            change=directions.get(src)=='rx' or any(directions.get(d)=='tx' for d in dst)
            if change:t+=self.c['radio']['guard_ns']*1000
            if t>=end:break
            directions[src]='tx'
            for d in dst:directions[d]='rx'
            t=self._service(domain,key,snapshots[key],t,end,self._rate(key),self.c['radio']['rus'])
    def _ofdma(self,domain,keys,snapshots,a,end):
        r=self.c['radio'];eta=self.d['ofdm_efficiency'];chosen=[];tx=set();rx=set();alloc={};need={};qbits={}
        for key in keys:
            src,dst,_=key
            if src in rx or any(d in tx for d in dst):continue
            if len(chosen)==r['rus']:break
            chosen.append(key);tx.add(src);rx.update(dst);alloc[key]=1
            bits=sum(p.remaining for p in list(self.flows[domain][key])[:snapshots[key]])
            qbits[key]=bits;need[key]=min(r['rus'],max(1,math.ceil(bits/(self._rate(key)*eta*(end-a)/r['rus']))))
        assert not tx.intersection(rx)
        remaining=r['rus']-len(chosen)
        while remaining:
            available=[k for k in chosen if alloc[k]<need[k]]
            if not available:break
            key=min(available,key=lambda k:(-qbits[k]/(self._rate(k)*alloc[k]),self.flowids[k]))
            alloc[key]+=1;remaining-=1
        used=sum(alloc.values());assert used<=r['rus'];self.max_ru=max(self.max_ru,used)
        for key in chosen:self._service(domain,key,snapshots[key],a,end,self._rate(key)*eta*alloc[key]/r['rus'],alloc[key])
