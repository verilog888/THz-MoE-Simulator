"""Integer-picosecond event calendar and auditable activity ledgers."""
from collections import defaultdict
from bisect import bisect_left
import heapq


class Calendar:
    def __init__(self):self.now=0;self.seq=0;self.queue=[];self.events=0
    def at(self,time,fn,*args,priority=0):
        assert time>=self.now,(time,self.now)
        self.seq+=1;heapq.heappush(self.queue,(int(time),priority,self.seq,fn,args))
    def after(self,delay,fn,*args,priority=0):self.at(self.now+int(delay),fn,*args,priority=priority)
    def run(self,done):
        while not done():
            if not self.queue:raise RuntimeError('Deadlock: unfinished tokens and empty calendar')
            time,_,_,fn,args=heapq.heappop(self.queue);self.now=time;self.events+=1;fn(*args)


class Intervals:
    """Sorted union, including a short future tail scheduled in the current frame."""
    def __init__(self):self.rows=defaultdict(list)
    def add(self,key,start,end):
        if end<=start:return
        rows=self.rows[key]
        if not rows or start>rows[-1][1]:rows.append((start,end));return
        i=bisect_left(rows,(start,-1))
        if i and rows[i-1][1]>=start:i-=1
        j=i
        while j<len(rows) and rows[j][0]<=end:
            start=min(start,rows[j][0]);end=max(end,rows[j][1]);j+=1
        rows[i:j]=[(start,end)]
    def through_by_key(self,time):
        totals={}
        for key,rows in self.rows.items():
            keep=[];total=0
            for a,b in rows:
                total+=max(0,min(b,time)-a)
                if b>time:keep.append((max(a,time),b))
            self.rows[key]=keep
            totals[key]=total
        return totals
    def through(self,time):return sum(self.through_by_key(time).values())


class Ledger:
    def __init__(self,c,wireless,ofdm):
        self.c=c;self.wireless=wireless;self.ofdm=ofdm;self.count=defaultdict(int)
        self.tx=Intervals();self.rx=Intervals();self.endpoints=Intervals();self.air=Intervals()
        self.energy={};self.last_time=0
    def radio(self,domain,node,role,a,b):
        (self.tx if role=='tx' else self.rx).add((domain,node),a,b)
        self.endpoints.add((domain,node),a,b)
    def delta(self,time):
        c=self.c;e=c['energy'];v=dict(self.count);self.count.clear()
        tx=self.tx.through(time);rx=self.rx.through(time);active=self.endpoints.through(time)
        air=self.air.through_by_key(time);dt=time-self.last_time;self.last_time=time
        parts=dict(mac_J=v.get('MAC',0)*e['mac_pJ']*1e-12,
                   scalar_J=v.get('scalar_ops',0)*e['scalar_pJ_per_op']*1e-12,
                   hb_J=v.get('HB_bytes',0)*8*e['hb_pJ_per_bit']*1e-12,
                   dram_core_J=v.get('HB_bytes',0)*8*e['dram_core_pJ_per_bit']*1e-12,
                   tsv_J=v.get('tsv_bits',0)*e['tsv_pJ_per_bit_hop']*1e-12,
                   rf_lo_J=(c['hardware']['dies']+1)*e['rf_lo_mW_per_node']*1e-3*dt*1e-12 if self.wireless else 0,
                   rf_tx_J=tx*1e-12*e['rf_tx_mW']*1e-3,
                   rf_rx_J=rx*1e-12*e['rf_rx_mW']*1e-3,
                   ofdm_J=active*1e-12*e['ofdm_W_per_active_endpoint'] if self.ofdm else 0,
                   background_J=(c['hardware']['dies']*e['background_W_per_die']+e['base_W'])*dt*1e-12)
        v.update(parts);v.update(energy_proxy_J=sum(parts.values()),ofdm_active_endpoint_ps=active if self.ofdm else 0,
                                radio_tx_ps=tx,radio_rx_ps=rx,domain_busy_ps=sum(air.values()),
                                domain_0_data_busy_ps=air.get(0,0),domain_1_data_busy_ps=air.get(1,0),elapsed_ps=dt)
        return v
