// Native acceleration for the Python discrete-event model. All times are integer ps.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>
namespace py=pybind11;
using I=int64_t;
constexpr I CLK=5000, Q=1088000;
enum Category {MATRIX,VECTOR,DMA,RADIO,CONTROL,WAIT,RXPIPE,BATCH,CLOCK,NCAT};
static const char* CAT[]={"matrix","vector","dma","radio","control","queue_wait","rx_pipeline","batch_wait","clock_alignment"};
I ceildiv(I a,I b){return (a+b-1)/b;}
I align(I t){return ceildiv(t,CLK)*CLK;}
void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
struct Path{
 I t=0; std::array<I,NCAT> parts{};
 Path until(I to,int cat)const{require(to>=t,"Causal time went backwards");Path p=*this;p.parts[cat]+=to-t;p.t=to;return p;}
};
Path later(const Path&a,const Path&b){return a.t>=b.t?a:b;}
struct Params{
 int d,h,E,k,shared;bool bias,rbias,gpt,ofdma,bc,lc,tw,allrx,decode;
 int tilem,tilen,tilek,lanes,setup,actops,qops,softops,ownerlimit,mtu;
 I ptp,br,scale4,frame,control[2],guard,rxpipe,prop,wait;
 I rxcap,txcap,poolcap,ownercap,tilecap,ctrlcap;
 double util,bw;
 Params(py::dict c,py::dict m,const std::string& baseline,bool dec):decode(dec){
  auto sec=[&](const char*s){return c[s].cast<py::dict>();};
  auto hw=sec("hardware"), co=sec("compute"),me=sec("memory"),ra=sec("radio"),ma=sec("mac");
  d=m["d_model"].cast<int>();h=m["d_ff"].cast<int>();E=m["routed_experts"].cast<int>();
  k=m["top_k"].cast<int>();shared=m["shared_experts"].cast<int>();
  bias=m["expert_linear_bias"].cast<bool>();rbias=m["router_linear_bias"].cast<bool>();
  gpt=m["activation"].cast<std::string>()=="gpt_oss_swiglu";
  ofdma=baseline[0]=='O';tw=baseline.substr(0,2)=="TW";bc=baseline.find("_B_")!=std::string::npos;lc=baseline.back()=='L';
  allrx=ra["multicast_receivers"].cast<std::string>()=="all_non_source";
  util=hw["compute_utilization"].cast<double>();bw=me["bandwidth_scale"].cast<double>();
  tilem=co["tile_m"].cast<int>();tilen=co["tile_n"].cast<int>();tilek=co["tile_k"].cast<int>();
  lanes=hw["vector_lanes"].cast<int>();setup=hw["compute_setup_cycles"].cast<int>();
  actops=co[gpt?"gpt_activation_ops_per_element":"activation_ops_per_element"].cast<int>();
  qops=co["quantize_ops_per_element"].cast<int>();softops=co["softmax_ops_per_element"].cast<int>();
  ownerlimit=hw["owner_inflight_limit"].cast<int>();mtu=ma["mtu_payload_bytes"].cast<int>();
  ptp=ra["ptp_raw_bps"].cast<I>()/1000000000; br=ra["broadcast_raw_bps"].cast<I>()/1000000000;
  scale4=llround(ra["rate_scale"].cast<double>()*4);
  require(scale4>0 && std::abs(ra["rate_scale"].cast<double>()*4-scale4)<1e-12,"Rate scale must be a quarter multiple");
  guard=ma["guard_cycles"].cast<I>()*CLK;frame=ma["frame_cycles"].cast<I>()*CLK;
  for(int med=0;med<2;med++){
   long double seconds=8.L*(med?16:17)*ma["queue_report_bytes"].cast<int>()/(ptp*1e9L*scale4/4)
    +8.L*(ma["grant_header_bytes"].cast<int>()+16*ma["grant_bytes_per_ru"].cast<int>())/(br*1e9L*scale4/4)
    +ma["arbitration_cycles"].cast<int>()*5e-9L;
   control[med]=std::max(ma["control_cycles"].cast<I>()*CLK,I(ceill(seconds*2e8L-1e-14L))*CLK);
  }
  rxpipe=ra["rx_pipeline_cycles"].cast<I>()*CLK;prop=ra["propagation_delay_ps"].cast<I>();
  wait=co["batch_wait_cycles"].cast<I>()*CLK;
  auto bp=hw["buffer_partition_bytes"].cast<std::vector<I>>();
  tilecap=bp[0];rxcap=bp[1]/2;txcap=bp[2]/2;ownercap=hw["partial_owner_reserve_bytes"].cast<I>();
  poolcap=bp[3]-ownercap;ctrlcap=bp[4];
  require(tilecap>0&&rxcap>0&&txcap>0,"Invalid buffer partition");
  require(hw["rx_return_reserve_bytes"].cast<I>()==rxcap&&hw["tx_return_reserve_bytes"].cast<I>()==txcap,"Asymmetric reserves are not supported");
 }
 I vbytes(int n)const{return n+4*ceildiv(n,128);}
 I rate(bool multi,int ru)const{
  I raw=(multi?br:ptp)*scale4;
  return ofdma?raw*14*ru:raw*272;
 }
 I frame_len(int med)const{return std::max(frame,control[med]+guard+CLK);}
 I scratch(int n)const{return n*(8*h+vbytes(h));}
};
struct Ledger{
 I matrix_ops=0,vector_ops=0,read_bytes=0,write_bytes=0,tx_ps=0,rx_ps=0,ofdma_ps=0;
 I control_tx_ps=0,control_rx_ps=0,air_bytes[4]{},received_bytes=0,frame_count[2]{};
 I ru_ps=0,available_ru_ps=0,tail_ru_ps=0,control_ps[2]{},data_window_ps[2]{};
 I cpu_ps[16]{},dma_ps[16]{},max_rx[16]{},max_tx[16]{},max_pool[16]{},max_owner[16]{},max_tile[16]{},max_control[16]{};
 I batch_calls=0,batch_tokens=0,events=0,packets=0,rx_groups=0,full_group_fanout=0;
 I compute_wait_ps=0,dma_wait_ps=0,network_wait_ps=0,injection_wait_ps=0;
 I rx_credit_blocks=0,partial_credit_blocks=0,tx_credit_blocks=0;
};
struct Event{I t;int phase;I seq;std::function<void()> fn;};
struct EventCmp{bool operator()(Event const&a,Event const&b)const{return std::tie(a.t,a.phase,a.seq)>std::tie(b.t,b.phase,b.seq);}};
struct Tile{I read=0,write=0,mat=0,vec=0;int stage=0;bool startup=false;};
struct Job{
 int die=0,priority=0,expert=-1,cur=0;I key=0;bool blocked=false,done=false,computing=false;
 std::shared_ptr<std::vector<Tile>> tiles;
 std::array<int,2> load_index{-1,-1},load_state{};
 std::array<Path,2> readpath;
 bool loaded(int i)const{return load_index[i%2]==i&&load_state[i%2]!=0;}
 bool ready(int i)const{return load_index[i%2]==i&&load_state[i%2]==2;}
 Path path; I scratch=0;
 std::function<void(Path)> finish;
};
struct MemReq{I read,write;Path path;std::function<void(Path)> finish;};
struct Die{
 bool compute_busy=false,memory_busy=false,expert_active=false,kicking=false;
 bool kick_scheduled=false,expert_scheduled=false;
 int slots=0,inflight=0;I rx[2]{},tx[2]{},pool=0,owner=0,control_bytes=0;
 std::vector<std::shared_ptr<Job>> jobs;
 std::deque<MemReq> mem;
 Path last_compute,last_dma;
};
struct Group{
 int die=0,expected=0,finished=0,pending=0;
 bool allocated=false,arrived=false,released=false;
 I rxheld=0,rxfrags=0;Path path;
 std::bitset<256> contributions;
};
struct Token{
 int id,req,pos,row,word,owner;std::vector<int> experts;
 bool injected=false,done=false,gate_done=false;int dispatch_pending=0;
 I ready=0,end=0;Path root,accum;int received=0;
 std::bitset<256> contributed;
 std::map<int,Group> groups;
 bool combining=false;
 struct Incoming{int source;std::bitset<256> bits;Path path;std::function<void()> release;};
 std::deque<Incoming> incoming;
};
enum Kind{INPUT,DISPATCH,RETURN,OUTPUT};
struct Message{
 int prepared=0,frags=0;I rxbytes[17]{};int arrived[17]{};Path last[17];
};
struct Packet{
 int token,src,med,kind,frag,frags;uint32_t mask;I bytes,remaining,ready;
 bool reserved=false,queued=false;
 std::bitset<256> contributors; Path path;
 std::shared_ptr<Message> message;
};
using FlowKey=std::tuple<int,uint32_t,int>;
struct Net{
 std::map<FlowKey,std::deque<std::shared_ptr<Packet>>> flows;
 bool scheduled=false;I available=0;
 Path last;
};
struct Request{std::vector<int> ids;int cursor=0;};

struct Sim{
 Params p; I now=0,sequence=0;Ledger l;std::array<Die,16> dies;
 std::priority_queue<Event,std::vector<Event>,EventCmp> events;
 std::array<Net,2> nets;std::vector<Token> toks;std::map<int,Request> requests;
 std::array<std::map<int,std::deque<std::pair<int,Path>>>,16> expertq;
 std::vector<std::shared_ptr<Packet>> pending_tx;
 std::map<std::tuple<int,int,int,int>,std::shared_ptr<std::vector<Tile>>> kernels;
 int completed=0,round=0,round_remaining=0,rr=0;
 bool injecting=false,pumping=false;Path barrier,last;
 std::vector<std::array<I,3>> rounds;
 Sim(const Params&params,const std::vector<std::vector<int>>& desc):p(params){
  for(auto const&r:desc){
   Token t; t.id=toks.size();t.req=r[0];t.pos=r[1];t.row=r[2];t.word=r[3];t.owner=t.req%16;
   for(size_t j=4;j<r.size();j++)t.experts.push_back(r[j]);
   require((int)t.experts.size()==p.k,"Trace top-k mismatch");
   require(std::set<int>(t.experts.begin(),t.experts.end()).size()==t.experts.size(),"Duplicate routed contribution");
   requests[t.req].ids.push_back(t.id);toks.push_back(t);
  }
 }
 void at(I time,std::function<void()> f,int phase=0){require(time>=now,"Past event");events.push({time,phase,sequence++,std::move(f)});}
 I uid(int tid)const{return (I(toks[tid].row)<<32)+toks[tid].pos;}
 void descriptor(int d,I delta){
  if(d==16)return;
  auto &x=dies[d];x.control_bytes+=delta;
  require(x.control_bytes>=0&&x.control_bytes<=p.ctrlcap,"Control descriptors exceed workspace");
  l.max_control[d]=std::max(l.max_control[d],x.control_bytes);
 }
 void audit(int d){
  auto &x=dies[d];
  require(x.pool>=0&&x.pool<=p.poolcap,"Partial buffer overflow");
  require(x.owner>=0&&x.owner<=p.ownercap,"Owner accumulator overflow");
  require(x.rx[0]>=0&&x.rx[0]<=p.rxcap&&x.rx[1]>=0&&x.rx[1]<=p.rxcap,"RX buffer overflow");
  require(x.tx[0]>=0&&x.tx[0]<=p.txcap&&x.tx[1]>=0&&x.tx[1]<=p.txcap,"TX buffer overflow");
  require(x.slots>=0&&x.slots<=2,"Tile buffers overflow");
  l.max_pool[d]=std::max(l.max_pool[d],x.pool);l.max_owner[d]=std::max(l.max_owner[d],x.owner);
  l.max_rx[d]=std::max(l.max_rx[d],x.rx[0]+x.rx[1]);l.max_tx[d]=std::max(l.max_tx[d],x.tx[0]+x.tx[1]);
  l.max_tile[d]=std::max(l.max_tile[d],I(x.slots)*p.tilecap/2);
 }
 void dma(int d,I read,I write,Path dep,std::function<void(Path)> f){
  if(d==16){f(dep.until(now,WAIT));return;}
  if(!read&&!write){f(dep.until(now,WAIT));return;}
  dies[d].mem.push_back({read,write,dep,std::move(f)});kick_mem(d);
 }
 void kick_mem(int d){
  auto &x=dies[d];if(x.memory_busy||x.mem.empty())return;
  auto r=std::move(x.mem.front());x.mem.pop_front();x.memory_busy=true;
  I start=align(now),duration=CLK+ceildiv(r.read+r.write,llround(1024*p.bw))*CLK;
  l.dma_wait_ps+=start-r.path.t;
  auto path=later(r.path,x.last_dma).until(now,WAIT).until(start,CLOCK).until(start+duration,DMA);
  l.read_bytes+=r.read;l.write_bytes+=r.write;l.dma_ps[d]+=duration;
  at(start+duration,[this,d,r=std::move(r),path](){dies[d].memory_busy=false;dies[d].last_dma=path;r.finish(path);kick_mem(d);});
 }
 std::shared_ptr<std::vector<Tile>> kernel(int type,int m,int expert=-1){
  auto key=std::make_tuple(type,m,expert>=p.E?1:0,0);
  auto it=kernels.find(key);if(it!=kernels.end())return it->second;
  auto v=std::make_shared<std::vector<Tile>>();int stage=0;
  auto vec=[&](I ops,I rd,I wr){v->push_back({rd,wr,0,ops,stage++,true});};
  auto matrix=[&](int n,int k,bool bias){
   int s=stage++;
   for(int ni=0;ni<n;ni+=p.tilen)for(int ki=0;ki<k;ki+=p.tilek){
    int nn=std::min(p.tilen,n-ni),kk=std::min(p.tilek,k-ki);
    I rd=I(nn)*kk+4+I(m)*(kk+4);
    I wr=ki+kk==k?I(4)*m*nn:0;
    v->push_back({rd,wr,I(2)*m*nn*kk,0,s,ni==0&&ki==0});
   }
   if(bias)vec(I(m)*n,I(4)*m*n+p.vbytes(n),I(4)*m*n);
  };
  if(type==1){
   matrix(p.E,p.d,p.rbias);
   vec(I(p.softops)*p.E+I(p.k)*(p.E-1),I(4)*p.E,4*p.k);
  }else if(type==2){
   matrix(p.h,p.d,p.bias);matrix(p.h,p.d,p.bias);
   // Vector activation/quantization streams one 128-element scale block at a time.
   for(int hi=0;hi<p.h;hi+=128){
    int n=std::min(128,p.h-hi);
    vec(I(m)*n*(p.actops+p.qops)+m,I(8)*m*n,I(m)*(n+4));
    v->back().startup=hi==0;
   }
   matrix(p.d,p.h,p.bias);
   if(expert<p.E)vec(I(m)*p.d,I(4)*m*p.d+4*m,I(4)*m*p.d);
  }
  kernels[key]=v;return v;
 }
 void job(int d,int priority,int key,int expert,std::shared_ptr<std::vector<Tile>> tiles,
          Path path,std::function<void(Path)> f,I scratch=0){
  auto j=std::make_shared<Job>();j->die=d;j->priority=priority;j->key=uid(key);j->expert=expert;
  j->tiles=tiles;j->path=path;j->finish=std::move(f);j->scratch=scratch;
  dies[d].jobs.push_back(j);kick_die(d);
  descriptor(d,32);
 }
 void vector_job(int d,int key,I ops,I read,I write,Path path,std::function<void(Path)> f){
  auto v=std::make_shared<std::vector<Tile>>();
  v->push_back({read,write,0,ops,0,true});job(d,0,key,-1,v,path,std::move(f));
 }
 void load_tile(std::shared_ptr<Job> j,int i){
  auto &x=dies[j->die];auto &tile=j->tiles->at(i);
  require(x.slots<2,"No tile credit");x.slots++;audit(j->die);j->load_index[i%2]=i;j->load_state[i%2]=1;
  require(tile.read<=p.tilecap/2,"Tile DMA exceeds double buffer");
  dma(j->die,tile.read,0,j->path,[this,j,i](Path path){j->load_state[i%2]=2;j->readpath[i%2]=path;kick_die(j->die);});
 }
 void kick_die(int d){
  auto &x=dies[d];if(x.kick_scheduled)return;x.kick_scheduled=true;
  at(now,[this,d](){dies[d].kick_scheduled=false;dispatch_die(d);},2);
 }
 void dispatch_die(int d){
  auto &x=dies[d];if(x.kicking)return;x.kicking=true;
  x.jobs.erase(std::remove_if(x.jobs.begin(),x.jobs.end(),[](auto j){return j->done;}),x.jobs.end());
  std::shared_ptr<Job> best;
  for(auto j:x.jobs)if(!j->done)
   if(!best||std::tie(j->priority,j->key,j->expert)<std::tie(best->priority,best->key,best->expert))best=j;
  if(best){
   int i=best->cur;
   if(!best->blocked&&!best->loaded(i)&&x.slots<2)load_tile(best,i);
   auto prefetch=best;
   std::shared_ptr<Job> executable;
   for(auto j:x.jobs)if(!j->done&&!j->blocked&&j->ready(j->cur))
    if(!executable||std::tie(j->priority,j->key,j->expert)<std::tie(executable->priority,executable->key,executable->expert))executable=j;
   if(!x.compute_busy&&executable){
    best=executable;i=best->cur;
    x.compute_busy=true;best->blocked=true;best->computing=true;auto tile=best->tiles->at(i);
    I start=align(now);
    I matcycles=ceildiv(tile.mat,llround(5000*p.util));
    I veccycles=ceildiv(tile.vec,p.lanes);
    I startup=tile.startup?p.setup*CLK:0;
    I duration=startup+(matcycles+veccycles)*CLK;
    l.compute_wait_ps+=start-std::max(best->path.t,best->readpath[i%2].t);
    Path path=later(later(best->path,best->readpath[i%2]),x.last_compute).until(now,WAIT).until(start,CLOCK);
    path=path.until(path.t+startup,CLOCK);
    path=path.until(path.t+matcycles*CLK,MATRIX);
    path=path.until(start+duration,VECTOR);
    l.matrix_ops+=tile.mat;l.vector_ops+=tile.vec;l.cpu_ps[d]+=duration;
    at(start+duration,[this,best,d,i,tile,path](){
     auto &x=dies[d];x.compute_busy=false;x.last_compute=path;best->computing=false;x.slots--;best->load_state[i%2]=0;
     best->cur++;best->path=path;audit(d);
     auto finished=[this,best,d](Path end){
      best->path=later(best->path,end);best->blocked=false;
      if(best->cur==(int)best->tiles->size()){
       best->done=true;descriptor(d,-32);dies[d].pool-=best->scratch;audit(d);best->finish(best->path);
      }
      kick_die(d);try_expert(d);pump_tx();wake_networks();
     };
     if(tile.write)dma(d,0,tile.write,path,finished);else finished(path);
     kick_die(d);
    });
   }
   // Only independent reads inside the same matrix may be prefetched.
   best=prefetch;int next=best->cur+1;
   if((!best->blocked||best->computing)&&next<(int)best->tiles->size()&&best->loaded(best->cur)&&x.slots<2
      &&best->tiles->at(best->cur).stage==best->tiles->at(next).stage
      &&!best->loaded(next))load_tile(best,next);
  }
  x.kicking=false;
 }
 void try_expert(int d);
 void dispatch_expert(int d);
 void inject();
 void gate(int tid,Path path);
 void group_arrived(int tid,int d,Path path);
 void expert_result(int tid,int d,int expert,Path path);
 void contribute(int tid,int d,std::bitset<256> bits,Path path,std::function<void()> release={});
 void combine_next(int tid);
 void release_group(int tid,int d);
 void message(int tid,int src,uint32_t mask,int kind,I body,std::bitset<256> bits,Path path);
 void pump_tx();
 void wake(int med);
 void wake_networks(){wake(0);wake(1);}
 void frame(int med);
 bool can_reserve(std::shared_ptr<Packet> q);
 void reserve(std::shared_ptr<Packet> q);
 void delivered(std::shared_ptr<Packet> q,Path path);
 void token_done(int tid,Path path);
 void run(){
  inject();
  while(completed<(int)toks.size()&&!events.empty()){
   auto ev=std::move(const_cast<Event&>(events.top()));events.pop();now=ev.t;l.events++;ev.fn();
  }
  if(completed!=(int)toks.size()){
   std::ostringstream s;s<<"Deadlock: "<<completed<<"/"<<toks.size()<<" tokens, time="<<now;
   for(int d=0;d<16;d++)s<<" die"<<d<<":pool="<<dies[d].pool<<",rx="<<dies[d].rx[0]<<","<<dies[d].rx[1]<<",jobs="<<dies[d].jobs.size()<<",slots="<<dies[d].slots;
   throw std::runtime_error(s.str());
  }
  for(int d=0;d<16;d++){audit(d);require(dies[d].pool==0&&dies[d].owner==0&&dies[d].control_bytes==0,"Buffers not drained");}
  require(std::accumulate(last.parts.begin(),last.parts.end(),I(0))==now,"Critical path does not close");
 }
};

void Sim::inject(){
 if(injecting)return;injecting=true;
 std::vector<int> ids;for(auto &[r,q]:requests)ids.push_back(r);
 if(p.decode&&round_remaining==0){
  for(auto &[r,q]:requests)if(q.cursor<(int)q.ids.size())round_remaining++;
  if(round_remaining)rounds.push_back({round,round_remaining,barrier.t});
 }
 bool progress=true;
 while(progress){
  progress=false;
  for(size_t z=0;z<ids.size();z++){
   int index=(rr+z)%ids.size();auto &r=requests[ids[index]];
   if(r.cursor==(int)r.ids.size())continue;
   auto &t=toks[r.ids[r.cursor]];
   if(p.decode&&t.pos!=round)continue;
   auto &x=dies[t.owner];if(x.inflight>=p.ownerlimit)continue;
   r.cursor++;x.inflight++;x.owner+=4*p.d;audit(t.owner);
   descriptor(t.owner,32);
   t.injected=true;t.ready=p.decode?barrier.t:0;t.root=p.decode?barrier:Path{};
   l.injection_wait_ps+=now-t.ready;
   auto &g=t.groups[t.owner];g.die=t.owner;
   message(t.id,16,1u<<t.owner,INPUT,p.vbytes(p.d),{},t.root);
   rr=(index+1)%ids.size();progress=true;break;
  }
 }
 injecting=false;
}

void Sim::gate(int tid,Path path){
 auto &t=toks[tid];
 job(t.owner,1,tid,-1,kernel(1,1),path,[this,tid](Path done){
  auto &t=toks[tid];t.gate_done=true;
  for(int e:t.experts){auto &g=t.groups[e%16];g.die=e%16;g.expected++;}
  for(int b=0;b<p.shared;b++){int d=t.owner%2==b?t.owner:t.owner^1;auto &g=t.groups[d];g.die=d;g.expected++;}
  uint32_t mask=0;int entries=0;
  for(auto &[d,g]:t.groups)if(d!=t.owner&&g.expected){mask|=1u<<d;entries+=g.expected;}
  if(p.bc&&mask)message(tid,t.owner,mask,DISPATCH,p.vbytes(p.d)+8*entries,{},done);
  else for(auto &[d,g]:t.groups)if(d!=t.owner&&g.expected)
   message(tid,t.owner,1u<<d,DISPATCH,p.vbytes(p.d)+8*g.expected,{},done);
  l.full_group_fanout+=__builtin_popcount(mask);
  group_arrived(tid,t.owner,done);release_group(tid,t.owner);
 });
}
void Sim::group_arrived(int tid,int d,Path path){
 auto &t=toks[tid];auto &g=t.groups.at(d);g.arrived=true;g.path=path;
 for(int e:t.experts)if(e%16==d){expertq[d][e].push_back({tid,path});descriptor(d,8);}
 for(int b=0;b<p.shared;b++)if((t.owner%2==b?t.owner:t.owner^1)==d){expertq[d][p.E+b].push_back({tid,path});descriptor(d,8);}
 for(auto &[e,q]:expertq[d])std::stable_sort(q.begin(),q.end(),[this](auto const&a,auto const&b){
  return std::make_tuple(a.second.t,uid(a.first))<std::make_tuple(b.second.t,uid(b.first));
 });
 try_expert(d);
 if(g.expected)at(now+p.wait,[this,d](){try_expert(d);},2);
}
void Sim::try_expert(int d){
 auto &x=dies[d];if(x.expert_active||x.expert_scheduled)return;x.expert_scheduled=true;
 at(now,[this,d](){dies[d].expert_scheduled=false;dispatch_expert(d);},1);
}
void Sim::dispatch_expert(int d){
 auto &x=dies[d];if(x.expert_active)return;
 int chosen=-1,first=-1;I oldest=std::numeric_limits<I>::max(),firstuid=std::numeric_limits<I>::max();
 for(auto &[e,q]:expertq[d]){
  if(q.empty())continue;
  auto &g=toks[q.front().first].groups.at(d);
  if(g.pending)continue;
  if(q.size()<size_t(p.tilem)&&now<q.front().second.t+p.wait)continue;
  if(std::make_tuple(q.front().second.t,uid(q.front().first),e)<std::tie(oldest,firstuid,chosen)){
   chosen=e;first=q.front().first;firstuid=uid(first);oldest=q.front().second.t;
  }
 }
 if(chosen<0)return;
 auto &q=expertq[d][chosen];std::vector<int> batch;Path path;
 int room=(p.poolcap-x.pool)/p.scratch(1);
 int count=std::min(p.tilem,room);if(count<=0)return;
 while(!q.empty()&&(int)batch.size()<count){
  int tid=q.front().first;if(toks[tid].groups.at(d).pending)break;
  path=later(path,q.front().second);batch.push_back(tid);q.pop_front();descriptor(d,-8);
 }
 if(batch.empty())return;
 I batch_ready=std::max(path.t,std::min(oldest+p.wait,now));
 path=path.until(batch_ready,BATCH);
 I scratch=p.scratch(batch.size());x.pool+=scratch;x.expert_active=true;audit(d);
 l.batch_calls++;l.batch_tokens+=batch.size();
 job(d,2,first,chosen,kernel(2,batch.size(),chosen),path,
  [this,d,chosen,batch](Path end){
   dies[d].expert_active=false;
   for(int tid:batch)expert_result(tid,d,chosen,end);
   try_expert(d);
  },scratch);
}
void Sim::expert_result(int tid,int d,int expert,Path path){
 auto &t=toks[tid];auto &g=t.groups.at(d);
 require(!g.contributions.test(expert),"Expert evaluated twice");
 g.contributions.set(expert);g.finished++;g.pending++;
 std::bitset<256> bits;bits.set(expert);
 if(d==t.owner){
  contribute(tid,d,bits,path,[this,tid,d](){toks[tid].groups.at(d).pending--;release_group(tid,d);try_expert(d);});
 }else if(p.lc){
  Path dep=later(g.path,path);
  vector_job(d,tid,g.finished>1?p.d:0,(g.finished>1?8:4)*p.d,4*p.d,dep,[this,tid,d](Path acc){
   auto &g=toks[tid].groups.at(d);g.path=acc;g.pending--;
   if(g.finished==g.expected){
    g.pending++;
    vector_job(d,tid,I(p.qops)*p.d+ceildiv(p.d,128),4*p.d,p.vbytes(p.d),acc,[this,tid,d](Path qp){
     auto &g=toks[tid].groups.at(d);
     message(tid,d,1u<<toks[tid].owner,RETURN,p.vbytes(p.d)+ceildiv(p.E+p.shared,8),g.contributions,qp);
    });
   }
   try_expert(d);
  });
 }else{
  vector_job(d,tid,I(p.qops)*p.d+ceildiv(p.d,128),4*p.d,p.vbytes(p.d),path,[this,tid,d,bits](Path qp){
   message(tid,d,1u<<toks[tid].owner,RETURN,p.vbytes(p.d)+ceildiv(p.E+p.shared,8),bits,qp);
  });
 }
}
void Sim::contribute(int tid,int d,std::bitset<256> bits,Path path,std::function<void()> release){
 auto &t=toks[tid];
 require((t.contributed&bits).none(),"Duplicate Combine contribution");t.contributed|=bits;
 t.incoming.push_back({d,bits,path,release});combine_next(tid);
}
void Sim::combine_next(int tid){
 auto &t=toks[tid];if(t.combining||t.incoming.empty())return;
 auto incoming=t.incoming.front();t.incoming.pop_front();t.combining=true;
 int owner=t.owner;
 vector_job(owner,tid,p.d,(incoming.source==owner?4*p.d:p.vbytes(p.d))+4*p.d,4*p.d,later(t.accum,incoming.path),
 [this,tid,incoming](Path acc){
  auto &t=toks[tid];t.accum=later(t.accum,acc);t.received+=incoming.bits.count();t.combining=false;
  if(incoming.release)incoming.release();
  require(t.received<=p.k+p.shared,"Too many contributions");
  if(t.received==p.k+p.shared){
   vector_job(t.owner,tid,I(p.qops)*p.d+ceildiv(p.d,128),4*p.d,p.vbytes(p.d),t.accum,[this,tid](Path qp){
    message(tid,toks[tid].owner,1u<<16,OUTPUT,p.vbytes(p.d),{},qp);
   });
  }else combine_next(tid);
 });
}
void Sim::release_group(int tid,int d){
 auto &t=toks[tid];auto &g=t.groups.at(d);
 if(g.released||!g.allocated||!g.arrived||!t.gate_done)return;
 if(g.finished!=g.expected||g.pending||(d==t.owner&&t.dispatch_pending))return;
 auto &x=dies[d];x.pool-=(p.lc&&d!=t.owner?8:4)*p.d;
 x.rx[0]-=g.rxheld;g.rxheld=0;g.released=true;audit(d);
 descriptor(d,-32*g.rxfrags);g.rxfrags=0;
 wake_networks();try_expert(d);
}
void Sim::message(int tid,int src,uint32_t mask,int kind,I body,std::bitset<256> bits,Path path){
 auto msg=std::make_shared<Message>();msg->frags=ceildiv(body,p.mtu);
 if(kind==DISPATCH)toks[tid].dispatch_pending++;
 for(int frag=0;frag<msg->frags;frag++){
  auto q=std::make_shared<Packet>();q->token=tid;q->src=src;q->mask=mask;q->kind=kind;
  q->med=(src==16||(mask&(1u<<16)))?0:1;q->bytes=std::min(I(p.mtu),body-I(frag)*p.mtu)+36;
  q->remaining=q->bytes*8*Q;q->frag=frag;q->frags=msg->frags;q->contributors=bits;q->path=path;q->message=msg;
  descriptor(src,32);
  pending_tx.push_back(q);
 }
 pump_tx();
}
void Sim::pump_tx(){
 if(pumping)return;pumping=true;
 std::vector<std::shared_ptr<Packet>> keep;
 for(auto q:pending_tx){
  int d=q->src,part=q->kind>=RETURN?1:0;
  if(d<16&&dies[d].tx[part]+q->bytes>p.txcap){l.tx_credit_blocks++;keep.push_back(q);continue;}
  if(d<16){dies[d].tx[part]+=q->bytes;audit(d);}
  // Serialize a body into the TX buffer and read it to the radio DMA.
  dma(d,q->bytes-36,q->bytes,q->path,[this,q](Path pack){
   dma(q->src,q->bytes,0,pack,[this,q](Path ready){
    q->path=ready;q->ready=now;q->queued=true;
    auto &flow=nets[q->med].flows[{q->src,q->mask,q->kind}];
    auto position=std::upper_bound(flow.begin(),flow.end(),q,[this](auto a,auto b){
     return std::make_tuple(a->ready,uid(a->token),a->frag)<std::make_tuple(b->ready,uid(b->token),b->frag);
    });
    flow.insert(position,q);
    auto &msg=*q->message;msg.prepared++;
    if(msg.prepared==msg.frags){
     if(q->kind==DISPATCH){toks[q->token].dispatch_pending--;release_group(q->token,q->src);}
     if(q->kind==RETURN){toks[q->token].groups.at(q->src).pending--;release_group(q->token,q->src);try_expert(q->src);}
    }
    wake(q->med);
   });
  });
 }
 pending_tx.swap(keep);pumping=false;
}
bool Sim::can_reserve(std::shared_ptr<Packet> q){
 if(q->reserved)return true;
 int part=q->kind>=RETURN?1:0;
 for(int d=0;d<16;d++)if(q->mask&(1u<<d)){
  auto &x=dies[d];if(x.rx[part]+q->bytes>p.rxcap){l.rx_credit_blocks++;return false;}
  if(q->kind==INPUT||q->kind==DISPATCH){
   auto &g=toks[q->token].groups.at(d);
   I need=(p.lc&&d!=toks[q->token].owner?8:4)*p.d;
   if(!g.allocated&&x.pool+need>p.poolcap-p.scratch(1)){l.partial_credit_blocks++;return false;}
  }
 }
 return true;
}
void Sim::reserve(std::shared_ptr<Packet> q){
 if(q->reserved)return;require(can_reserve(q),"Granted without RX/partial credit");q->reserved=true;
 int part=q->kind>=RETURN?1:0;
 for(int d=0;d<16;d++)if(q->mask&(1u<<d)){
  auto &x=dies[d];x.rx[part]+=q->bytes;q->message->rxbytes[d]+=q->bytes;
  descriptor(d,32);
  if(q->kind==INPUT||q->kind==DISPATCH){
   auto &g=toks[q->token].groups.at(d);
   if(!g.allocated){g.allocated=true;x.pool+=(p.lc&&d!=toks[q->token].owner?8:4)*p.d;}
   g.rxheld+=q->bytes;g.rxfrags++;
  }
  audit(d);
 }
}
void Sim::delivered(std::shared_ptr<Packet> q,Path path){
 int d=q->src;
 descriptor(d,-32);
 if(d<16){dies[d].tx[q->kind>=RETURN?1:0]-=q->bytes;audit(d);}
 l.air_bytes[q->kind]+=q->bytes;l.received_bytes+=q->bytes*__builtin_popcount(q->mask);l.packets++;
 for(int dest=0;dest<17;dest++)if(q->mask&(1u<<dest)){
  auto dep=path.until(path.t+p.prop,RADIO);dep=dep.until(align(dep.t),CLOCK).until(align(dep.t)+p.rxpipe,RXPIPE);
  at(dep.t,[this,q,dest,dep](){
   dma(dest,0,q->bytes,dep,[this,q,dest](Path stored){
    auto &msg=*q->message;msg.last[dest]=later(msg.last[dest],stored);msg.arrived[dest]++;
    if(msg.arrived[dest]!=msg.frags)return;
    Path end=msg.last[dest];int tid=q->token;
    if(q->kind==INPUT)gate(tid,end);
    else if(q->kind==DISPATCH)group_arrived(tid,dest,end);
    else if(q->kind==OUTPUT)token_done(tid,end);
    else contribute(tid,q->src,q->contributors,end,[this,q,dest](){
     dies[dest].rx[1]-=q->message->rxbytes[dest];descriptor(dest,-32*q->frags);audit(dest);wake_networks();
    });
   });
  });
 }
 pump_tx();wake_networks();
}
void Sim::token_done(int tid,Path path){
 auto &t=toks[tid];require(!t.done,"Token delivered twice");t.done=true;t.end=now;
 require(t.received==p.k+p.shared&&int(t.contributed.count())==p.k+p.shared,"Missing contributions");
 auto &x=dies[t.owner];x.inflight--;x.owner-=4*p.d;audit(t.owner);
 descriptor(t.owner,-32);
 completed++;last=later(last,path);
 if(p.decode){
  round_remaining--;
  if(round_remaining==0){rounds.back()[2]=now-rounds.back()[2];barrier=last;round++;}
 }
 inject();
}

void Sim::wake(int med){
 auto &net=nets[med];if(net.scheduled||net.flows.empty())return;
 bool any=false;for(auto &[key,q]:net.flows)if(!q.empty()){any=true;break;}
 if(!any)return;
 I start=std::max(now,net.available);
 if(!p.tw)start=ceildiv(start,p.frame_len(med))*p.frame_len(med);
 net.scheduled=true;at(start,[this,med](){nets[med].scheduled=false;frame(med);},3);
}
void Sim::frame(int med){
 auto &net=nets[med];
 struct Candidate{FlowKey key;std::shared_ptr<Packet> packet;uint32_t physical;int ru=0;I demand=0;};
 auto rank=[&](const std::shared_ptr<Packet>&q){
  return std::make_tuple(now-q->ready>=16*p.frame_len(med)?0:1,q->ready,q->kind>=RETURN?0:1,q->src,q->mask);
 };
 std::array<std::shared_ptr<Packet>,17> hol{};
 std::array<FlowKey,17> keys{};
 for(auto it=net.flows.begin();it!=net.flows.end();){
  if(it->second.empty()){it=net.flows.erase(it);continue;}
  auto q=it->second.front();
  if(q->ready<=now&&can_reserve(q)&&(!hol[q->src]||rank(q)<rank(hol[q->src]))){hol[q->src]=q;keys[q->src]=it->first;}
  ++it;
 }
 if(net.flows.empty())return;
 std::vector<Candidate> candidates;
 for(int src=0;src<17;src++)if(hol[src]){
  auto q=hol[src];uint32_t physical=q->mask;
  if(p.allrx&&q->kind==DISPATCH&&__builtin_popcount(q->mask)>1)physical=65535u^(1u<<src);
  candidates.push_back({keys[src],q,physical,0,0});
 }
 std::sort(candidates.begin(),candidates.end(),[&](auto &a,auto &b){return rank(a.packet)<rank(b.packet);});
 I control=p.control[med],flen=p.frame_len(med),data=flen-control-p.guard;
 if(candidates.empty()){
  // Packet-granular TDMA resumes on a credit/queue event, without a frame tail.
  if(p.tw)return;
  // Integrate repeated control-only frames until some state-changing event.
  require(!events.empty(),"No event can restore network credit");
  I count=std::max(I(1),ceildiv(events.top().t-now,flen));
  l.frame_count[med]+=count;l.control_ps[med]+=control*count;
  l.control_tx_ps+=control*count;l.control_rx_ps+=control*count*(med?15:16);
  net.available=now+flen*count;wake(med);return;
 }
 uint32_t txmask=0,rxmask=0;std::vector<Candidate> grant;
 for(auto c:candidates){
  if(!can_reserve(c.packet))continue;
  uint32_t src=1u<<c.packet->src;
  if((src&rxmask)||(c.physical&txmask))continue;
  reserve(c.packet);txmask|=src;rxmask|=c.physical;c.ru=p.ofdma?1:16;grant.push_back(c);
  if(!p.ofdma||grant.size()==16)break;
 }
 require(!grant.empty(),"No feasible grant despite candidates");require(!(txmask&rxmask),"Half-duplex violated");
 if(p.tw)data=ceildiv(grant[0].packet->remaining,p.rate(__builtin_popcount(grant[0].packet->mask)>1,16));
 for(auto &c:grant){
  I rx[16][2],pool[16];for(int d=0;d<16;d++){rx[d][0]=dies[d].rx[0];rx[d][1]=dies[d].rx[1];pool[d]=dies[d].pool;}
  std::set<std::pair<int,int>> newgroups;
  for(auto q:net.flows.at(c.key)){
   if(q->ready>now)break;bool valid=true;int part=q->kind>=RETURN?1:0;
   if(!q->reserved)for(int d=0;d<16;d++)if(q->mask&(1u<<d)){
    if(rx[d][part]+q->bytes>p.rxcap){valid=false;break;}
    if(q->kind==INPUT||q->kind==DISPATCH){
     auto &g=toks[q->token].groups.at(d);auto group=std::make_pair(q->token,d);
     if(!g.allocated&&!newgroups.count(group)&&pool[d]+(p.lc&&d!=toks[q->token].owner?8:4)*p.d>p.poolcap-p.scratch(1)){valid=false;break;}
    }
   }
   if(!valid)break;
   c.demand+=q->remaining;
   if(!q->reserved)for(int d=0;d<16;d++)if(q->mask&(1u<<d)){
    rx[d][part]+=q->bytes;
    if(q->kind==INPUT||q->kind==DISPATCH){
     auto group=std::make_pair(q->token,d);
     if(!toks[q->token].groups.at(d).allocated&&!newgroups.count(group)){
      pool[d]+=(p.lc&&d!=toks[q->token].owner?8:4)*p.d;newgroups.insert(group);
     }
    }
   }
   if(p.tw)break;
  }
 }
 if(p.ofdma)for(int left=16-grant.size();left>0;left--){
  int best=-1;
  for(int i=0;i<(int)grant.size();i++){
   auto &g=grant[i];
   if(g.demand<=p.rate(__builtin_popcount(g.packet->mask)>1,g.ru)*data)continue;
   if(best<0||(__int128)g.demand*grant[best].ru>(__int128)grant[best].demand*g.ru)best=i;
  }
  if(best<0)break;grant[best].ru++;
 }
 l.frame_count[med]++;l.control_ps[med]+=control;l.data_window_ps[med]+=data;
 l.control_tx_ps+=control;l.control_rx_ps+=control*(med?15:16);
 I start=now+control+p.guard,used_ru=0;I txend[17]{},rxend[17]{};
 l.available_ru_ps+=16*data;
 for(auto &g:grant){
  used_ru+=g.ru;auto &flow=net.flows.at(g.key);I current=start;
  I rate=p.rate(__builtin_popcount(g.packet->mask)>1,g.ru);
  while(!flow.empty()&&current<start+data){
   auto q=flow.front();if(q->ready>now||!can_reserve(q))break;
   reserve(q);
   I duration=std::min(start+data-current,ceildiv(q->remaining,rate));
   l.network_wait_ps+=now-q->path.t;
   I supplied=std::min(q->remaining,duration*rate);q->remaining-=supplied;
   Path dep=later(q->path,net.last).until(now,WAIT).until(now+control,CONTROL).until(start,CLOCK).until(current,WAIT);
   dep=dep.until(current+duration,RADIO);q->path=dep;current+=duration;
   if(q->remaining==0){
    flow.pop_front();at(current,[this,q,dep](){delivered(q,dep);});
    if(p.tw)break;
   }else break;
  }
  I busy=current-start;l.ru_ps+=busy*g.ru;l.tail_ru_ps+=(data-busy)*g.ru;
  txend[g.packet->src]=std::max(txend[g.packet->src],busy);
  for(int d=0;d<17;d++)if(g.physical&(1u<<d))rxend[d]=std::max(rxend[d],busy);
 }
 require(used_ru<=16,"RU oversubscription");
 for(int d=0;d<17;d++){
  require(!(txend[d]&&rxend[d]),"Same group TX and RX overlap");
  l.tx_ps+=txend[d];l.rx_ps+=rxend[d];
  if(p.ofdma)l.ofdma_ps+=txend[d]+rxend[d];
 }
 Path framepath=net.last;
 for(auto &g:grant)framepath=later(framepath,g.packet->path);
 net.available=p.tw?start+data:now+flen;
 net.last=framepath.until(net.available,WAIT);wake(med);
}

py::dict ledger_dict(const Sim&s,int cohort){
 auto &l=s.l;py::dict r;
 r["cohort"]=cohort;r["duration_ps"]=s.now;r["tokens"]=s.toks.size();
 r["matrix_ops"]=l.matrix_ops;r["vector_ops"]=l.vector_ops;
 r["read_bytes"]=l.read_bytes;r["write_bytes"]=l.write_bytes;
 r["tx_ps"]=l.tx_ps;r["rx_ps"]=l.rx_ps;r["ofdma_ps"]=l.ofdma_ps;
 r["control_tx_ps"]=l.control_tx_ps;r["control_rx_ps"]=l.control_rx_ps;
 r["air_input_bytes"]=l.air_bytes[0];r["air_dispatch_bytes"]=l.air_bytes[1];
 r["air_return_bytes"]=l.air_bytes[2];r["air_output_bytes"]=l.air_bytes[3];
 r["received_bytes"]=l.received_bytes;r["ru_ps"]=l.ru_ps;
 r["available_ru_ps"]=l.available_ru_ps;r["tail_ru_ps"]=l.tail_ru_ps;
 r["batch_calls"]=l.batch_calls;r["batch_tokens"]=l.batch_tokens;r["events"]=l.events;r["packets"]=l.packets;
 r["compute_wait_ps"]=l.compute_wait_ps;r["dma_wait_ps"]=l.dma_wait_ps;r["network_wait_ps"]=l.network_wait_ps;r["injection_wait_ps"]=l.injection_wait_ps;
 r["rx_credit_blocks"]=l.rx_credit_blocks;r["partial_credit_blocks"]=l.partial_credit_blocks;r["tx_credit_blocks"]=l.tx_credit_blocks;
 r["fanout_sum"]=l.full_group_fanout;r["base_frames"]=l.frame_count[0];r["peer_frames"]=l.frame_count[1];
 r["base_control_ps"]=l.control_ps[0];r["peer_control_ps"]=l.control_ps[1];
 for(int i=0;i<NCAT;i++)r[py::str(std::string("critical_")+CAT[i]+"_ps")]=s.last.parts[i];
 auto arr=[](const I *a){return std::vector<I>(a,a+16);};
 r["die_compute_ps"]=arr(l.cpu_ps);r["die_dma_ps"]=arr(l.dma_ps);
 r["peak_rx_bytes"]=arr(l.max_rx);r["peak_tx_bytes"]=arr(l.max_tx);r["peak_partial_bytes"]=arr(l.max_pool);
 r["peak_owner_bytes"]=arr(l.max_owner);r["peak_tile_bytes"]=arr(l.max_tile);
 r["peak_control_bytes"]=arr(l.max_control);
 r["resource_audit_passed"]=true;r["token_conservation_passed"]=true;
 return r;
}
py::dict simulate(py::dict config,py::dict model,py::array_t<int,py::array::c_style|py::array::forcecast> array,
                  int batch,bool decode,const std::string& baseline){
 require(batch>0,"Batch must be positive");Params p(config,model,baseline,decode);
 auto a=array.unchecked<2>();require(a.shape(1)==p.k+4,"Invalid trace width");
 std::map<int,std::vector<std::vector<int>>> cohorts;
 for(ssize_t i=0;i<a.shape(0);i++){std::vector<int> r;for(ssize_t j=0;j<a.shape(1);j++)r.push_back(a(i,j));cohorts[r[0]/batch].push_back(r);}
 py::list ledgers,tokens,rounds;
 for(auto &[cid,desc]:cohorts){
  Sim s(p,desc);{py::gil_scoped_release unlock;s.run();}
  ledgers.append(ledger_dict(s,cid));
  for(auto &t:s.toks)tokens.append(py::make_tuple(cid,t.req,t.pos,t.row,t.word,t.owner,t.ready,t.end));
  for(auto &r:s.rounds)rounds.append(py::make_tuple(cid,r[0],r[1],r[2]));
 }
 py::dict result;result["cohorts"]=ledgers;result["tokens"]=tokens;result["rounds"]=rounds;return result;
}
PYBIND11_MODULE(_native,m){
 m.doc()="Integer-picosecond event kernel; Python owns configuration, inputs, statistics and audits.";
 m.def("simulate",&simulate);
}
