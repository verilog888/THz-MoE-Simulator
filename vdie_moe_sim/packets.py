from .config import vector_bytes
from .placement import contributions

def wire_bytes(body, c):
    mtu=c['mac']['mtu_payload_bytes']
    return body+((body+mtu-1)//mtu)*(c['mac']['packet_header_bytes']+c['mac']['crc_bytes'])

def expected_bytes(c,mid,descriptors,baseline):
    m=c['models'][mid];v=vector_bytes(m['d_model'],c);broadcast='_B_' in baseline;local=baseline.endswith('_L')
    result=dict(air_input_bytes=0,air_dispatch_bytes=0,air_return_bytes=0,air_output_bytes=0,received_bytes=0)
    for row in descriptors:
        groups=contributions(m,int(row[0])%16,row[4:]);owner=int(row[0])%16
        result['air_input_bytes']+=wire_bytes(v,c);result['air_output_bytes']+=wire_bytes(v,c)
        result['received_bytes']+=2*wire_bytes(v,c)
        remote={d:es for d,es in groups.items() if d!=owner}
        if broadcast and remote:
            w=wire_bytes(v+8*sum(map(len,remote.values())),c)
            result['air_dispatch_bytes']+=w;result['received_bytes']+=len(remote)*w
        elif remote:
            for es in remote.values():
                w=wire_bytes(v+8*len(es),c)
                result['air_dispatch_bytes']+=w;result['received_bytes']+=w
        returns=len(remote) if local else sum(map(len,remote.values()))
        w=returns*wire_bytes(v+(m['routed_experts']+m['shared_experts']+7)//8,c)
        result['air_return_bytes']+=w;result['received_bytes']+=w
    return result
