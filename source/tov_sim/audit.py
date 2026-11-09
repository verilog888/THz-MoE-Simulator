"""Independent full-run checks from frozen routes, not the event callbacks."""
from collections import Counter
from functools import lru_cache
import csv
import gzip
import json
import math
from pathlib import Path
from .config import qbytes,ceildiv,resolve,stable_hash
from .io import read_gzip,write_json
from .simulator import Simulator


def full_audit(c,out,repeat=True):
    out=Path(out)
    with (out/'run_manifest.csv').open(encoding='utf-8-sig') as f:plan=list(csv.DictReader(f))
    traces=json.loads((out/'route_manifest.json').read_text());catalog={(t['model'],t['phase'],t['profile'],t['seed']):t for t in traces}
    @lru_cache(maxsize=36)
    def routes(name):
        with gzip.open(out/'routes'/name,'rt') as h:return [json.loads(line) for line in h]
    verified=0;tokens=0;repeat_checks=[];errors=[];physical_runs=0
    for row in plan:
        if row['reference'] or row['family']=='energy':continue
        cc=resolve(c,json.loads(row['overrides']));m=next(m for m in c['models'] if m['id']==row['model']);n=cc['hardware']['dies']
        data=read_gzip(out/'runs'/(row['run_id']+'.json.gz'));records=routes(catalog[row['model'],row['phase'],row['profile'],int(row['seed'])]['file']);B=int(row['B'])
        header=c['packet']['header_bytes']+c['packet']['crc_bytes'];q=qbytes(c,m['d']);ret=header+q+ceildiv(m['E']+m['shared'],8)
        for cohort,raw in enumerate(data['cohorts']):
            chosen=[r for r in records if cohort*B<=r['request_rank']<(cohort+1)*B];tx=Counter();rx=Counter();hops=0;receivers=0;useful=0
            layer=row['measurement']=='layer';wireless=not row['method'].startswith('TSV');lc=not row['method'].endswith('NO_LC')
            for token in chosen:
                owner=token['request_rank']%n;dest=Counter(e%n for e in token['expert_ids'])
                for sid in range(m['shared']):dest[owner if owner%2==sid else owner^1]+=1
                remote={d:k for d,k in dest.items() if d!=owner};useful+=2*m['d']*len(remote)
                if layer:
                    for kind in ['INPUT','OUTPUT']:
                        tx[kind]+=header+q;rx[kind]+=header+q;hops+=1;receivers+=1
                sizes=[header+q+c['packet']['expert_entry_bytes']*k for k in remote.values()]
                if remote:
                    if wireless and 'BCAST' in row['method']:
                        size=header+q+c['packet']['expert_entry_bytes']*sum(remote.values());tx['DISPATCH']+=size;rx['DISPATCH']+=len(remote)*size;hops+=1;receivers+=len(remote)
                    elif not wireless and '_MC_' in row['method']:
                        size=header+q+c['packet']['expert_entry_bytes']*sum(remote.values());total=size+sum(sizes);tx['DISPATCH']+=total;rx['DISPATCH']+=total;hops+=1+len(remote);receivers+=1+len(remote)
                    else:
                        factor=1 if wireless else 2;total=factor*sum(sizes);tx['DISPATCH']+=total;rx['DISPATCH']+=total;hops+=factor*len(remote);receivers+=factor*len(remote)
                count=len(remote) if lc else sum(remote.values());factor=1 if wireless else 2;total=count*factor*ret
                tx['RETURN']+=total;rx['RETURN']+=total;hops+=count*factor;receivers+=count*factor
            ledger=raw['ledger'];prefix=f'{row["run_id"]}/{cohort}'
            assert raw['tokens']==len(chosen) and raw['contributor_checks']==len(chosen),prefix
            assert raw['useful_bytes']==useful,prefix
            for kind in ['INPUT','OUTPUT','DISPATCH','RETURN']:assert ledger.get(kind+'_tx_bits',0)==tx[kind]*8,(prefix,kind,ledger.get(kind+'_tx_bits'),tx[kind]*8)
            assert ledger['DMA_bytes']==sum(tx.values())+sum(rx.values()),(prefix,'DMA')
            assert ledger['data_rx_bits']==8*sum(rx.values()),(prefix,'RX')
            assert raw['packet_hops']==hops and raw['receiver_deliveries']==receivers,(prefix,'deliveries')
            expected_mac=len(chosen)*(m['d']*m['E']+3*m['d']*m['h']*(m['k']+m['shared'])) if layer else 0
            assert ledger.get('MAC',0)==expected_mac,(prefix,'MAC')
            if layer:assert sum(int(k)*v for k,v in raw['batch_histogram'].items())==len(chosen)*(m['k']+m['shared']),prefix
            else:assert not raw['batch_histogram'] and not ledger.get('nmp_busy_ps',0),prefix
            for busy in raw['nmp_busy_ps_by_node']+raw['dma_busy_ps_by_node']:assert 0<=busy<=raw['total_ps'],(prefix,'busy')
            assert sum(raw['nmp_busy_ps_by_node'])==ledger.get('nmp_busy_ps',0),prefix
            assert sum(raw['dma_busy_ps_by_node'])==ledger['dma_busy_ps'],prefix
            assert len(raw['token_timings'])==len(chosen),prefix
            assert len({(a[0],a[1]) for a in raw['token_timings']})==len(chosen),prefix
            # INPUT/OUTPUT are in the Base domain; replay has none.
            if wireless:
                expected_control=0
                for domain in [0,1]:
                    ctrl=8*((n+(domain==0))*c['radio']['report_bytes']+c['radio']['grant_base_bytes']+c['radio']['rus']*c['radio']['grant_bytes_per_ru'])*ledger.get(f'domain_{domain}_frames',0)
                    assert ctrl==ledger.get(f'domain_{domain}_control_bits',0),prefix
                    expected_control+=ctrl
                assert ledger['physical_tx_bits']==sum(tx.values())*8+expected_control,prefix
                assert ledger.get('domain_0_data_bits',0)==8*(tx['INPUT']+tx['OUTPUT']),prefix
                assert ledger.get('domain_1_data_bits',0)==8*(tx['DISPATCH']+tx['RETURN']),prefix
            else:assert ledger['physical_tx_bits']==sum(tx.values())*8==ledger['tsv_bits'],prefix
            assert math.isclose(sum(a['ledger']['energy_proxy_J'] for a in raw['rows']),ledger['energy_proxy_J'],rel_tol=1e-12),prefix
            if repeat and cohort==0 and row['family']=='main' and int(row['B'])==8 and row['phase']=='decode' and row['seed']=='11' and row['profile']=='hot_shift' and row['model']=='ernie_21b_a3b':
                again=Simulator(cc,m,row['method'],row['measurement'],row['phase'],chosen,cohort).run()
                assert stable_hash(again)==stable_hash(raw),(prefix,'determinism')
                repeat_checks.append(dict(run_id=row['run_id'],cohort=cohort,result_sha256=stable_hash(again)))
            verified+=1;tokens+=len(chosen)
        physical_runs+=1
    result=dict(passed=True,independently_checked_physical_runs=physical_runs,cohorts=verified,tokens=tokens,
                checks=['analytic packet bytes per kind and hop','every broadcast receiver and TSV multicast outlet','complete-packet endpoint DMA bytes','exact model MAC','batch token conservation','owner/contributor and token uniqueness','per-Die NMP/DMA busy time bounds','per-domain data/control bit conservation','round energy increments'],
                deterministic_full_cohort_replays=repeat_checks)
    write_json(out/'independent_audit.json',result)
    validation=json.loads((out/'validation.json').read_text());validation['independent_audit']=result;write_json(out/'validation.json',validation)
    return result
