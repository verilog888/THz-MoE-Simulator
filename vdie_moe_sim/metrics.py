"""Per-token, per-request, and per-cohort metrics with an independent byte audit."""
import math
import numpy as np
import pandas as pd
from .compute import expected_matrix_ops
from .packets import expected_bytes
from .energy import energy

TOKEN_COLUMNS=['cohort','request_ordinal','step','row_id','token_id','owner','ready_ps','completed_ps']

def tables(result,c,mid,descriptors,baseline):
    tokens=pd.DataFrame(result['tokens'],columns=TOKEN_COLUMNS)
    tokens['latency_s']=(tokens.completed_ps-tokens.ready_ps)*1e-12
    if len(tokens)!=len(descriptors) or tokens.duplicated(['request_ordinal','step']).any():
        raise ValueError('Token count or identity conservation failed')
    ledgers=pd.DataFrame(result['cohorts'])
    if int(ledgers.matrix_ops.sum())!=expected_matrix_ops(c['models'][mid],len(descriptors)):
        raise ValueError('Independent matrix-OP audit failed')
    for name,value in expected_bytes(c,mid,descriptors,baseline).items():
        if int(ledgers[name].sum())!=value:
            raise ValueError(f'Independent wire byte audit failed: {name}')
    critical=[x for x in ledgers if x.startswith('critical_')]
    if not np.array_equal(ledgers[critical].sum(axis=1),ledgers.duration_ps):
        raise ValueError('Dependency path must sum to cohort duration')
    return ledgers,tokens,pd.DataFrame(result['rounds'],columns=['cohort','step','active_requests','duration_ps'])

def summarize(ledgers,tokens,rounds,c,mode):
    requests=[]
    for (cohort,req),t in tokens.groupby(['cohort','request_ordinal'],sort=True):
        latency=t.completed_ps.max()*1e-12 if mode=='prefill' else t.latency_s.mean()
        requests.append(dict(cohort=int(cohort),request_ordinal=int(req),row_id=int(t.row_id.iloc[0]),
                             actual_tokens=len(t),request_layer_latency_s=float(latency)))
    requests=pd.DataFrame(requests);units=[]
    for l in ledgers.to_dict('records'):
        cid=l['cohort'];t=tokens[tokens.cohort==cid];rq=requests[requests.cohort==cid]
        unit=dict(cohort=cid,tokens=len(t),requests=len(rq),duration_s=l['duration_ps']*1e-12,
                  request_layer_latency_s=float(rq.request_layer_latency_s.mean()),
                  token_mean_s=float(t.latency_s.mean()),
                  token_p50_s=float(t.latency_s.quantile(.5)),token_p95_s=float(t.latency_s.quantile(.95)))
        if mode=='decode':
            rr=rounds[rounds.cohort==cid]
            unit.update(round_mean_s=float(rr.duration_ps.mean()*1e-12),rounds=len(rr),
                        active_B_mean=float(rr.active_requests.mean()),active_B_min=int(rr.active_requests.min()))
        unit.update({f'energy_{k}_j':v for k,v in energy(l,c).items()})
        units.append(unit)
    units=pd.DataFrame(units)
    summary=dict(tokens=len(tokens),requests=len(requests),cohorts=len(units),
                 mean_request_layer_s=float(requests.request_layer_latency_s.mean()),
                 p50_request_layer_s=float(requests.request_layer_latency_s.quantile(.5)),
                 p95_request_layer_s=float(requests.request_layer_latency_s.quantile(.95)),
                 mean_token_layer_s=float(tokens.latency_s.mean()),
                 p50_token_layer_s=float(tokens.latency_s.quantile(.5)),
                 p95_token_layer_s=float(tokens.latency_s.quantile(.95)),
                 sum_cohort_s=float(units.duration_s.sum()),
                 mean_cohort_s=float(units.duration_s.mean()))
    for column in units:
        if column.startswith('energy_'):summary[column]=float(units[column].sum())
    for mode_e in ['frontend_proxy','baseband_stress']:
        summary[f'{mode_e}_uj_token']=summary[f'energy_{mode_e}_j']*1e6/len(tokens)
    summary['mean_expert_batch']=float(ledgers.batch_tokens.sum()/ledgers.batch_calls.sum())
    summary['different_remote_dies_per_token']=float(ledgers.fanout_sum.sum()/len(tokens))
    summary['ru_utilization_data_window']=float(ledgers.ru_ps.sum()/max(ledgers.available_ru_ps.sum(),1))
    summary['ru_utilization_cohort_window']=float(ledgers.ru_ps.sum()/max(32*ledgers.duration_ps.sum(),1))
    for name in ['matrix_ops','vector_ops','read_bytes','write_bytes','air_input_bytes','air_dispatch_bytes',
                 'air_return_bytes','air_output_bytes','received_bytes','events','packets','tx_ps','rx_ps',
                 'control_rx_ps','ofdma_ps','base_frames','peer_frames','tail_ru_ps','compute_wait_ps',
                 'dma_wait_ps','network_wait_ps','injection_wait_ps','rx_credit_blocks','tx_credit_blocks','partial_credit_blocks']:
        summary[name]=int(ledgers[name].sum())
    for name in ['peak_rx_bytes','peak_tx_bytes','peak_partial_bytes','peak_owner_bytes','peak_tile_bytes','peak_control_bytes']:
        summary[name]=int(max(max(x) for x in ledgers[name]))
    if mode=='decode':
        summary.update(mean_round_s=float(rounds.duration_ps.mean()*1e-12),
                       actual_rounds=len(rounds),minimum_active_B=int(rounds.active_requests.min()))
    return summary,units,requests
