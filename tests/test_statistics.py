import numpy as np
import pandas as pd
import pytest
from vdie_moe_sim.statistics import paired_effects

def samples():
    rows=[]
    for seed in range(5):
        for cohort in range(4):
            x=1+seed*7+cohort
            rows.append(dict(seed=seed,cohort=cohort,tokens=x*11,
                             request_layer_latency_s=x,
                             energy_frontend_proxy_j=x*3,
                             energy_baseband_stress_j=x*5))
    a=pd.DataFrame(rows);b=a.copy()
    b['request_layer_latency_s']/=2
    b['energy_frontend_proxy_j']*=.75
    b['energy_baseband_stress_j']*=1.2
    return a,b

def test_paired_resampling_preserves_known_ratios_and_negative_saving():
    a,b=samples();r=paired_effects(a,b,['test'],1000)
    for name,expected in [('speedup',2),('frontend_saving',.25),('stress_saving',-.2)]:
        for suffix in ['', '_low','_high']:
            assert r[name+suffix]==pytest.approx(expected,abs=1e-12)
    assert r['paired_seeds']==5 and r['paired_cohorts']==20

def test_pairing_is_by_identity_not_dataframe_order():
    a,b=samples()
    assert paired_effects(a,b,['test'])==paired_effects(a,b.sample(frac=1,random_state=2),['test'])

def test_missing_or_changed_workload_rejected():
    a,b=samples()
    with pytest.raises(ValueError,match='Unpaired'):paired_effects(a,b.iloc[:-1],['test'])
    b.loc[0,'tokens']+=1
    with pytest.raises(ValueError,match='Different paired workload'):paired_effects(a,b,['test'])

def test_ratio_is_aggregate_ratio_not_average_individual_ratio():
    a,b=samples();b['request_layer_latency_s']=a.request_layer_latency_s+10
    r=paired_effects(a,b,['test'],1000)
    assert r['speedup']==pytest.approx(a.request_layer_latency_s.mean()/b.request_layer_latency_s.mean())
    assert r['speedup']!=pytest.approx((a.request_layer_latency_s/b.request_layer_latency_s).mean())
    assert r['speedup_low']<r['speedup']<r['speedup_high']

def test_factorial_contrast_is_orthogonal_and_preserves_pairing(config):
    from vdie_moe_sim.report import SCHEMES,interaction_summary
    rows=[]
    for seed in range(5):
        for cohort in range(3):
            for baseline in SCHEMES:
                x=1 if baseline[0]=='O' else -1;y=1 if '_B_' in baseline else -1
                common=1+seed+cohort
                rows.append(dict(family='main',model='M1',mode='prefill',batch=32,seed=seed,cohort=cohort,
                                 tokens=common*10,baseline=baseline,
                                 request_layer_latency_s=common*np.exp(.05*x*y),
                                 energy_frontend_proxy_j=common*np.exp(-.02*x*y),
                                 energy_baseband_stress_j=common*np.exp(.03*x*y)))
    result=interaction_summary(config,pd.DataFrame(rows))
    expected={'mean_request_layer_s':.1,'frontend_proxy_uj_token':-.04,'baseband_stress_uj_token':.06}
    for r in result.to_dict('records'):
        value=expected[r['metric']] if r['factors']=='OFDMA*Broadcast' else 0
        for column in ['log_factorial_contrast','log_low','log_high']:
            assert r[column]==pytest.approx(value,abs=1e-12)
