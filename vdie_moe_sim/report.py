"""Rebuild statistical comparisons, scientific figures and the final Word report."""
from pathlib import Path
import itertools
import json
import math
import re
import numpy as np
import pandas as pd
from .config import ROOT
from .statistics import paired_effects

SCHEMES=['T_P_N','T_B_N','T_P_L','T_B_L','O_P_N','O_B_N','O_P_L','O_B_L']
COLORS=['#555555','#9c755f','#af7aa1','#edc948','#4e79a7','#76b7b2','#f28e2b','#e15759']
METRICS=[('mean_request_layer_s',1000,'Mean request layer time (ms)'),
         ('frontend_proxy_uj_token',1,'Frontend proxy (microjoules/token)'),
         ('baseband_stress_uj_token',1,'Baseband stress (microjoules/token)')]

def aggregate(raw):
    rows=[]
    keys=['family','model','mode','batch','axis','value','baseline']
    for key,g in raw.groupby(keys,dropna=False,sort=False):
        row=dict(zip(keys,key));row['seeds']=len(g)
        for name in ['mean_request_layer_s','p50_request_layer_s','p95_request_layer_s',
                     'mean_token_layer_s','p50_token_layer_s','mean_cohort_s','p95_token_layer_s','mean_expert_batch',
                     'different_remote_dies_per_token','ru_utilization_data_window','ru_utilization_cohort_window']:
            row[name]=g[name].mean()
            row[name+'_seed_std']=g[name].std(ddof=1)
        for mode in ['frontend_proxy','baseband_stress']:
            row[mode+'_uj_token']=g['energy_'+mode+'_j'].sum()/g.tokens.sum()*1e6
        for name in [x for x in g if x.startswith('energy_')]:
            row[name+'_per_token']=g[name].sum()/g.tokens.sum()
        for name in ['air_dispatch_bytes','air_return_bytes','received_bytes','read_bytes','write_bytes',
                     'matrix_ops','vector_ops','ofdma_ps','rx_ps','control_rx_ps']:
            row[name+'_per_token']=g[name].sum()/g.tokens.sum()
        rows.append(row)
    return pd.DataFrame(rows)

def comparisons(c,raw,units):
    rows=[];cache={}
    main=units[units.family=='main']
    def compare(group,baseline,variant,label,model,mode,batch,axis='',value=''):
        a=group[group.baseline==baseline];b=group[group.baseline==variant]
        key=(model,mode,batch,axis,str(value),baseline,variant)
        if key not in cache:
            cache[key]=paired_effects(a,b,[c['simulation']['seed_list'][0],'bootstrap',model,mode,int(batch),axis,str(value)],
                                      c['experiment']['bootstrap_resamples'],c['experiment']['confidence'])
        rows.append(dict(comparison=label,model=model,mode=mode,batch=batch,axis=axis,value=value,
                         baseline=baseline,variant=variant,**cache[key]))
    for (model,mode,batch),g in main.groupby(['model','mode','batch']):
        for variant in SCHEMES:
            compare(g,'T_P_N',variant,'vs_T_P_N',model,mode,batch)
        for letter,options,position in [('ofdma',('T','O'),0),('broadcast',('P','B'),1),('local_combine',('N','L'),2)]:
            for baseline in SCHEMES:
                parts=baseline.split('_')
                if parts[position]!=options[0]:continue
                parts[position]=options[1]
                compare(g,baseline,'_'.join(parts),letter,model,mode,batch)
    strong=units[units.family.isin(['main','strong_tdma'])&(units.batch==32)]
    for (model,mode),g in strong.groupby(['model','mode']):
        compare(g,'TW_P_N','O_P_N','vs_strong_tdma',model,mode,32)
        compare(g,'TW_B_L','O_B_L','vs_strong_tdma',model,mode,32)
    sensitivity=units[units.family.isin(['sensitivity','extra_vector_sensitivity'])]
    for (model,mode,axis,value),g in sensitivity.groupby(['model','mode','axis','value']):
        for baseline,variant in [('T_P_N','O_P_N'),('T_B_L','O_B_L')]:
            compare(g,baseline,variant,'sensitivity_ofdma',model,mode,32,axis,value)
    return pd.DataFrame(rows)

def interactions(raw):
    rows=[];main=raw[raw.family=='main']
    for (model,mode,batch,seed),g in main.groupby(['model','mode','batch','seed']):
        g=g.set_index('baseline')
        for metric in ['mean_request_layer_s','frontend_proxy_uj_token','baseband_stress_uj_token']:
            for subset in [(0,1),(0,2),(1,2),(0,1,2)]:
                total=0.
                for b in SCHEMES:
                    levels=[1 if b[0]=='O' else -1,1 if '_B_' in b else -1,1 if b.endswith('_L') else -1]
                    total+=math.prod(levels[x] for x in subset)*math.log(float(g.loc[b,metric]))
                rows.append(dict(model=model,mode=mode,batch=batch,seed=seed,metric=metric,
                                 factors='*'.join(['OFDMA','Broadcast','LocalCombine'][x] for x in subset),
                                 log_factorial_contrast=total/4,multiplicative_contrast=math.exp(total/4)))
    return pd.DataFrame(rows)

def interaction_summary(c,units):
    from .routing import rng_for
    subsets=[(0,1),(0,2),(1,2),(0,1,2)]
    levels=np.asarray([[1 if b[0]=='O' else -1,1 if '_B_' in b else -1,
                        1 if b.endswith('_L') else -1] for b in SCHEMES])
    weights=np.stack([np.prod(levels[:,subset],axis=1) for subset in subsets],axis=1)
    columns=['request_layer_latency_s','energy_frontend_proxy_j','energy_baseband_stress_j']
    rows=[];alpha=(1-c['experiment']['confidence'])/2
    for (model,mode,batch),g in units[units.family=='main'].groupby(['model','mode','batch']):
        paired=[]
        for baseline in SCHEMES:
            t=g[g.baseline==baseline].sort_values(['seed','cohort'])
            identity=t[['seed','cohort','tokens']].to_numpy()
            if paired and not np.array_equal(identity,reference):raise ValueError('Unpaired factorial workloads')
            reference=identity;paired.append(t[columns].to_numpy())
        values=np.stack(paired,axis=1)
        groups=[values[reference[:,0]==seed] for seed in np.unique(reference[:,0])]
        def effects(mean):return weights.T@np.log(mean)/4
        point=effects(values.mean(axis=0));samples=[]
        rng=rng_for([c['simulation']['seed_list'][0],'bootstrap',model,mode,int(batch),'factorial','all_metrics'])
        for _ in range(c['experiment']['bootstrap_resamples']):
            total=np.zeros((8,3));n=0
            for index in rng.integers(0,len(groups),size=len(groups)):
                a=groups[index];selected=a[rng.integers(0,len(a),size=len(a))]
                total+=selected.sum(axis=0);n+=len(selected)
            samples.append(effects(total/n))
        bounds=np.quantile(samples,[alpha,1-alpha],axis=0)
        for i,subset in enumerate(subsets):
            for j,metric in enumerate(['mean_request_layer_s','frontend_proxy_uj_token','baseband_stress_uj_token']):
                rows.append(dict(model=model,mode=mode,batch=int(batch),metric=metric,
                         factors='*'.join(['OFDMA','Broadcast','LocalCombine'][x] for x in subset),
                         log_factorial_contrast=float(point[i,j]),log_low=float(bounds[0,i,j]),log_high=float(bounds[1,i,j]),
                         multiplicative_contrast=float(np.exp(point[i,j])),
                         multiplicative_low=float(np.exp(bounds[0,i,j])),multiplicative_high=float(np.exp(bounds[1,i,j])),
                         paired_seeds=len(groups),paired_cohorts=len(reference)))
    return pd.DataFrame(rows)

def break_even(c,raw):
    rows=[]
    # Use the matched sample at each seed. Negative thresholds remain visible.
    for (model,mode,batch,seed),g in raw[raw.family.isin(['main','strong_tdma'])].groupby(['model','mode','batch','seed']):
        g=g.set_index('baseline')
        for baseline in ['T_P_N','T_B_N','T_P_L','T_B_L','TW_P_N','TW_B_L']:
            variant='O'+baseline[baseline.index('_'):]
            if baseline not in g.index or variant not in g.index:continue
            a,b=g.loc[baseline],g.loc[variant]
            numerator=(a.energy_frontend_proxy_j+a.energy_online_eq_j-b.energy_frontend_proxy_j-b.energy_online_eq_j)
            denominator=b.ofdma_ps*1e-12
            rows.append(dict(model=model,mode=mode,batch=batch,seed=seed,baseline=baseline,variant=variant,
                             max_ofdma_extra_w_per_group=float(numerator/denominator) if denominator else None,
                             fixed_equalizer_w=c['energy']['equalizer_extra_w_per_rx']))
    return pd.DataFrame(rows)

def fanout_comparison(c,raw):
    rows=[]
    selected=raw[(raw.family=='main')&(raw.batch==32)&(raw.baseline=='T_P_N')]
    for (mid,mode),g in selected.groupby(['model','mode']):
        m=c['models'][mid];e,k=m['routed_experts'],m['top_k']
        hit=1-math.comb(e-e//16,k)/math.comb(e,k)
        theoretical=1+14*hit if m['shared_experts'] else 15*hit
        observed=float((g.different_remote_dies_per_token*g.tokens).sum()/g.tokens.sum())
        rows.append(dict(model=mid,mode=mode,uniform_expected_remote_dies=theoretical,
                         observed_remote_dies=observed,actual_tokens=int(g.tokens.sum()),shared_buddy_included=bool(m['shared_experts'])))
    return pd.DataFrame(rows)

def scheduler_metrics(c,raw):
    from .config import override
    from .run_all import read_plan
    from .network import control_ps
    plan={r['plan_id']:r for r in read_plan()};rows=[]
    for r in raw.to_dict('records'):
        effective=override(c,plan[r['plan_id']]['overrides']);ma,ra=effective['mac'],effective['radio']
        base,peer=control_ps(effective,17),control_ps(effective,16)
        guard=ma['guard_cycles']*5000;nominal=ma['frame_cycles']*5000
        bf,pf=int(r['base_frames']),int(r['peer_frames']);grants=bf+pf
        control=base*bf+peer*pf
        if int(r['control_rx_ps'])!=16*base*bf+15*peer*pf:
            raise ValueError('Recorded control activity disagrees with frame counts')
        packet=r['baseline'].startswith('TW')
        base_frame=max(nominal,base+guard+5000);peer_frame=max(nominal,peer+guard+5000)
        allocated=control+guard*grants+r['tx_ps'] if packet else bf*base_frame+pf*peer_frame
        air=[]
        for nodes,window in [(17,base),(16,peer)]:
            serial=8*nodes*ma['queue_report_bytes']/(ra['ptp_raw_bps']*ra['rate_scale'])
            serial+=8*(ma['grant_header_bytes']+16*ma['grant_bytes_per_ru'])/(ra['broadcast_raw_bps']*ra['rate_scale'])
            air.append(serial*1e12/window)
        rows.append(dict(plan_id=r['plan_id'],model=r['model'],mode=r['mode'],batch=r['batch'],seed=r['seed'],
                         baseline=r['baseline'],axis=r['axis'],value=r['value'],schedule='packet_grant' if packet else 'frame',
                         nominal_frame_ps=nominal,base_control_ps_per_grant=base,peer_control_ps_per_grant=peer,
                         effective_base_frame_ps=None if packet else base_frame,effective_peer_frame_ps=None if packet else peer_frame,
                         base_control_air_utilization=air[0],peer_control_air_utilization=air[1],
                         allocated_schedule_ps=allocated,mean_frame_or_grant_service_ps=allocated/grants,
                         control_fraction_of_allocated_schedule=control/allocated))
    return pd.DataFrame(rows)

def make_figures(out,agg,paired,breaks,critical,factor_summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.dpi':180,'svg.hashsalt':'VDIE-SPEC-1.0'})
    figures=out/'figures';index=[]
    def save(fig,name,data,caption):
        fig.tight_layout()
        fig.savefig(figures/f'{name}.png',bbox_inches='tight',metadata={'Software':'THz-MoE-Simulator'})
        fig.savefig(figures/f'{name}.svg',bbox_inches='tight',metadata={'Date':None})
        data.to_csv(figures/f'{name}.csv',index=False)
        plt.close(fig);index.append(dict(name=name,caption=caption))
    main=agg[agg.family=='main']
    for (model,mode),g in main.groupby(['model','mode']):
        fig,axes=plt.subplots(1,3,figsize=(13.5,3.6))
        for ax,(metric,scale,label) in zip(axes,METRICS):
            for b,color in zip(SCHEMES,COLORS):
                t=g[g.baseline==b].sort_values('batch')
                ax.plot(t.batch.astype(str),t[metric]*scale,'o-',ms=4,label=b,color=color,lw=1.4)
            ax.set_xlabel('Concurrent requests B');ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].set_title(f'{model} / {mode}')
        fig.legend(*axes[-1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,0),ncol=8,fontsize=10)
        save(fig,f'absolute_{model}_{mode}',g,f'{model} {mode}：八组合绝对延迟与两种能量账本；能量按实际完成token归一。')
        norm=paired[(paired.comparison=='vs_T_P_N')&(paired.model==model)&(paired['mode']==mode)]
        fig,axes=plt.subplots(1,3,figsize=(13.5,3.6))
        for ax,metric,label in zip(axes,['speedup','frontend_saving','stress_saving'],
                                   ['Speedup vs T_P_N','Frontend energy saving','Stress energy saving']):
            for b,color in zip(SCHEMES,COLORS):
                t=norm[norm.variant==b].sort_values('batch')
                ax.errorbar(t.batch.astype(str),t[metric],
                            yerr=np.maximum(0,np.vstack([t[metric]-t[metric+'_low'],t[metric+'_high']-t[metric]])),
                            marker='o',ms=3,lw=1,label=b,color=color,capsize=2)
            ax.axhline(1 if metric=='speedup' else 0,color='black',lw=.6,ls='--')
            if metric!='speedup':ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xlabel('Concurrent requests B');ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].set_title(f'{model} / {mode} / paired 95% CI')
        fig.legend(*axes[-1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,0),ncol=8,fontsize=10)
        save(fig,f'normalized_{model}_{mode}',norm,f'{model} {mode}：相对T_P_N的配对加速比及节能率，误差线为分层bootstrap 95%区间。')
    # All single-factor axes, with the default point taken from the main experiment.
    sensitivity=agg[agg.family.isin(['sensitivity','extra_vector_sensitivity'])]
    baseline4=['T_P_N','O_P_N','T_B_L','O_B_L']
    from .config import load_spec
    defaults=load_spec()
    for axis,gs in sensitivity.groupby('axis'):
        node=defaults
        for k in axis.split('.'):node=node[k]
        base=main[(main.batch==32)&main.baseline.isin(baseline4)].copy()
        base['axis']=axis;base['value']=str(node)
        allrows=pd.concat([gs,base],ignore_index=True)
        try:levels=sorted(allrows.value.astype(str).unique(),key=float)
        except ValueError:levels=sorted(allrows.value.astype(str).unique())
        for metric,scale,label in METRICS:
            fig,axes=plt.subplots(2,3,figsize=(12.5,7))
            for mi,model in enumerate(['M1','M2','M3']):
                for wi,mode in enumerate(['prefill','decode']):
                    ax=axes[wi,mi]
                    for b,color in zip(baseline4,[COLORS[0],COLORS[4],COLORS[3],COLORS[7]]):
                        t=allrows[(allrows.model==model)&(allrows['mode']==mode)&(allrows.baseline==b)].set_index('value')
                        t.index=t.index.astype(str);t=t.reindex(levels)
                        ax.plot(range(len(levels)),t[metric]*scale,'o-',label=b,color=color,ms=4)
                    ax.set_xticks(range(len(levels)),levels,rotation=15,fontsize=10)
                    ax.set_title(f'{model} / {mode} B32');ax.set_ylabel(label);ax.grid(alpha=.2)
            fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,0),ncol=4,fontsize=10)
            fig.suptitle(axis,fontsize=12)
            save(fig,'sensitivity_'+axis.replace('.','_')+'_'+metric,allrows,f'单因素敏感性：{axis}；指标：{label}；其余参数维持默认。')
    # Mechanism effects at B32: each mechanism has all four conditional contrasts.
    for mechanism in ['ofdma','broadcast','local_combine']:
        g=paired[(paired.comparison==mechanism)&(paired.batch==32)]
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        low=min(1,float(g.speedup_low.min()));high=max(1,float(g.speedup_high.max()))
        margin=max(.001,(high-low)*.12)
        for mi,model in enumerate(['M1','M2','M3']):
            for wi,mode in enumerate(['prefill','decode']):
                ax=axes[wi,mi];t=g[(g.model==model)&(g['mode']==mode)].copy()
                x=np.arange(len(t))
                ax.errorbar(x,t.speedup,yerr=np.maximum(0,np.vstack([t.speedup-t.speedup_low,t.speedup_high-t.speedup])),
                            fmt='o',color=COLORS[mi+4],capsize=4)
                ax.set_xticks(x,[a+'\nto '+b for a,b in zip(t.baseline,t.variant)],fontsize=10)
                ax.axhline(1,color='black',ls='--',lw=.8);ax.set_title(f'{model} / {mode} B32');ax.set_ylabel('Paired speedup')
                ax.set_ylim(low-margin,high+margin);ax.grid(axis='y',alpha=.2)
        save(fig,'mechanism_'+mechanism,g,f'{mechanism}：其它两项固定时的四组条件效应；95%区间；不能将三机制百分比直接相加。')
    g=paired[paired.comparison=='vs_strong_tdma']
    fig,axes=plt.subplots(1,2,figsize=(12,3.8))
    low=min(1,float(g.speedup_low.min()));high=max(1,float(g.speedup_high.max()))
    margin=max(.001,(high-low)*.12)
    for ax,mode in zip(axes,['prefill','decode']):
        t=g[g['mode']==mode];x=np.arange(len(t))
        ax.errorbar(x,t.speedup,yerr=np.maximum(0,np.vstack([t.speedup-t.speedup_low,t.speedup_high-t.speedup])),fmt='o',color='#4e79a7',capsize=4)
        ax.set_xticks(x,[m+'\n'+v for m,v in zip(t.model,t.variant)],fontsize=10)
        ax.axhline(1,color='black',ls='--',lw=.8);ax.set_title(mode+' B32: OFDMA vs TW');ax.set_ylabel('Paired speedup')
        ax.set_ylim(low-margin,high+margin);ax.grid(axis='y',alpha=.2)
    save(fig,'strong_tdma',g,'强TDMA反证：O_P_N/O_B_L分别相对包粒度TW_P_N/TW_B_L。')
    g=breaks[breaks.batch==32].groupby(['model','mode','baseline'],as_index=False).max_ofdma_extra_w_per_group.mean()
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,mode in zip(axes,['prefill','decode']):
        t=g[g['mode']==mode]
        ax.bar(np.arange(len(t)),t.max_ofdma_extra_w_per_group,color='#4e79a7')
        ax.set_xticks(np.arange(len(t)),[m+'\n'+b for m,b in zip(t.model,t.baseline)],rotation=45,ha='right',fontsize=10)
        ax.axhline(0,color='black',lw=.8);ax.axhline(.40896428571428567,color='#e15759',ls='--',label='Stress proxy')
        ax.set_title(mode+' B32');ax.set_ylabel('Break-even extra W / active group');ax.legend(fontsize=10)
    save(fig,'ofdma_power_threshold',g,'在线EQ维持20mW时的OFDMA额外基带功耗盈亏阈值；负值表示零额外功耗也不能节能。')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,mode in zip(axes,['prefill','decode']):
        t=critical[critical['mode']==mode]
        labels=[m+'\n'+b for m,b in zip(t.model,t.baseline)];bottom=np.zeros(len(t))
        for name in [x for x in t if x.startswith('critical_')]:
            y=t[name].to_numpy()/t.duration_ps.to_numpy()
            ax.bar(range(len(t)),y,bottom=bottom,label=name.removeprefix('critical_').removesuffix('_ps'));bottom+=y
        ax.set_xticks(range(len(t)),labels,fontsize=10);ax.set_title(mode+' B32: dependency path');ax.set_ylabel('Fraction of cohort duration')
    fig.legend(*axes[-1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,0),ncol=5,fontsize=10)
    save(fig,'dependency_path',critical,'依赖路径包含计算、DMA、无线、控制与显式等待；分项闭合到cohort完成时间，区别于可重叠的资源busy-time。')
    g=main[main.batch==32]
    fig,axes=plt.subplots(2,3,figsize=(13.5,7))
    parts=[('Compute/vector',['compute','vector']),('eDRAM R/W',['edram_read','edram_write']),
           ('Refresh',['refresh']),('RF',['rf_lo','rf_tx','rf_rx']),('Idle/leak',['control_idle','memory_leak']),
           ('Online EQ',['online_eq']),('OFDMA baseband',['ofdma_extra'])]
    for mi,model in enumerate(['M1','M2','M3']):
        for wi,mode in enumerate(['prefill','decode']):
            ax=axes[wi,mi];t=g[(g.model==model)&(g['mode']==mode)].set_index('baseline').reindex(SCHEMES)
            bottom=np.zeros(8)
            for label,columns in parts:
                y=sum(t['energy_'+x+'_j_per_token'] for x in columns).to_numpy()*1e6
                ax.bar(range(8),y,bottom=bottom,label=label);bottom+=y
            ax.set_xticks(range(8),SCHEMES,rotation=60,fontsize=10);ax.set_ylabel('microjoules/token');ax.set_title(f'{model} / {mode} B32')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,0),ncol=4,fontsize=10)
    save(fig,'energy_components',g,'B32下的能耗分项；刷新、在线均衡和OFDMA基带压力分别保留，不隐藏负收益。')
    fig,axes=plt.subplots(2,3,figsize=(13.5,7))
    for mi,model in enumerate(['M1','M2','M3']):
        for wi,mode in enumerate(['prefill','decode']):
            ax=axes[wi,mi];t=g[(g.model==model)&(g['mode']==mode)].set_index('baseline').reindex(SCHEMES)
            dispatch=t.air_dispatch_bytes_per_token.to_numpy()/1000
            returns=t.air_return_bytes_per_token.to_numpy()/1000
            ax.bar(range(8),dispatch,label='Dispatch',color='#4e79a7')
            ax.bar(range(8),returns,bottom=dispatch,label='Return',color='#f28e2b')
            ax.set_xticks(range(8),SCHEMES,rotation=60,fontsize=10)
            ax.set_title(f'{model} / {mode} B32');ax.set_ylabel('Peer air kB/token')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,0),ncol=2,fontsize=10)
    save(fig,'communication_bytes',g,'通信局部效果：Dispatch与Return空口字节；广播只按一次发射计量，含包头、CRC和scale；Base输入/输出另记。')
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,mode in zip(axes,['prefill','decode']):
        t=critical[critical['mode']==mode];x=np.arange(len(t))
        ax.bar(x-.18,t.compute_utilization,width=.36,label='Compute',color='#4e79a7')
        ax.bar(x+.18,t.dma_utilization,width=.36,label='DMA',color='#f28e2b')
        ax.set_xticks(x,[m+'\n'+b for m,b in zip(t.model,t.baseline)],fontsize=10)
        ax.set_title(mode+' B32');ax.set_ylabel('Busy / (16 dies x cohort duration)');ax.legend(fontsize=10)
        ax.set_ylim(0,1)
    save(fig,'resource_utilization',critical,'资源busy-time按16个Die可用时间归一。计算与DMA可重叠，因此两项不相加解释整层延迟。')
    g=factor_summary[factor_summary.batch==32]
    factors=['OFDMA*Broadcast','OFDMA*LocalCombine','Broadcast*LocalCombine','OFDMA*Broadcast*LocalCombine']
    cases=[(model,mode) for model in ['M1','M2','M3'] for mode in ['prefill','decode']]
    fig,axes=plt.subplots(1,3,figsize=(13.5,4.8))
    for ax,(metric,_,label) in zip(axes,METRICS):
        t=g[g.metric==metric].set_index(['model','mode','factors'])
        values=np.asarray([[100*(t.loc[(model,mode,factor),'multiplicative_contrast']-1) for factor in factors]
                           for model,mode in cases])
        extent=max(float(np.abs(values).max()),.01)
        im=ax.imshow(values,cmap='RdBu_r',vmin=-extent,vmax=extent,aspect='auto')
        for i,(model,mode) in enumerate(cases):
            for j,factor in enumerate(factors):
                r=t.loc[(model,mode,factor)];star='*' if r.log_low>0 or r.log_high<0 else ''
                ax.text(j,i,f'{values[i,j]:+.2f}{star}',ha='center',va='center',fontsize=9,
                        color='white' if abs(values[i,j])>.55*extent else 'black')
        ax.set_xticks(range(4),['O x B','O x L','B x L','O x B x L'],rotation=30,ha='right')
        ax.set_yticks(range(6),[model+' '+mode for model,mode in cases],fontsize=10)
        ax.set_title(label.split(' (')[0],fontsize=11)
        fig.colorbar(im,ax=ax,shrink=.7,label='100 x (exp(effect) - 1)')
    save(fig,'factorial_interactions',g,'二阶与三阶交互：标准2³因子效应的乘性形式；星号表示配对95%区间不含零效应。数值不是可相加的单机制节能率。')
    (figures/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    return index

def critical_table(out,raw):
    rows=[]
    selected=raw[(raw.family=='main')&(raw.batch==32)&raw.baseline.isin(['T_P_N','O_B_L'])]
    for (model,mode,baseline),g in selected.groupby(['model','mode','baseline']):
        ledgers=pd.concat([pd.read_parquet(out/'runs'/id/'ledger.parquet') for id in g.plan_id],ignore_index=True)
        row=dict(model=model,mode=mode,baseline=baseline,duration_ps=int(ledgers.duration_ps.sum()))
        row.update({col:int(ledgers[col].sum()) for col in ledgers if col.startswith('critical_')})
        for kind,column in [('compute','die_compute_ps'),('dma','die_dma_ps')]:
            row[kind+'_busy_ps']=int(sum(sum(a) for a in ledgers[column]))
            row[kind+'_utilization']=row[kind+'_busy_ps']/(16*row['duration_ps'])
        rows.append(row)
    return pd.DataFrame(rows)

def build_report(out,c,raw=None,cohort=None,manifest=None):
    out=Path(out)
    if raw is None:raw=pd.read_csv(out/'raw_runs.csv',keep_default_na=False)
    if cohort is None:cohort=pd.read_parquet(out/'cohort_metrics.parquet')
    if manifest is None:manifest=json.loads((out/'manifest.json').read_text())
    if not manifest['experiments_complete'] or len(raw)!=4020 or raw.plan_id.nunique()!=4020:
        raise ValueError('The final report requires all 4020 approved records')
    from .run_all import validate_tests
    # Include tests added for the completed statistical/reporting implementation.
    manifest['validation']=validate_tests(out)
    manifest['complete']=manifest['validation']['all_acceptance_passed']
    from .audit import audit_results
    manifest['offline_audit']=audit_results(out,c,raw,manifest)
    if not manifest['offline_audit']['passed']:
        raise ValueError('Offline audit failed')
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    agg=aggregate(raw);agg.to_csv(out/'aggregate.csv',index=False)
    paired=comparisons(c,raw,cohort);paired.to_csv(out/'paired_effects.csv',index=False)
    interaction=interactions(raw);interaction.to_csv(out/'factorial_interactions.csv',index=False)
    factor_summary=interaction_summary(c,cohort);factor_summary.to_csv(out/'factorial_summary.csv',index=False)
    breaks=break_even(c,raw);breaks.to_csv(out/'ofdma_power_threshold.csv',index=False)
    fanout_comparison(c,raw).to_csv(out/'fanout_validation.csv',index=False)
    scheduler_metrics(c,raw).to_csv(out/'scheduler_metrics.csv',index=False)
    critical=critical_table(out,raw)
    critical.to_csv(out/'resource_and_dependency_metrics.csv',index=False)
    figures=make_figures(out,agg,paired,breaks,critical,factor_summary)
    from .document import write_documents
    write_documents(out,c,raw,agg,paired,manifest,figures)
