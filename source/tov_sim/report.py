"""Figures and Chinese Markdown/Word report derived from measured output only."""
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import re
import numpy as np
os.environ.setdefault('MPLCONFIGDIR','/tmp/tov-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import ROOT,derive
from .io import write_csv,write_json

LABELS=['TSV-P2P','TSV-MC (NORM)','TSV-MC (HIGH)','TDMA-P2P','TDMA-BC','OFDMA-P2P','OFDMA-BC no LC','ToV']


def table(rows,fields):
    def value(v):return f'{v:.6g}' if isinstance(v,float) else str(v)
    lines=['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']
    lines.extend('| '+' | '.join(value(r.get(k,'')) for k in fields)+' |' for r in rows)
    return '\n'.join(lines)


def parameters(c):
    meanings={
    'schema_version':'配置格式版本；用于拒绝不兼容配置',
    'hardware.dies':'V-Die 数量；Base 另计一个端点','hardware.dram_bytes_per_die':'每 Die DRAM 容量，byte',
    'hardware.banks_per_die':'每 Die 内存 bank 数','hardware.pe_per_die':'每 Die NMP PE 数量','hardware.mac_rows':'每 PE MAC 阵列行数','hardware.mac_cols':'每 PE MAC 阵列列数',
    'hardware.clock_hz':'NMP 主频，Hz；本规范恢复为 1 GHz','hardware.hb_pins_per_bank':'每 bank 的 HB pin 数','hardware.hb_pin_gbps':'每 HB pin 的速率，Gb/s',
    'hardware.compute_efficiency':'矩阵计算效率，乘峰值 OP/s','hardware.kernel_memory_efficiency':'kernel 分区内部有效访存比例','hardware.dma_bandwidth_fraction':'HB 峰值固定分配给网络 DMA 的比例',
    'hardware.vector_elements_per_cycle':'逐元素操作吞吐，元素/周期','hardware.kernel_startup_ns':'每算子任务启动时间，ns',
    'hardware.staging_bytes_per_die':'每 Die 有限 DRAM staging 空间，byte','hardware.base_staging_bytes':'Base staging 空间，byte','hardware.base_dma_GBps':'Base 串行 DMA 带宽，十进制 GB/s',
    'precision.weight':'权重存储格式','precision.activation':'线上向量格式','precision.accumulation':'内部累加格式','precision.weight_block':'权重量化二维 block 边长','precision.activation_block':'向量量化 block 元素数','precision.scale_bytes':'每个量化 scale 的字节数',
    'execution.owner_window':'每 owner 最多同时在途 token 数','execution.expert_batch_max':'每专家微批 token 上限','execution.batch_wait_ns':'专家队列最老输入的聚批等待上限，ns',
    'packet.header_bytes':'每包固定包头 byte，含目的 mask','packet.crc_bytes':'每包 CRC byte','packet.expert_entry_bytes':'每个 expert ID/gate 目录项 byte',
    'radio.groups_per_die':'独立频谱域数；Base-facing 与 peer-facing','radio.unicast_gbps':'每域共享的单播原始总速率，Gb/s','radio.broadcast_gbps':'每域共享的多目的广播原始总速率，Gb/s','radio.carrier_GHz':'载波频率记录，GHz；不用于倍乘数字带宽',
    'radio.frame_ns':'全局对齐调度帧长，ns','radio.guard_ns':'帧保护/TDMA 帧内换向保护时间，ns','radio.controller_ns':'控制仲裁时间代理，ns','radio.report_bytes':'每源 HOL flow 报告 byte','radio.grant_base_bytes':'固定 grant byte','radio.grant_bytes_per_ru':'每 RU grant byte','radio.rus':'每域资源单元总数','radio.fft_bins':'FFT bin 数','radio.data_bins':'携带数据的 bin 数','radio.cp_bins':'循环前缀长度','radio.endpoint_pipeline_ns':'每包每端点流水时间，ns','radio.ofdm_extra_endpoint_ns':'OFDM 每端点附加流水时间，ns','radio.rate_scale':'单播和广播速率共同缩放系数',
    'tsv.normalized_total_gbps':'NORM 所有端口聚合原始设计预算，Gb/s；不是 TSV 技术上限','tsv.high_bw_total_GBps':'HIGH 强有线聚合带宽情景，GB/s','tsv.payload_efficiency':'TSV payload 服务效率','tsv.turnaround_ns':'半双工端口换向时间，ns','tsv.base_route_ns':'Base 完整收包后的转发处理时间，ns','tsv.endpoint_pipeline_ns':'每跳每端点流水时间，ns',
    'workload.dataset':'公开文本数据集 ID','workload.split':'数据集划分','workload.requests_per_seed':'每种子无放回请求数','workload.seeds':'PCG64 抽样与路由种子','workload.batch_sizes':'每 cohort 并发请求数 B','workload.phases':'Prefill 与 teacher-forced 单层 Decode','workload.prefill_token_cap':'每请求最大输入 token 数','workload.decode_step_cap':'每请求最大 Decode 轮数','workload.routing_profiles':'受控合成路由分布','workload.zipf_alpha':'hot_shift 排名权重指数','workload.prefill_hot_epoch_tokens':'Prefill 每个热集周期的逻辑 token 数','workload.decode_hot_epoch_steps':'Decode 每个热集周期的轮数',
    'primary_baseline':'主要对照方法','primary_proposal':'完整 ToV 方案','methods':'全部八个互连/数据流对照',
    'energy.mac_pJ':'每有效 MAC 的代理能耗，pJ','energy.hb_pJ_per_bit':'HB 接口每访问 bit 代理能耗，pJ','energy.dram_core_pJ_per_bit':'DRAM core 每访问 bit 代理能耗，pJ','energy.scalar_pJ_per_op':'每标量操作代理能耗，pJ','energy.tsv_pJ_per_bit_hop':'TSV 每实际 bit-hop 代理能耗，pJ','energy.rf_lo_mW_per_node':'每节点两域共享 LO 功率，mW','energy.rf_tx_mW':'活动域 TX 电路功率，mW','energy.rf_rx_mW':'活动域 RX 电路功率，mW','energy.ofdm_W_per_active_endpoint':'每活动端点/域 OFDM/EQ 基带功率代理，W','energy.background_W_per_die':'每 Die 背景功率代理，W','energy.base_W':'Base 背景功率代理，W',
    'experiment.bootstrap_samples':'分层配对 bootstrap 重采样次数','experiment.bootstrap_seed':'bootstrap 固定 PCG64 种子',
    'sensitivity.model':'敏感性分析模型 ID','sensitivity.B':'敏感性分析并发请求数','sensitivity.routing_profile':'敏感性分析路由分布','sensitivity.tsv_budget_multipliers':'TSV NORM 聚合预算倍数，单轴变化','sensitivity.thz_rate_scales':'THz 速率缩放点；控制时间同步重算','sensitivity.die_counts':'配对节点数敏感性点','sensitivity.memory_efficiencies':'kernel 访存效率敏感性点；DMA 分区不变','sensitivity.ofdm_endpoint_power_W':'基带功率代理重计价点，W；不重跑时序'}
    modelmean={'id':'模型内部标识','repo':'官方 Hugging Face 仓库','layer_index':'所模拟 MoE 层索引','d':'隐藏向量维度','h':'每 routed/shared 分支中间维度','E':'routed 专家总数','k':'每 token routed Top-k 数','shared':'每 token 额外共享分支数','expert_bias':'专家矩阵是否含 bias','router_bias':'Router 是否含 bias','activation_rule':'模型对应的门控非线性规则'}
    result=[]
    def walk(value,key=''):
        if isinstance(value,dict):
            for name,v in value.items():walk(v,(key+'.' if key else '')+name)
        elif key=='models':
            for i,m in enumerate(value):
                for name,v in m.items():result.append(dict(parameter=f'models[{i}].{name}',value=json.dumps(v,ensure_ascii=False),meaning=modelmean[name]))
        else:
            if key not in meanings:raise ValueError('Parameter meaning missing: '+key)
            result.append(dict(parameter=key,value=json.dumps(value,ensure_ascii=False),meaning=meanings[key]))
    walk(c);return result


def plots(c,out,absolute,mechanisms,sens):
    directory=out/'figures';directory.mkdir(exist_ok=True);images=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'savefig.dpi':160,'svg.hashsalt':'ToV-MoE-minimal-1.0'})
    def save(name):
        plt.tight_layout();plt.savefig(directory/(name+'.png'));plt.savefig(directory/(name+'.svg'),metadata={'Date':None});plt.close();images.append(name)
    methods=c['methods'];lookup={(a['model'],a['phase'],a['B'],a['profile'],a['method']):a for a in absolute}
    for metric,name,label in [('T_ns','layer_latency','Single-layer latency / TSV_NORM_MC_LC'),('G_Gbps','d2d_goodput','Useful D2D goodput / TSV_NORM_MC_LC')]:
        fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(8)
        for i,phase in enumerate(['prefill','decode']):
            values=[]
            for method in methods:
                rr=[a[metric]/lookup[a['model'],a['phase'],a['B'],a['profile'],c['primary_baseline']][metric] for a in absolute if a['phase']==phase and a['method']==method]
                values.append(np.exp(np.log(rr).mean()))
            ax.bar(x+(i-.5)*.36,values,.36,label=phase)
        ax.axhline(1,color='black',lw=.8);ax.set_xticks(x,LABELS,rotation=25,ha='right');ax.set_ylabel(label);ax.legend();ax.set_title('Equal case weights; controlled routing; separate layer/replay measurements');save(name)
    fig,ax=plt.subplots(figsize=(10,4.4));rr=[r for r in mechanisms if r['phase']=='all'];x=np.arange(len(rr))
    ax.bar(x-.18,[r['SG'] for r in rr],.36,label='D2D throughput ratio');ax.bar(x+.18,[r['ST'] for r in rr],.36,label='Layer latency speedup')
    ax.set_xticks(x,[r['mechanism'].replace('_',' ') for r in rr],rotation=20,ha='right');ax.axhline(1,color='black',lw=.8);ax.legend();ax.set_title('Paired mechanism comparisons; values below 1 retain regressions');save('mechanisms')
    for axis in ['tsv_budget','memory_efficiency','dies','thz_rate']:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8))
        for ax,metric in zip(axes,['SG','ST']):
            for phase in ['prefill','decode']:
                rr=sorted([r for r in sens if r['axis']==axis and r['phase']==phase],key=lambda r:r['value'])
                ax.plot([r['value'] for r in rr],[r[metric] for r in rr],marker='o',label=phase)
            ax.axhline(1,color='black',lw=.8);ax.set_xlabel(axis.replace('_',' '));ax.set_ylabel(metric+' (ToV vs TSV NORM)');ax.legend()
        fig.suptitle('Qwen3; B=8; hot_shift; 3 seeds; one-axis measured points');save('sensitivity_'+axis)
    from .metrics import ENERGY
    bycase=defaultdict(lambda:defaultdict(float))
    with (out/'energy_breakdown.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['family']!='main':continue
            acc=bycase[r['case_id'],r['method']];acc['tokens']+=int(r['tokens'])
            for key in ENERGY:acc[key]+=float(r[key])
    fig,ax=plt.subplots(figsize=(11,5));bottom=np.zeros(8)
    for key in ENERGY:
        values=np.array([np.mean([v[key]/v['tokens']*1e6 for (case,method),v in bycase.items() if method==m]) for m in methods])
        ax.bar(np.arange(8),values,bottom=bottom,label=key[:-2]);bottom+=values
    ax.set_xticks(np.arange(8),LABELS,rotation=25,ha='right');ax.set_ylabel('Energy proxy (uJ/token), equal case mean');ax.legend(ncol=5,fontsize=8);ax.set_title('Activity-based proxy; no validated chip-energy claim');save('energy_proxy_components')
    return images


def render(c,out,summaries,absolute,mechanisms,sensitivity,energy_rows,validation):
    params=parameters(c);write_csv(out/'simulation_parameters.csv',params)
    images=plots(c,out,absolute,mechanisms,sensitivity);d=derive(c)
    headline=[r for r in summaries if r['scope'] in ['overall','phase']]
    headline_display=[dict(phase=r['group'],baseline=r['baseline'],SG_95CI=f'{r["SG"]:.4f} [{r["SG_ci_low"]:.4f}, {r["SG_ci_high"]:.4f}]',ST_95CI=f'{r["ST"]:.4f} [{r["ST_ci_low"]:.4f}, {r["ST_ci_high"]:.4f}]',latency_reduction_percent=r['latency_reduction_percent']) for r in headline]
    main=next(r for r in headline if r['scope']=='overall' and r['baseline']==c['primary_baseline'])
    high=next(r for r in headline if r['scope']=='overall' and r['baseline']=='TSV_HIGH_MC_LC')
    inp=json.loads((out/'input_manifest.json').read_text());v=validation['full_experiment'];run=json.loads((out/'execution.json').read_text())
    abstract=[]
    for phase in ['all','prefill','decode']:
        row=next(r for r in headline if r['group']==phase and r['baseline']==c['primary_baseline'])
        abstract.append(f'{phase}: compared with TSV_NORM_MC_LC at the same 264 Gb/s aggregate raw-link budget, ToV achieves {row["SG"]:.4f}x effective D2D useful-delivery throughput (95% paired CI {row["SG_ci_low"]:.4f}–{row["SG_ci_high"]:.4f}); single-MoE-layer latency is {1/row["ST"]:.4f} of the baseline (ST={row["ST"]:.4f}, 95% CI {row["ST_ci_low"]:.4f}–{row["ST_ci_high"]:.4f}; reduction {row["latency_reduction_percent"]:.2f}%).')
    abstract.append(f'Against TSV_HIGH_MC_LC (896 GB/s aggregate), overall SG={high["SG"]:.4f}, ST={high["ST"]:.4f}. Controlled synthetic routes and ideal, retransmission-free radio reachability are architectural assumptions. These are separate communication replay and single-layer simulations, not pretrained-model accuracy or full-model inference results.')
    (out/'abstract_result.txt').write_text('\n\n'.join(abstract)+'\n',encoding='utf-8')
    crossing=[]
    for phase in c['workload']['phases']:
        rr=sorted([r for r in sensitivity if r['axis']=='tsv_budget' and r['phase']==phase],key=lambda r:r['value'])
        for metric in ['SG','ST']:
            brackets=[f'[{a["value"]:g}, {b["value"]:g}]×' for a,b in zip(rr,rr[1:]) if (a[metric]-1)*(b[metric]-1)<=0]
            crossing.append(f'{phase} 的 {metric}：'+('跨过 1 的已测预算区间为 '+', '.join(brackets) if brackets else '所测预算点间未出现跨过 1 的区间')+'。')
    numerical=next(x['details'] for x in validation['checks'] if x['name']=='fp32_math_FP8_reference_shared_GPT_activation')
    source_spec=(ROOT/'ToV_MoE_MINIMAL_SIM_SPEC.md').read_text();parameter_sources=source_spec[source_spec.index('| hardware.dies'):source_spec.index('## 3.')]
    lines=[
        '# ToV-MoE 完整仿真报告',
        '## 1. 仿真目标',
        '按上传的 ToV_MoE_MINIMAL_SIM_SPEC.md，分别衡量有效 D2D 交付吞吐率和完整单 MoE 层闭环延迟，并检查广播、OFDMA、Local Combine 的独立作用及适用边界。NORM 采用相同聚合 raw-link 预算；HIGH 提供强有线情景。',
        '## 2. 仿真简述',
        f'默认 16 个 V-Die、每 Die 4 GiB DRAM、1 GHz、FP8 E4M3FN/FP32 累加。每节点峰值 {d["peak_ops_per_second"]/1e12:g} TOPS，总峰值 {d["system_peak_ops_per_second"]/1e12:g} TOPS；HB 峰值 {d["hb_bytes_per_second"]/1e9:g} GB/s。kernel 与 DMA 带宽分别为 {d["kernel_bytes_per_second"]/1e9:g} / {d["dma_bytes_per_second"]/1e9:g} GB/s。此版本完全采用新规范的频率和路由定义。',
        f'使用三个官方 config 与 AutoTokenizer。数据集原始 {inp["original_rows"]} 行，过滤去重后 {inp["clean_rows"]} 行；每种子 32 条相同请求，3 个种子；不加载模型权重。响应只决定 teacher-forced Decode 长度。路由为冻结 PCG64 生成的 uniform/hot_shift 受控 Top-k，不能解释为真实 Router 的实测行为。',
        table([dict(model=m['id'],d=m['d'],h=m['h'],E=m['E'],k=m['k'],shared=m['shared']) for m in c['models']],['model','d','h','E','k','shared']),
        '## 3. 仿真参数表与含义',
        '所有可配置参数和含义均列于下表及 simulation_parameters.csv；config_resolved.json 记录派生量、官方结构、软件版本和规范 SHA-256。MB/GB 使用十进制，MiB/GiB 使用二进制。',
        table(params,['parameter','value','meaning']),
        '参数来源与 P/A/D 分类沿用上传规范。以下是规范中的硬件与设计来源表；P 为论文锚点，A 为架构代理，D 为设计选择，并非本次流片测量。',
        '| 参数 | 数值 | 含义/来源 |\n|---|---|---|\n'+parameter_sources.strip(),
        '## 4. 仿真流程',
        '冻结小型输入及官方仓库 revision → 按固定请求次序生成 36 个路由 trace → 单位、容量、因果与数学验证 → 576 个完整层配置 + 576 个通信回放配置 → 单轴敏感性 → 能耗账本重计价 → 2000 次分层配对 bootstrap → 输出 CSV、图表与本文档。',
        '事件日历采用整数 ps 的 Python heapq。完成/释放先于新工作和仲裁；一个 Die 只有一个 NMP 引擎，Combine 优先，其次 Router，最后已满足条件的最老专家批。专家批不超过 8，等待上限 200 ns；每 owner 最多 4 个在途 token。所有 32/B 个 cohort 均执行，Decode 每轮的全部输出到达 Base 后才开始下一轮。',
        'Base 输入 → owner Router → 远端 Dispatch → 专家微批 → Local Combine/逐专家 Return → owner 最终 Combine → Base 输出。通信回放移除 Base 输入/输出及算子时间，保留相同路由、DMA、窗口、包长、调度和完整返回依赖。',
        '实现中明确的细节：Prefill 在固定请求序列间按 token 位置轮转 admission，再进入每 owner FIFO；hot_shift 使用正方向循环移位排名。Local Combine 将同 Die 的 FP32 输出求和并量化融合为一个向量任务；最终 Combine 直接读取本地 FP32 和远端 FP8 向量，在寄存器中反量化并做 FP32 求和，避免不存在的 DRAM 中间副本。端点 DMA 处理完整包，单独计费。全部输入、FP32 输出、待发和接收包均计入 staging。',
        'TSV 使用 N 条可并行半双工星形端口，完整包经 Base 存储转发；MC 在源端发送一份，Base 出口逐份收费。无线每域共用一份频谱，控制只见当前 HOL flow；TDMA 帧内工作保持，OFDMA 固定本帧 RU 并计入 224/272 效率。每服务段整数 ps 向上取整不提高服务速率，实际包 bit 数完整计费。',
        '能耗包括 MAC、scalar、HB、DRAM core、互连及背景代理。TX/RX 与 OFDM 按端点/域活动区间并集计算；OFDM 活动包括控制和端点流水的保守代理。两个域共享节点 LO，保护时间仅计 LO。TDMA 在线均衡未知成本未标定。',
        '## 5. 仿真结果',
        f'所有 {v["planned_records"]} 条计划记录完成：主层 576、主回放 576、敏感性 336、能耗重计价 {v["energy_ledger_records"]}。其中 {v["physical_runs"]} 个独立时序配置实际运行，{v["reused_timing_records"]} 个相同/不受该轴影响的敏感性点复用，并保留 reference。运行了 {v["total_heap_events"]:,} 个日历事件，验证 {v["all_tokens_completed_and_contributors_checked"]:,} 个实际仿真 token 完整贡献集合。计时实验 wall-clock 为 {run["wall_seconds"]:.1f} 秒，4 个进程；这是主机执行开销，不是芯片延迟。',
        '下表吞吐与延迟倍数分别来自独立回放和完整层实验。ST>1 表示 ToV 延迟较低，SG>1 表示有效交付吞吐较高；95% CI 为配对 bootstrap 区间。HIGH 结果紧邻 NORM，不能将 NORM 带宽写为 TSV 技术上限。',
        table(headline_display,['phase','baseline','SG_95CI','ST_95CI','latency_reduction_percent']),
        '![完整层延迟](figures/layer_latency.png)\n\n![D2D 吞吐](figures/d2d_goodput.png)',
        '## 6. 结果整理与对照',
        '每个 model×phase×B×profile 先用三个种子的原始观测合并；Prefill 用 cohort makespan 均值，Decode 用所有有效轮次延迟均值，吞吐为总 useful bytes/总回放时间。再对 24 个 case 比值等权取几何均值。bootstrap 先抽种子，再抽整 cohort；同 phase/B 的模型与方法共用索引，Decode 的全部轮次保持成簇。仅三个种子，区间只覆盖此受控工作负载的不确定性。',
        table([r for r in mechanisms if r['phase']=='all'],['mechanism','baseline','proposal','SG','ST','SE_proxy']),
        '![机制对照](figures/mechanisms.png)',
        '所有收益和负收益均保留。广播、Local Combine 的包量减少并不保证完整层等比例加速；微批、NMP 串行占用、控制等待及固定 OFDMA 分配共同决定路径。累计 NMP/DMA/网络忙时允许重叠，资源审计只表示利用率或活动量，不能相加当作延迟分解。',
        '### 单轴敏感性',
        table(sensitivity,['axis','value','phase','SG','ST','SE_proxy']),
        '\n'.join(crossing)+' 这些只是在离散实验点上的区间，不是实物互连的精确阈值。',
        '![TSV 预算](figures/sensitivity_tsv_budget.png)\n\n![访存效率](figures/sensitivity_memory_efficiency.png)\n\n![节点数](figures/sensitivity_dies.png)\n\n![无线降速](figures/sensitivity_thz_rate.png)',
        '### 能耗代理与 FP8 小算例',
        'energy_sensitivity.csv 对 Qwen3/B=8/hot_shift 的两个 phase、三个种子及两种主方法，使用已记录活动时间将 OFDM 系数重算为 0/0.1/0.4 W；没有重新运行时序。energy_breakdown.csv 提供完整层分项，energy proxy 不作为已验证芯片节能结论。',
        '![能耗代理分项](figures/energy_proxy_components.png)',
        table([dict(rule=x['rule'],LC_relative_L2=x['fp8_LC']['relative_l2'],LC_cosine=x['fp8_LC']['cosine'],NO_LC_relative_L2=x['fp8_NO_LC']['relative_l2'],NO_LC_cosine=x['fp8_NO_LC']['cosine']) for x in numerical['tests']],['rule','LC_relative_L2','LC_cosine','NO_LC_relative_L2','NO_LC_cosine']),
        '该数值校验采用 d=32、h=16、16 token 随机矩阵，验证 E4M3FN 有限值/448 scale/nearest-even、GPT 特定门控、FP32 等价和 ERNIE shared 融合等价。LC/NO_LC 量化路径允许不同；它不是三个预训练模型的准确率验收，本规范也未设置旧版的 5% 门槛。',
        '## 7. 校验与结论',
        f'全部预检及全量运行审计通过。最大单节点 staging 为 {v["peak_staging_bytes"]:,} B，低于 8 MiB；所有 contributor 恰好覆盖 k+shared，无丢 token/重复贡献。所有 Decode 轮增量时间和能耗与 cohort 总量相符。方法间输入/路由/放置和 useful work 的 {v["fairness_groups_checked"]} 组配对均一致。',
        f'在本受控架构仿真和归一化预算下，整体 SG={main["SG"]:.4f}，ST={main["ST"]:.4f}，延迟为基线的 {1/main["ST"]:.4f}。在强有线 HIGH 情景下 SG={high["SG"]:.4f}，ST={high["ST"]:.4f}。上述结果限定于单 MoE 层、公开文本长度和受控路由；不外推为完整模型推理提升、实物 16 节点无线验证或已测能效。',
        '模型假设包含理想可达、无重传、共享频域速率锚点、roofline/流水/控制/功率代理。没有模拟物理传播、BER、热、面积或完整 DRAM 命令行为。NORM 等 raw budget 不表示等面积。参数来源 DOI 与官方模型链接详见上传规范第 10 节。',
        '## 8. 可复现产物',
        'Results 中：config_resolved.json、simulation_parameters.csv、input_manifest.json、inputs/source_lock.json、routes/、run_manifest.csv、layer_results.csv、replay_results.csv、sensitivity_*_results.csv、cohort_summary.csv、tail_latency.csv、summary.csv、method_summary.csv、energy_*.csv、resource_audit.csv、validation.json、independent_audit.json、figures/、abstract_result.txt。runs/*.json.gz 保存本次每个物理配置的原始轮增量、每请求/每 token 时间及每 Die 忙时；复用点只引用原配置。token_timings 列依次为 request_rank、token_index、admission_ps、completion_ps、latency_ps。',
        '本次结果来源于实际事件仿真。旧阶段代码与结果已移出仓库，仅保留本次代码及 Results；所有小型输入、结果和校验值可离线重放。详细命令见根 README。',
    ]
    text='\n\n'.join(lines)+'\n';(out/'report.md').write_text(text,encoding='utf-8')
    word_report(out,text,images)


def word_report(out,text,images):
    from docx import Document
    from docx.shared import Inches,Pt
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    doc=Document();section=doc.sections[0];section.top_margin=Inches(.7);section.bottom_margin=Inches(.7)
    normal=doc.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(9)
    normal.element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Sans CJK SC')
    lines=text.splitlines();i=0
    while i<len(lines):
        line=lines[i]
        if line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].startswith('|'):
                if not re.match(r'^\|[\s|:-]+\|$',lines[i]):block.append([s.strip() for s in lines[i].strip('|').split('|')])
                i+=1
            if block:
                count=max(len(r) for r in block);tab=doc.add_table(rows=1,cols=count);tab.style='Light Shading Accent 1'
                for j,v in enumerate(block[0]):tab.rows[0].cells[j].text=v
                for row in block[1:]:
                    cells=tab.add_row().cells
                    for j,v in enumerate(row):cells[j].text=v
                props=tab.rows[0]._tr.get_or_add_trPr();repeat=OxmlElement('w:tblHeader');props.append(repeat)
            continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));doc.add_heading(line[level:].strip(),min(level,3)-1 if level==1 else min(level,3))
        elif line.startswith('!['):
            match=re.search(r'\]\(([^)]+)\)',line)
            if match:doc.add_picture(str(out/match.group(1)),width=Inches(6.2))
        elif line.strip():doc.add_paragraph(line.replace('**','').replace('`',''))
        i+=1
    doc.core_properties.title='ToV-MoE 完整架构仿真报告';doc.core_properties.subject='Controlled routing, single-layer simulation and communication replay'
    doc.save(out/'simulation_report.docx')
