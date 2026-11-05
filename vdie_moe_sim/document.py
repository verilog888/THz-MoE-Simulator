"""Chinese Word/Markdown report generated only from completed experiment records."""
import csv
import json
from pathlib import Path
from datetime import datetime,timezone
import re
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION,WD_ORIENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm,Pt,RGBColor
from .config import ROOT

def fmt(value,digits=5):
    if isinstance(value,(float,int)):
        return f'{value:.{digits}g}'
    return str(value)

def write_documents(out,c,raw,agg,paired,manifest,figures):
    out=Path(out);validation=manifest['validation'];all_passed=validation['all_acceptance_passed']
    doc=Document();md=['# V-Die 单MoE层完整仿真报告','']
    section=doc.sections[0]
    section.page_width,section.page_height=Cm(21),Cm(29.7)
    section.top_margin=section.bottom_margin=Cm(1.6)
    section.left_margin=section.right_margin=Cm(1.7)
    for name,size in [('Normal',10),('Title',24),('Subtitle',12),('Heading 1',16),('Heading 2',12),('Caption',9)]:
        style=doc.styles[name];style.font.name='Noto Sans CJK SC';style.font.size=Pt(size)
        style._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Sans CJK SC')
    doc.styles['Normal'].paragraph_format.line_spacing=1.15
    doc.styles['Normal'].paragraph_format.space_after=Pt(6)
    for name in ['Title','Heading 1','Heading 2']:
        doc.styles[name].font.color.rgb=RGBColor.from_string('17365D')
    section.header.paragraphs[0].text='THz-MoE Simulator ｜ 批准规范 v1.0 ｜ 单MoE层架构模型'
    footer=section.footer.paragraphs[0];footer.alignment=2;footer.add_run('第 ')
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field);footer.add_run(' 页')
    doc.add_heading('V-Die 单MoE层完整仿真报告',0)
    doc.add_paragraph('三模型 · 八组合完整消融 · 强TDMA对照 · 单因素敏感性',style='Subtitle')
    def para(text,bold=False):
        p=doc.add_paragraph();p.add_run(text).bold=bold;md.extend([text,''])
    def heading(text,level=1):
        doc.add_heading(text,level);md.extend(['#'*(level+1)+' '+text,''])
    def table(headers,rows,widths=None):
        rows=[[fmt(x) for x in row] for row in rows]
        t=doc.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.autofit=False
        available=26.3 if doc.sections[-1].orientation==WD_ORIENT.LANDSCAPE else 17.6
        widths=widths or [available/len(headers)]*len(headers)
        for cell,label,width in zip(t.rows[0].cells,headers,widths):
            cell.text=label;cell.width=Cm(width)
            color=OxmlElement('w:shd');color.set(qn('w:fill'),'17365D');cell._tc.get_or_add_tcPr().append(color)
            for run in cell.paragraphs[0].runs:run.bold=True;run.font.color.rgb=RGBColor(255,255,255)
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for values in rows:
            row=t.add_row();keep=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(keep)
            for cell,text,width in zip(row.cells,values,widths):
                cell.width=Cm(width)
                cell.text=text.replace('_','_\u200b').replace('/','/\u200b').replace('.','.\u200b')
                for p in cell.paragraphs:
                    p.paragraph_format.space_after=Pt(1)
                    for run in p.runs:run.font.size=Pt(8)
        for col,width in zip(t.columns,widths):col.width=Cm(width)
        doc.add_paragraph()
        md.append('| '+' | '.join(headers)+' |');md.append('|'+'|'.join(['---']*len(headers))+'|')
        for row in rows:md.append('| '+' | '.join(x.replace('|',r'\|') for x in row)+' |')
        md.append('')
    def image(name):
        record=next(f for f in figures if f['name']==name)
        path=out/'figures'/f'{name}.png'
        width=25.8 if doc.sections[-1].orientation==WD_ORIENT.LANDSCAPE else 17.3
        doc.add_picture(str(path),width=Cm(width))
        doc.paragraphs[-1].paragraph_format.keep_with_next=True
        caption=doc.add_paragraph(record['caption'],style='Caption')
        caption.paragraph_format.keep_with_next=False
        md.extend([f"![{record['caption']}](figures/{name}.png)",''])
    def orientation(landscape):
        sec=doc.add_section(WD_SECTION.NEW_PAGE)
        sec.orientation=WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
        sec.page_width,sec.page_height=(Cm(29.7),Cm(21)) if landscape else (Cm(21),Cm(29.7))
        sec.left_margin=sec.right_margin=Cm(1.7)
    para(f"批准版本：{manifest['approval']['approval_id']}；用户确认日期：2026年11月5日（Asia/Shanghai）。")
    if all_passed:
        para(f'全部 {len(raw):,} 条计划记录完成，所有验收检查通过。',True)
    else:
        para(f'全部 {len(raw):,} 条计划记录完成；FP8端到端数值误差验收未通过。experiments_complete=true，complete=false。',True)
        para('批准的NRMSE上限仍为5%，未改阈值、未删除失败断言。该失败与时序/能耗描述符实验分开报告，不能宣称已满足规范的全部验收要求。')
    heading('1. 仿真目标')
    para('测量Base上输入就绪至Base收到输出的单MoE层延迟与模型估计能耗，通过完整2³消融分离动态OFDMA、Dispatch广播、Local Combine及其交互作用；另与包粒度强TDMA比较。')
    para('层内计算、共享内存接口、无线控制/数据服务、半双工冲突、缓冲区背压、shared专家和本地归并进入同一事件模型；不通过总字节除总带宽估算整层时间。')
    doc.add_page_break()
    heading('2. 仿真简述')
    para('数据路径为Base输入→owner Gate/Top-k→Dispatch→专家计算→可选Local Combine→owner最终Combine→Base输出。Python组织配置、输入、统计、审计与报告，本地C++扩展加速整数皮秒离散事件内核。计算/DMA按5ns主时钟对齐，无线按1ps事件分辨率处理，跨帧保留分数bit服务。')
    para('文本来自MohamedRashad/ChatGPT-prompts，使用三个模型冻结revision中的官方tokenizer，不加特殊token或聊天模板。路由为受控合成，标记dataset_tokens_synthetic_router；没有加载预训练模型权重或采集真实隐藏状态。Decode重放回复token，单层耗时不代表完整模型TBT。')
    para('权重预驻留，计本层热态执行。Attention、KV cache、其它层、初始化权重装载、外部主存及residual add均不在计时边界内。M3每token只计算两个shared分支，owner/buddy各一个。')
    table(['模型','隐藏/专家中间维度','routed / Top-k / shared','每Die驻留字节'],
          [[mid, f"{m['d_model']} / {m['d_ff']}",f"{m['routed_experts']} / {m['top_k']} / {m['shared_experts']}",f"{m['resident_bytes_per_die']:,}"] for mid,m in c['models'].items()])
    para('两个能耗账本均含计算、向量处理、eDRAM读写、刷新、泄漏、RF及控制待机；baseband_stress再加未标定的在线EQ和OFDMA额外基带proxy。默认刷新约3.02545W/Die、16Die约48.40727W，保持连续后台积分。这些是参数外推，不是目标芯片实测功耗。')
    orientation(True)
    heading('3. 仿真参数表')
    para('下表包含全部211项原规范机器参数。取值、单位、含义与依据来自已批准规范，运行期未改变模型参数或5%数值门槛；扫描值见experiment组。U/P/C/D/A分别表示用户固定、论文记录、官方配置、设计选择、未验证外推。参数表CSV另保留逐项文献URL，完整引用见附录。')
    parameters=pd.read_csv(out/'parameters.csv',keep_default_na=False)
    prefixes=['spec','simulation','models.M1','models.M2','models.M3','precision','hardware','compute','memory','mapping','radio','mac','energy','workload','experiment']
    for prefix in prefixes:
        heading(prefix,2)
        rows=[]
        for r in parameters[parameters.key.str.startswith(prefix+'.')].to_dict('records'):
            rows.append([r['key'],r['value'],r['unit'],r['meaning'],r['basis']])
        table(['参数键','值','单位','含义与规则','依据'],rows,[5.0,4.0,1.5,11.1,4.7])
    heading('已批准补充与执行约定',2)
    para('FP32向量能耗倍率增加2/4/8单因素扫描（默认4）；all_non_source广播将所有非源节点纳入实际RX半双工冲突约束，重跑时序。Prefill的Decode复用概率0.8记录为语义不适用的等价控制，明确引用默认轨迹。')
    implementation=pd.read_csv(ROOT/'docs/approval/implementation_options.csv',keep_default_na=False)
    implrows=[]
    for r in implementation.to_dict('records'):
        if r['key']=='review.simulation_allowed':r['value']='true（用户已确认）'
        if r['key']=='output.final_report':r['value']='results/simulation_report.docx'
        r['meaning']=r['meaning'].replace('当前为待审批，不能据此自动运行。','用户已确认批准。').replace('本稿提出供审批。','此范围已随参数表获批准。')
        r['basis']=r['basis'].replace('待审批','已批准')
        implrows.append([r['key'],r['value'],r['unit'],r['meaning'],r['basis']])
    table(['约定键','执行值','单位','含义','依据'],implrows,[5.2,6.0,1.2,10.5,3.4])
    heading('派生量复算',2)
    derived=pd.read_csv(ROOT/'docs/approval/derived_parameters.csv',keep_default_na=False)
    table(['派生项','数值','单位','含义','公式'],derived.values.tolist(),[5.3,3.3,1.7,7.7,8.3])
    orientation(False)
    heading('4. 仿真流程')
    para('首先核对容量和算量，冻结数据集/tokenizer及文件哈希，按五个种子无放回抽样128请求；同种子三个模型使用相同行ID，各基线共用同一输入/路由trace。')
    para('Prefill按B=1/8/32分cohort，不补齐短prompt；Decode按B=1/8/32/128，每轮每活跃请求仅释放一个token，等全部本轮输出到Base后进入下一轮，不回填请求。owner最多16个inflight token，超限在Base等待。')
    para('每个Die共用计算资源和FIFO内存接口。矩阵采用M≤16、N/K≤128的tile及两块有界预取缓冲；向量激活/量化按128元素scale块流式处理。Combine、Gate、expert按优先级在任务/tile边界调度，DMA与无线存储操作竞争。返回和owner累加器空间先预留。')
    para('base/peer分别共享一份频谱；同组半双工，不同组可重叠。T逐帧独占一条flow；O先每条相容flow授1RU，再按剩余需求分配；TW收实际控制与guard后按包长服务。所有方案均计包头、CRC、MTU分片、控制报告和grant。')
    para('scheduler_metrics.csv逐配置记录实际控制窗口、T/O有效帧长、TW平均grant服务时长以及控制占用。控制空口利用率以报告/grant串行时间除以控制窗口；调度控制占比以控制时间除以已分配帧或grant时长。该调度分母包含已预留帧尾，不等同于cohort实测持续时间；TW没有固定帧长，相关列留空。')
    table(['部分','记录数','执行方式'],[
        ['主实验',840,'三模型×八组合×七负载×五种子'],
        ['强TDMA',60,'三模型×两策略×两代表负载×五种子'],
        ['原规范敏感性',2880,'14轴；单因素；去重默认点'],
        ['补充向量能耗敏感性',240,'2/8替代默认4；只重算能量'],
        ['合计',len(raw),'2,280时序 + 1,680能耗重算 + 60等价控制']])
    para('先执行结构/边界检查和独立FP8校验，再完成全部清单。每个时序配置额外复算通信字节与矩阵OP，验证token、资源和依赖路径闭合。能耗情景从保存的事件计数重算；已有结果须通过配置、引擎与文件哈希校验才允许续跑。')
    para('统计以种子和cohort分层配对bootstrap，重抽1000次，使用95%区间。速度比是成对均值之比，节能率为1减成对总能量比；公共待机先按cohort累计，再除实际完成token。资源busy-time可以重叠，不与依赖路径总延迟混用。')
    para('RU诊断提供两种分母：data_window是存在grant的完整数据窗口，含最后一帧剩余空尾；cohort_window是两介质×16RU×实际cohort持续时间。前者描述已调度窗口内利用率，后者描述整个测量区间内利用率。空闲介质没有凭空占用射频数据活动功率。')
    heading('5. 仿真结果')
    structural=validation['structural'];numeric=validation['numeric_gate']
    table(['检查','实际结果'],[
        ['计划记录',f"{len(raw):,} / 4,020 已解决；各项状态保存在manifest"],
        ['结构与流程测试',f"{structural['tests']} 项；失败{structural['failures']}、错误{structural['errors']}、跳过{structural['skipped']}"],
        ['FP8门槛测试',f"{numeric['tests']} 项；失败{numeric['failures']}、错误{numeric['errors']}；原NRMSE门槛0.05"],
        ['配置级审计','每个时序配置均检查字节、矩阵OP、token与资源守恒、依赖路径闭合'],
        ['完成状态',f"experiments_complete={manifest['experiments_complete']}；complete={manifest['complete']}"]],[4.2,13.4])
    diagnosis=manifest['offline_audit']['numeric_diagnosis']
    para(f"独立ml_dtypes {diagnosis['reference_version']}对照已核对{diagnosis['encode_comparisons']:,}个E4M3FN编码及全部256个解码码字，编码逐位一致。无量化的FP32 Combine等价性通过；完整FP8量化链的NRMSE仍超过批准上限，说明这一误差门槛没有被当前指定小算例满足。")
    table(['模型','M1非零bias','Local Combine','NRMSE','余弦','验收'],
          [[r['model'],str(r['nonzero_bias']),str(r['local_combine']),fmt(r['nrmse']),fmt(r['cosine']),'通过' if r['passed'] else '未通过']
           for r in validation['fp8_cases']],[1.3,2.3,2.6,2.2,2.2,2.1])
    para('以下结果为已批准时序/能耗模型的运行输出，不能据此声称FP8数值验收通过或真实模型精度得到验证。所有曲线及配对原始表均在results中保留。')
    orientation(True)
    for model in ['M1','M2','M3']:
        heading(f'{model}：绝对延迟与能耗',2)
        for mode in ['prefill','decode']:image(f'absolute_{model}_{mode}')
    orientation(False)
    heading('B32请求层延迟分布',2)
    para('下列均值、p50和p95单位为毫秒；分别先在每个种子的128个请求上计算，再取五种子平均。Prefill为整个prompt完成延迟，Decode为每个请求实际步数的平均单层延迟。token分布和每轮活跃B另存原始Parquet与运行表。')
    t=agg[(agg.family=='main')&(agg.batch==32)]
    table(['模型','负载','策略','均值 ms','p50 ms','p95 ms'],
          [[r['model'],r['mode'],r['baseline'],1000*r['mean_request_layer_s'],
            1000*r['p50_request_layer_s'],1000*r['p95_request_layer_s']] for r in t.to_dict('records')],
          [1.2,2.1,2.4,3.9,4,4])
    heading('6. 结果整理与对照')
    para('重点表采用B32。完整B=1/8/32及Decode B128结果见aggregate.csv、raw_runs.csv与对应图表。跨模型仅讨论各自相对基线的变化，不把不同矩阵规模导致的原始延迟差异归因于通信机制。')
    selected=paired[(paired.comparison=='vs_T_P_N')&(paired.batch==32)&(paired.variant=='O_B_L')]
    comparison_rows=[]
    for r in selected.to_dict('records'):
        comparison_rows.append([r['model'],r['mode'],f"{r['speedup']:.4f} [{r['speedup_low']:.4f}, {r['speedup_high']:.4f}]",
                                f"{100*r['frontend_saving']:.3f}%",f"{100*r['stress_saving']:.3f}%"])
    table(['模型','负载','O_B_L加速比 [95%区间]','前端节能','压力节能'],comparison_rows,[1.2,1.9,7.5,3.5,3.5])
    orientation(True)
    para('通信字节减少与请求延迟变化分别如下；节省为正、增加为负。空口字节按物理发射一次计，不把广播接收方数乘回发射字节。专家批量变化会改变权重复用和eDRAM流量；以下T→O指T_P_N→O_B_L整组对比。')
    traffic=[]
    for (mid,mode),g in agg[(agg.family=='main')&(agg.batch==32)].groupby(['model','mode']):
        g=g.set_index('baseline');a,b=g.loc['T_P_N'],g.loc['O_B_L']
        traffic.append([mid,mode,
          f"{100*(1-b.air_dispatch_bytes_per_token/a.air_dispatch_bytes_per_token):.2f}%",
          f"{100*(1-b.air_return_bytes_per_token/a.air_return_bytes_per_token):.2f}%",
          f"{a.mean_expert_batch:.3f} → {b.mean_expert_batch:.3f}",
          f"{(a.read_bytes_per_token+a.write_bytes_per_token)/1e6:.3f} → {(b.read_bytes_per_token+b.write_bytes_per_token)/1e6:.3f}",
          f"{100*(b.mean_request_layer_s/a.mean_request_layer_s-1):+.3f}%"])
    table(['模型','负载','Dispatch节省','Return节省','平均专家批量 T→O','eDRAM读写 MB/token T→O','请求延迟变化'],traffic,[1.2,2.0,3.5,3.5,5.0,5.5,5.6])
    for name in ['mechanism_ofdma','mechanism_broadcast','mechanism_local_combine','strong_tdma',
                 'factorial_interactions','communication_bytes','resource_utilization','energy_components','dependency_path','ofdma_power_threshold']:image(name)
    orientation(False)
    para('广播在同一输入发往多个Die时复用空口，单目的退化为单播。P2P基线也按目的Die去重。Local Combine只合并同token同Die的贡献，是否减少返回取决于实际命中重合；两个shared贡献的owner/buddy重合也已计入。')
    fanout=pd.read_csv(out/'fanout_validation.csv')
    para('均匀无放回路由中，令p=1−C(E−E/16,k)/C(E,k)。无shared模型的远端目的Die数期望为15p；M3的buddy必达，故为1+14p。下表与冻结trace实际命中数对照，排除owner本地贡献；不把Top-k直接当作通信份数。')
    table(['模型','负载','均匀理论远端Die数','实际远端Die数','实际token数'],
          [[r['model'],r['mode'],r['uniform_expected_remote_dies'],r['observed_remote_dies'],r['actual_tokens']]
           for r in fanout.to_dict('records')],[1.3,2.3,5.1,5,3.9])
    para('OFDMA保持同一满带宽锚点，另付14/17效率损失。与TW对照用于判断收益是否依赖帧空尾；不能把“可并发发送”直接等同于更低完整层延迟。factorial_interactions.csv给出对数尺度二阶和三阶交互，三项百分比不可直接相加。')
    para('交互采用标准2³因子效应：T/P/N编码−1，O/B/L编码+1；对因素集合S计算effect=Σ[∏(x_i, i∈S)×log(metric)]/4。乘性形式为exp(effect)，对应回归交互系数的两倍。factorial_summary.csv提供全部八组合共同配对、按种子/cohort重抽的95%区间。能量对比的公共token分母在该零和对比中消去。')
    heading('敏感性与不确定性',2)
    para('15个单因素轴的三个指标均已出图：速率降额、广播速率、计算利用率、内存带宽与能耗修正、刷新周期、计算能耗、OFDMA和EQ功耗、帧长、广播监听范围、路由热点、Decode复用、待机及向量能耗倍率。所有曲线均含默认点，且默认点来自主实验。')
    orientation(True)
    for axis in ['radio_rate_scale','memory_bandwidth_scale','workload_routing_profile']:
        image('sensitivity_'+axis+'_mean_request_layer_s')
    for axis in ['memory_refresh_period_ps','energy_ofdma_extra_w_per_active_group']:
        image('sensitivity_'+axis+'_baseband_stress_uj_token')
    orientation(False)
    heading('7. 结论')
    for r in selected.to_dict('records'):
        para(f"{r['model']} {r['mode']} B32：O_B_L相对T_P_N的加速比为{r['speedup']:.4f}，"
             f"95%区间[{r['speedup_low']:.4f}, {r['speedup_high']:.4f}]；"
             f"前端proxy节能率{100*r['frontend_saving']:.3f}%（95%区间[{100*r['frontend_saving_low']:.3f}%, {100*r['frontend_saving_high']:.3f}%]），"
             f"基带压力账本节能率{100*r['stress_saving']:.3f}%（95%区间[{100*r['stress_saving_low']:.3f}%, {100*r['stress_saving_high']:.3f}%]）。")
    para('加速比区间包含1时，尚未显示稳定的延迟优势；节能率区间包含0时，尚未显示稳定的能量优势。组合变化不能全部归功于OFDMA，须结合第6节固定其它机制的条件比较。')
    strong=paired[paired.comparison=='vs_strong_tdma']
    for r in strong.to_dict('records'):
        state='区间完全高于1' if r['speedup_low']>1 else '区间完全低于1' if r['speedup_high']<1 else '区间包含1'
        para(f"{r['model']} {r['mode']} B32：{r['variant']}相对强基线{r['baseline']}加速比{r['speedup']:.4f}，{state}。该比较保留实际控制成本并去除了固定帧空尾。")
        fixed=paired[(paired.comparison=='ofdma')&(paired.model==r['model'])&(paired['mode']==r['mode'])
                     &(paired.batch==32)&(paired.variant==r['variant'])].iloc[0]
        if fixed.speedup_low>1 and r['speedup_low']<=1:
            para('该条件下OFDMA对帧TDMA有稳定收益，但对包粒度TW尚无稳定优势；收益依赖固定帧的调度粒度和空尾回收，不能归因为同带宽下的天然加速。')
    para('所有结果是单层、热态、真实文本长度与合成路由下的架构模型估计。16节点覆盖、两介质隔离、在线均衡/OFDMA基带、64MB eDRAM外围及跨精度计算能耗仍含未实测外推；置信区间描述路由种子和请求变化，不覆盖硬件模型不确定性。')
    para('当前未满足的验收项为FP8小算例NRMSE≤5%。本文保留失败结果和原门槛；可用于检查时序/资源/能耗模型和比较机制，但不能作为满足数值精度要求或真实目标芯片PPA的证据。',True)
    heading('附录：复现与文件索引',2)
    para('安装：bash scripts/setup_env.sh。完整复现：python -m vdie_moe_sim.run_all --spec VDIE_MOE_SIM_SPEC.md --output results。已有同版本结果使用--resume；--require-numeric-gate可要求在该数值验收失败时先停止时序实验。')
    para('token_metrics.parquet是按时序配置分片的Parquet数据集，避免复制能量情景的同一token时间表；raw_runs.csv中的temporal_run_id连接重算/等价记录。runs/<id>/ledger.parquet保留事件计数，record.json保存输入/引擎/文件哈希。')
    table(['文件','内容'],[
        ['manifest.json / validation.json','完整计划、来源与依赖、完成状态、通过/失败检查'],
        ['parameters.csv / config_resolved.yaml','参数含义与实际主配置；每条overrides在manifest中'],
        ['traces/*.jsonl.gz / input_manifest.json','冻结路由和真实分词长度、revision、文件哈希'],
        ['raw_runs.csv / aggregate.csv','全部记录及五种子汇总'],
        ['token_metrics.parquet / cohort_metrics.parquet','token细节及配对cohort单位'],
        ['paired_effects.csv / factorial_interactions.csv / factorial_summary.csv','三机制配对效应与交互项；含共同配对95%区间'],
        ['resource_and_dependency_metrics.csv','计算/DMA资源busy-time、利用率与依赖路径分解'],
        ['scheduler_metrics.csv','有效帧长、控制窗口/利用率、包粒度grant平均服务时长'],
        ['energy_breakdown.csv / ofdma_power_threshold.csv','能耗分项与新增基带功耗盈亏阈值'],
        ['figures/*.png / *.svg / *.csv','可导出图表及每图源数据']],[6.1,11.5])
    refs=dict(re.findall(r'^\[([PCD]\d+)\]: (\S+)$',(ROOT/'VDIE_MOE_SIM_SPEC.md').read_text(),re.M))
    para('论文数据与条件采用上传规范中的来源标注；官方模型配置、数据集和tokenizer均冻结版本。P1的264Gb/s为双通道合计；P2为FP16 SRAM-CIM原型；P7为小宏仿真；P8为DMT接收器压力参照，均不伪装为目标16-Die芯片实测。')
    table(['引用','URL'],list(refs.items()),[1.3,16.3])
    doc.core_properties.title='V-Die 单MoE层完整仿真报告'
    doc.core_properties.subject='包含完整参数、实验、结果对照以及未通过的FP8数值验收'
    doc.core_properties.created=datetime.now(timezone.utc)
    doc.core_properties.modified=doc.core_properties.created
    doc.save(out/'simulation_report.docx')
    (out/'report.md').write_text('\n'.join(md),encoding='utf-8')
