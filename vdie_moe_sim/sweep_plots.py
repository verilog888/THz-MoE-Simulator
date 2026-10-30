"""Exportable sweep figures derived only from aggregate.csv and views.csv."""

import json
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


PANELS = (("layer_makespan_s",1e6,"Layer latency (us)"),
          ("throughput_tokens_per_s",1,"Throughput (tokens/s)"),
          ("token_latency_p95_s",1e6,"Mean of per-run p95 latency (us)"),
          ("token_latency_p99_s",1e6,"Mean of per-run p99 latency (us)"),
          ("thz_utilization",1,"THz utilization"),
          ("mean_compute_utilization",1,"Mean Die compute utilization"),
          ("pepd",1,"PEPD (includes setup)"),
          ("bandwidth_waste",1,"Allocated bandwidth waste"),
          ("speedup_vs_C",1,"Mean paired speedup vs TDMA C"))


def plot_sweep(directory, study, statistics):
    root=Path(directory)
    aggregate=pd.read_csv(root/"aggregate.csv",float_precision="round_trip")
    views=pd.read_csv(root/"views.csv",float_precision="round_trip",keep_default_na=False,
                      dtype={"value_json":str,"changes_json":str,"point_id":str})
    plots=root/"plots";plots.mkdir(exist_ok=True)
    outputs=[]
    series=("C","D","Proposed") if study=="ablation" else ("A","B","C","D","Proposed")
    colors={"A":"#7c8b96","B":"#9b6db0","C":"#457b9d","D":"#2a9d8f","Proposed":"#d17a45"}
    labels={"C":"TDMA","D":"Static OFDMA","Proposed":"CA-OFDMA"} if study=="ablation" else {s:s for s in series}
    for axis_name,coordinates in views.groupby("axis",sort=False):
        coordinates=coordinates.sort_values("view_order")
        values=[json.loads(v) for v in coordinates["value_json"]]
        numeric=all(type(v) in (int,float) for v in values) and axis_name!="grid"
        x=np.array(values,dtype=float) if numeric else np.arange(len(values),dtype=float)
        xlabel=axis_name
        if axis_name=="phy.total_bandwidth_hz":
            x=x/1e9;xlabel="Total THz bandwidth (GHz)"
        if axis_name=="grid":
            ticks=[", ".join(f"{k.split('.')[-1]}={v}" for k,v in json.loads(c).items()) for c in coordinates["changes_json"]]
            xlabel="Explicit grid combinations (all coordinates shown)"
        else:
            ticks=[("on" if v else "off") if type(v) is bool else str(v) for v in values]
        fig=Figure(figsize=(14,10.5),layout="constrained");FigureCanvasAgg(fig)
        axes=fig.subplots(3,3)
        for panel,(metric,scale,ylabel) in zip(axes.flat,PANELS):
            for name in series:
                subset=aggregate[(aggregate["metric"]==metric)&(aggregate["series"]==name)].set_index("point_id")
                table=subset.loc[list(coordinates["point_id"])]
                y=table["mean"].to_numpy(dtype=float)*scale
                order=np.argsort(x) if numeric else np.arange(len(x))
                panel.plot(x[order],y[order],"o-",markersize=3,linewidth=1.3,label=labels[name],color=colors[name])
                low,high=table["ci_low"].to_numpy(dtype=float)*scale,table["ci_high"].to_numpy(dtype=float)*scale
                if np.isfinite(low).all() and np.isfinite(high).all():
                    panel.fill_between(x[order],low[order],high[order],color=colors[name],alpha=.12)
            panel.set_ylabel(ylabel if study!="ablation" or metric!="speedup_vs_C" else "Mean paired speedup vs matched TDMA")
            panel.set_xlabel(xlabel,fontsize=8)
            if not numeric:
                panel.set_xticks(x,ticks,rotation=35 if axis_name=="grid" else 0,ha="right" if axis_name=="grid" else "center",fontsize=6 if axis_name=="grid" else 8)
            elif axis_name in ("workload.num_tokens","topology.num_dies","phy.num_subcarriers"):
                panel.set_xscale("log",base=2)
                panel.set_xticks(x,ticks,fontsize=8)
            panel.grid(alpha=.2)
            if metric=="speedup_vs_C": panel.axhline(1,color="#666666",linestyle="--",linewidth=.7)
        handles,legend=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend,loc="outside lower center",ncol=len(series),fontsize=9)
        interval=f"exploratory {statistics.confidence:.0%} bootstrap bands" if statistics.bootstrap_resamples else "no confidence intervals"
        fig.suptitle(f"{study}: seed means; {interval}\nUncalibrated reference; identical budget within each parameter point",fontsize=12)
        stem=axis_name.replace(".","_")
        png=plots/f"{stem}.png";svg=plots/f"{stem}.svg"
        fig.savefig(png,dpi=150)
        with matplotlib.rc_context({"svg.hashsalt":"vdie-moe-stage5","svg.fonttype":"none"}):
            fig.savefig(svg,metadata={"Date":None})
        fig.clear()
        outputs.extend([png.relative_to(root).as_posix(),svg.relative_to(root).as_posix()])
    return outputs
