"""Headless reference timeline. Zero-time causal markers are not busy intervals."""

from pathlib import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .metrics import SimulationResult


def plot_timeline(result: SimulationResult, path: str | Path) -> None:
    result.summary  # Refuse to plot a trace which fails accounting checks.
    figure = Figure(figsize=(11, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    origin = min(c.token.arrival_s for c in result.completions)
    resources = list(dict.fromkeys(span.resource for span in result.intervals))
    positions = {resource: i for i, resource in enumerate(resources)}
    for span in result.intervals:
        if span.kind in ("dispatch", "return"):
            continue  # Draw setup and payload separately from the transmission records.
        color = "#2a9d8f" if span.kind == "expert_compute" else "#7b8290"
        x = (span.start_s - origin) * 1e6
        width = (span.end_s - span.start_s) * 1e6
        if width:
            axis.broken_barh([(x, width)], (positions[span.resource] - 0.3, 0.6), facecolors=color)
        else:
            axis.plot(x, positions[span.resource], "|", color=color)
    for tx in result.transmissions:
        row = positions[f"THz {tx.flow.src}->{tx.flow.dsts[0]}"]
        for start, end, color in ((tx.start_s, tx.payload_start_s, "#e9c46a"),
                                  (tx.payload_start_s, tx.end_s, "#457b9d")):
            axis.broken_barh([((start - origin) * 1e6, (end - start) * 1e6)],
                             (row - 0.3, 0.6), facecolors=color)
    axis.set_yticks(list(positions.values()), list(positions.keys()))
    axis.invert_yaxis()
    axis.set_xlabel("Time since arrival (microseconds)")
    axis.set_title("Stage 1 reference: one token / one remote expert (uncalibrated)")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(handles=[Patch(color=c, label=l) for c, l in
                         (("#e9c46a", "RF setup"), ("#457b9d", "RF payload"),
                          ("#2a9d8f", "Expert compute"), ("#7b8290", "Configured delay / bypass"))],
                loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    figure.savefig(path, dpi=150)
    figure.clear()


def plot_scheduled_timeline(result, path):
    """Actual carrier/time rectangles show concurrency and static idle remainder."""
    result.audit()
    figure = Figure(figsize=(13, 7), layout="constrained")
    FigureCanvasAgg(figure)
    spectrum, compute = figure.subplots(2, 1, sharex=True, height_ratios=(1, 1))
    origin = min(t.arrival_s for t in result.workload.tokens)
    flows = {r.flow.flow_id: r.flow for r in result.flows}
    colors = {"dispatch": "#457b9d", "return": "#d17a45", "setup": "#e9c46a"}
    # Batch rectangles into collections: large traces remain practical to render.
    from matplotlib.collections import PolyCollection
    polygons, facecolors = [], []
    for s in result.segments:
        x1, x2 = (s.start_s-origin)*1e6, (s.end_s-origin)*1e6
        y1, y2 = min(s.subcarriers), max(s.subcarriers)+1
        polygons.append([(x1,y1), (x2,y1), (x2,y2), (x1,y2)])
        kind = "setup" if s.phase == "setup" else "dispatch" if flows[s.flow_id].flow_type.startswith("dispatch") else "return"
        facecolors.append(colors[kind])
    small = len(flows) <= 12
    spectrum.add_collection(PolyCollection(polygons, facecolors=facecolors,
                                           edgecolors="white" if small else "none", linewidths=.4))
    if small:
        labelled = set()
        for s in result.segments:
            if s.phase != "payload" or s.flow_id in labelled:
                continue
            labelled.add(s.flow_id)
            flow = flows[s.flow_id]
            spectrum.text(((s.start_s+s.end_s)/2-origin)*1e6,
                          (min(s.subcarriers)+max(s.subcarriers)+1)/2,
                          f"{flow.flow_type[0].upper()}{flow.flow_id} t{flow.token_id}",
                          ha="center", va="center", fontsize=8, color="white")
    spectrum.set_ylim(0, result.config.phy.num_subcarriers)
    spectrum.set_ylabel("Subcarrier ID (shared total budget)")
    from .config import scheme_label
    spectrum.set_title(f"{scheme_label(result.config)}: {result.config.scheduling.policy}; uncalibrated reference")
    spectrum.legend(handles=[Patch(color=color, label=kind) for kind,color in colors.items()], ncol=3, fontsize=8)
    dies = result.config.topology.num_dies
    spans, colors_ = [], []
    for x in result.executions:
        row = 2*x.batch.die_id
        a,b = (x.start_s-origin)*1e6, (x.end_s-origin)*1e6
        spans.append([(a,row-.35),(b,row-.35),(b,row+.35),(a,row+.35)])
        colors_.append("#2a9d8f")
    for j in result.jobs:
        if j.end_s == j.start_s:
            continue
        row = 2*j.die_id+1
        a,b = (j.start_s-origin)*1e6,(j.end_s-origin)*1e6
        spans.append([(a,row-.35),(b,row-.35),(b,row+.35),(a,row+.35)])
        colors_.append("#8866aa")
    compute.add_collection(PolyCollection(spans, facecolors=colors_,
                                         edgecolors="white" if len(result.executions) <= 24 else "none", linewidths=.4))
    compute.set_yticks([2*d for d in range(dies)], [f"Die {d}" for d in range(dies)], fontsize=8)
    compute.set_ylim(2*dies-.4, -.7)
    compute.set_ylabel("Expert / Combine (adjacent rows)")
    horizon = max(x.completed_s for x in result.completions)-origin
    compute.set_xlim(0, horizon*1e6*1.02)
    compute.set_xlabel("Time since earliest arrival (microseconds)")
    compute.legend(handles=[Patch(color="#2a9d8f", label="Expert compute"), Patch(color="#8866aa", label="Combine")], ncol=2, fontsize=8)
    for axis in (spectrum, compute):
        axis.grid(axis="x", alpha=.15)
    figure.savefig(path, dpi=150)
    figure.clear()


def plot_scheme_comparison(csv_path, path):
    """Figure derives exclusively from exported comparison.csv."""
    import pandas as pd
    table = pd.read_csv(csv_path)
    figure = Figure(figsize=(12, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    for axis, field, scale, label in zip(axes, ("layer_makespan_s", "wireless_payload_bytes", "pepd"),
                                        (1e6, 1, 1), ("Layer latency (us)", "Wireless payload (bytes, all hops)", "PEPD (includes setup)")):
        axis.bar(table["scheme"], table[field]*scale, color=["#8093a0", "#8db7c6", "#457b9d", "#2a9d8f", "#d17a45"])
        axis.set_ylabel(label)
        axis.tick_params(axis="x", labelrotation=15)
        axis.grid(axis="y", alpha=.2)
    figure.suptitle("Five schemes, identical total THz budget (uncalibrated reference)")
    figure.savefig(path, dpi=150)
    figure.clear()


def plot_pipeline_timeline(result, path: str | Path) -> None:
    """Single RF row and one row per Die; a batch is drawn once for all its tokens."""
    result.audit()
    origin = min(t.arrival_s for t in result.workload.tokens)
    num_dies = result.config.topology.num_dies
    figure = Figure(figsize=(12, max(4.2, 1.6 + 0.38 * num_dies)), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for tx in result.transmissions:
        for start, end, color in ((tx.start_s, tx.payload_start_s, "#e9c46a"),
                                  (tx.payload_start_s, tx.end_s, "#457b9d")):
            axis.broken_barh([((start - origin) * 1e6, (end - start) * 1e6)], (-0.3, 0.6), facecolors=color)
        if len(result.transmissions) <= 12:
            label = f"t{tx.flow.token_id}: {tx.flow.src}->" + ",".join(map(str, tx.flow.dsts))
            axis.text((tx.start_s - origin) * 1e6, -0.38, label, fontsize=7, va="bottom")
    for execution in result.executions:
        batch = execution.batch
        row = batch.die_id + 1
        x, width = (execution.start_s - origin) * 1e6, (execution.end_s - execution.start_s) * 1e6
        color = "#2a9d8f" if batch.expert_id % 2 == 0 else "#68b6a8"
        axis.broken_barh([(x, width)], (row - 0.3, 0.6), facecolors=color, edgecolors="white", linewidth=0.3)
        if len(result.executions) <= 24:
            axis.text(x + width / 2, row, f"e{batch.expert_id} m={len(batch.request_ids)}",
                      fontsize=7, ha="center", va="center")
    axis.set_yticks(range(num_dies + 1), ["THz Dispatch (FIFO)", *[f"Compute die {d}" for d in range(num_dies)]])
    axis.set_ylim(num_dies + 0.6, -0.9)
    end = max(c.completed_s for c in result.completions)
    axis.set_xlim(0, (end - origin) * 1e6 * 1.02)
    axis.set_xlabel("Time since earliest arrival (microseconds)")
    axis.set_title("Stage 2: Dispatch + Expert Compute only; uncalibrated (no Combine / Return)")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(handles=[Patch(color=c, label=l) for c, l in
                         (("#e9c46a", "RF setup"), ("#457b9d", "Broadcast payload"),
                          ("#2a9d8f", "Expert batch (shared Die resource)"))],
                loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    figure.savefig(path, dpi=150)
    figure.clear()


def plot_layer_timeline(result, path: str | Path) -> None:
    """One shared RF row plus separate expert/Combine resources per Die."""
    result.audit()
    origin = min(t.arrival_s for t in result.workload.tokens)
    dies = result.config.topology.num_dies
    completion_row = 1 + 2 * dies
    figure = Figure(figsize=(13, max(5, 2.5 + 0.55 * dies)), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    for tx in result.transmissions:
        payload_color = "#457b9d" if tx.flow.flow_type == "dispatch" else "#d17a45"
        for start, end, color in ((tx.start_s, tx.payload_start_s, "#e9c46a"),
                                  (tx.payload_start_s, tx.end_s, payload_color)):
            axis.broken_barh([((start - origin) * 1e6, (end - start) * 1e6)], (-0.3, 0.6), facecolors=color)
        if len(result.transmissions) <= 8:
            axis.text((tx.start_s - origin) * 1e6, -0.4,
                      f"{tx.flow.flow_type[0].upper()} t{tx.flow.token_id}", fontsize=7, va="bottom")
    for execution in result.executions:
        row = 1 + 2 * execution.batch.die_id
        start, width = (execution.start_s - origin) * 1e6, (execution.end_s - execution.start_s) * 1e6
        axis.broken_barh([(start, width)], (row - 0.3, 0.6), facecolors="#2a9d8f", edgecolors="white", linewidth=0.3)
        if len(result.executions) <= 16:
            axis.text(start + width / 2, row, f"e{execution.batch.expert_id}", ha="center", va="center", fontsize=7)
    job_colors = {"weight": "#a8a0cb", "local_reduce": "#8866aa", "source_reduce": "#5c3d80"}
    for job in result.jobs:
        row = 2 + 2 * job.die_id
        x, width = (job.start_s - origin) * 1e6, (job.end_s - job.start_s) * 1e6
        if width:
            axis.broken_barh([(x, width)], (row - 0.3, 0.6), facecolors=job_colors[job.kind])
        else:
            axis.plot(x, row, "|", color=job_colors[job.kind], markersize=5)
    for span in result.intervals:
        if span.kind == "source_to_base":
            axis.broken_barh([((span.start_s - origin) * 1e6, (span.end_s - span.start_s) * 1e6)],
                             (completion_row - 0.3, 0.6), facecolors="#7b8290", alpha=0.6)
    for completion in result.completions:
        axis.plot((completion.completed_s - origin) * 1e6, completion_row, "v", color="#263238", markersize=4)
    labels = ["THz shared: Dispatch + Return"]
    for die in range(dies):
        labels.extend((f"Expert compute die {die}", f"Combine die {die}"))
    labels.append("Source to Base / completion")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_ylim(completion_row + 0.6, -0.9)
    horizon = max(c.completed_s for c in result.completions) - origin
    axis.set_xlim(0, horizon * 1e6 * 1.03)
    axis.set_xlabel("Time since earliest arrival (microseconds)")
    axis.set_title("Stage 3: routed-expert layer through Base completion (uncalibrated)")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(handles=[Patch(color=c, label=l) for c, l in
        (("#e9c46a", "RF setup"), ("#457b9d", "Dispatch"), ("#d17a45", "Return"), ("#2a9d8f", "Expert compute"),
         ("#a8a0cb", "Weight"), ("#8866aa", "Local Reduce"), ("#5c3d80", "Source Reduce"))],
        loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, fontsize=8)
    figure.savefig(path, dpi=150)
    figure.clear()
