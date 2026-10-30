import argparse
import json
from pathlib import Path

from .config import load_config
from .simulator import run_reference


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Uncalibrated V-Die MoE staged discrete-event simulator")
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3", "stage4", "stage5"), default="stage1",
                        help="stage1 singleton; stage2 Dispatch/Compute; stage3 full layer; stage4 five schemes; stage5 sweep")
    parser.add_argument("--scheme", choices=("A", "B", "C", "D", "Proposed"), help="override stage4 scheme")
    parser.add_argument("--compare", action="store_true", help="stage4: run all five schemes on the same workload and budget")
    parser.add_argument("--config", help="strict JSON simulation configuration (stages 1-4)")
    parser.add_argument("--out", help="new or empty output directory; saved directory for resume/audit")
    parser.add_argument("--plan", help="stage5 strict JSON sweep plan")
    parser.add_argument("--dry-run", action="store_true", help="stage5: validate and count runs without simulation or writes")
    parser.add_argument("--resume", action="store_true", help="stage5: reuse verified results of exactly the same saved experiment")
    parser.add_argument("--audit", action="store_true", help="stage5: audit a saved sweep from CSV/NPY; no simulation")
    parser.add_argument("--timeline", action="store_true", help="render a headless communication/compute timeline")
    args = parser.parse_args(argv)
    try:
        if args.stage == "stage5":
            if args.config or args.scheme or args.compare or args.timeline:
                raise ValueError("stage5 uses --plan; simulation settings and timelines belong in that plan")
            if args.audit:
                if not args.out or args.plan or args.resume or args.dry_run:
                    raise ValueError("stage5 --audit requires only --out and no --plan/--resume/--dry-run")
                from .sweep import audit_sweep
                print(json.dumps(audit_sweep(args.out),indent=2,sort_keys=True))
                return 0
            if not args.plan or (args.resume and args.dry_run):
                raise ValueError("stage5 requires --plan; --resume cannot be combined with --dry-run")
            from .sweep_plan import load_plan
            plan = load_plan(args.plan)
            if args.dry_run:
                print(json.dumps(plan.preview(),indent=2,sort_keys=True))
                return 0
            if not args.out:
                raise ValueError("stage5 execution requires --out")
            from .sweep import run_sweep
            result = run_sweep(plan,args.out,resume=args.resume,progress=lambda message:print(message,flush=True))
            print(json.dumps(result,indent=2,sort_keys=True))
            return 0
        if args.plan or args.dry_run or args.resume or args.audit:
            raise ValueError("--plan/--dry-run/--resume/--audit require --stage stage5")
        if not args.config or not args.out:
            raise ValueError("stages 1-4 require --config and --out")
        config = load_config(args.config)
        if (args.scheme or args.compare) and args.stage != "stage4":
            raise ValueError("--scheme/--compare require --stage stage4")
        if args.scheme and args.compare:
            raise ValueError("choose one scheme or the five-scheme comparison")
        if args.compare:
            from .experiments import run_comparison
            table = run_comparison(config, args.out, trace_base_dir=Path(args.config).resolve().parent, timeline=args.timeline)
            print(table[["scheme", "layer_makespan_s", "pepd", "wireless_payload_bytes", "speedup_vs_C"]].to_string(index=False))
            return 0
        if args.stage == "stage4":
            from .scheduled_layer import effective_config, run_scheduled
            result = run_scheduled(effective_config(config, args.scheme), trace_base_dir=Path(args.config).resolve().parent)
        elif args.stage == "stage3":
            from .layer import run_layer
            result = run_layer(config, trace_base_dir=Path(args.config).resolve().parent)
        elif args.stage == "stage2":
            from .pipeline import run_dispatch_compute
            result = run_dispatch_compute(config, trace_base_dir=Path(args.config).resolve().parent)
        else:
            result = run_reference(config)
        output = result.export(args.out)
        if args.timeline:
            from .plots import plot_layer_timeline, plot_pipeline_timeline, plot_timeline, plot_scheduled_timeline
            plotter = {"stage1": plot_timeline, "stage2": plot_pipeline_timeline, "stage3": plot_layer_timeline,
                       "stage4": plot_scheduled_timeline}[args.stage]
            plotter(result, output / "timeline.png")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result.summary, sort_keys=True, indent=2, allow_nan=False))
    return 0
