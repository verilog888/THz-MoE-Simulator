from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from vdie_moe_sim.config import BatchingConfig, load_config
from vdie_moe_sim.engine import EventType, PipelineEventType as E
from vdie_moe_sim.metrics import summarize
from vdie_moe_sim.pipeline import run_dispatch_compute
from vdie_moe_sim.simulator import run_reference
from vdie_moe_sim.thz_phy import THzLink

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/stage2_tiny.json")

    def test_singleton_prefix_matches_stage1_oracle(self):
        config = load_config(ROOT / "configs/stage1_tiny.json")
        reference = run_reference(config)
        pipeline = run_dispatch_compute(config)
        expert_done = next(e.time_s for e in reference.events if e.kind == EventType.EXPERT_COMPUTE_DONE)
        self.assertAlmostEqual(pipeline.summary["dispatch_compute_makespan_s"], 12e-6, delta=1e-18)
        self.assertEqual(pipeline.completions[0].completed_s, expert_done)
        self.assertEqual(pipeline.transmissions[0], reference.transmissions[0])

    def test_mixed_local_remote_hand_oracle(self):
        result = run_dispatch_compute(self.config)
        self.assertEqual(len(result.inputs), 4)
        self.assertEqual(len(result.transmissions), 2)
        np.testing.assert_allclose([c.completed_s for c in result.completions], [7e-6, 12e-6], rtol=0, atol=1e-18)
        for die, expected in ((0, [(0, 2), (2, 4)]), (1, [(5, 7), (10, 12)])):
            actual = [(x.start_s * 1e6, x.end_s * 1e6) for x in result.executions if x.batch.die_id == die]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        self.assertEqual([(t.flow.src, t.flow.dsts, t.flow.payload_bytes) for t in result.transmissions],
                         [(0, (1,), 8), (0, (1,), 8)])

    def test_partial_metrics_and_busy_integrals(self):
        result = run_dispatch_compute(self.config)
        metrics = result.summary
        self.assertNotIn("layer_makespan_s", metrics)
        self.assertNotIn("token_latency_p50_s", metrics)
        self.assertNotIn("throughput_tokens_per_s", metrics)
        self.assertAlmostEqual(metrics["experts_done_latency_p50_s"], 9.5e-6, delta=1e-18)
        self.assertAlmostEqual(metrics["experts_done_latency_p95_s"], 11.75e-6, delta=1e-18)
        self.assertAlmostEqual(metrics["experts_done_latency_p99_s"], 11.95e-6, delta=1e-18)
        self.assertAlmostEqual(metrics["thz_utilization"], 10 / 12)
        self.assertAlmostEqual(metrics["pepd"], 10 / 12)
        self.assertAlmostEqual(metrics["mean_input_to_compute_wait_s"], 0.5e-6, delta=1e-18)
        _, dies = summarize(result.completions, result.transmissions, result.intervals, THzLink(self.config.phy), 2)
        self.assertEqual(len(dies), 2)
        for die in dies:
            self.assertAlmostEqual(die["busy_s"], 4e-6, delta=1e-18)
            self.assertAlmostEqual(die["utilization"], 4 / 12)

    def test_broadcast_duplicates_destinations_not_payload(self):
        model = replace(self.config.model, num_experts=8, top_k=4)
        workload = replace(self.config.workload, num_tokens=1, expert_ids=(0, 2, 3, 4), gate_weights=(0.25,) * 4)
        config = replace(self.config, model=model, topology=replace(self.config.topology, num_dies=4), workload=workload)
        result = run_dispatch_compute(config)
        self.assertEqual(len(result.transmissions), 1)
        self.assertEqual(result.transmissions[0].flow.dsts, (1, 2))
        self.assertEqual(result.transmissions[0].flow.payload_bytes, 8)
        self.assertEqual(result.summary["remote_assignments"], 3)
        self.assertAlmostEqual(result.summary["dispatch_compute_makespan_s"], 9e-6, delta=1e-18)
        # Two experts on Die 1 must serialize; Die 2 can compute concurrently.
        self.assertEqual(sum(x.batch.die_id == 1 for x in result.executions), 2)
        for r in result.inputs:
            self.assertAlmostEqual(r.ready_s, 0 if r.local else 5e-6, delta=1e-18)

    def test_same_die_different_experts_share_compute(self):
        config = replace(self.config, workload=replace(self.config.workload, num_tokens=1, expert_ids=(0, 1)))
        result = run_dispatch_compute(config)
        self.assertEqual(result.transmissions, ())
        self.assertAlmostEqual(result.summary["dispatch_compute_makespan_s"], 4e-6, delta=1e-18)
        self.assertEqual([x.batch.expert_id for x in result.executions], [0, 1])

    def test_different_dies_compute_in_parallel_with_dispatch(self):
        config = replace(self.config, workload=replace(self.config.workload, num_tokens=1),
                         compute=replace(self.config.compute, memory_bytes_per_s=2.4e6))
        result = run_dispatch_compute(config)
        # Local compute [0,10], RF [0,5], remote compute [5,15] microseconds.
        np.testing.assert_allclose([(x.start_s, x.end_s) for x in result.executions],
                                   [(0, 10e-6), (5e-6, 15e-6)], rtol=0, atol=1e-18)
        self.assertAlmostEqual(result.summary["dispatch_compute_makespan_s"], 15e-6, delta=1e-18)

    def test_all_local_batching_and_reuse_hand_oracle(self):
        config = load_config(ROOT / "configs/stage2_batching.json")
        result = run_dispatch_compute(config)
        self.assertEqual(result.transmissions, ())
        self.assertEqual(result.summary["wireless_payload_bytes"], 0)
        self.assertEqual(result.summary["pepd"], 0)
        self.assertEqual([x.batch.request_ids for x in result.executions], [(0, 1, 2), (3, 4)])
        self.assertEqual([x.weight_reads for x in result.executions], [2, 1])
        np.testing.assert_allclose([(x.start_s, x.end_s) for x in result.executions],
                                   [(1e-6, 5e-6), (5e-6, 7e-6)], rtol=0, atol=1e-18)
        self.assertEqual(result.summary["local_weight_bytes_read"], 72)
        self.assertAlmostEqual(result.summary["dispatch_compute_makespan_s"], 7e-6, delta=1e-18)
        # Five member tokens consume six microseconds of one Die, not 3*4+2*2.
        busy = sum(x.end_s - x.start_s for x in result.executions)
        self.assertAlmostEqual(busy, 6e-6, delta=1e-18)

    def test_weight_reuse_parameter_changes_batch_memory_cost(self):
        config = load_config(ROOT / "configs/stage2_batching.json")
        once = run_dispatch_compute(replace(config, compute=replace(config.compute, weight_reuse_tokens=None)))
        self.assertEqual(once.summary["weight_reads"], 2)
        self.assertAlmostEqual(once.summary["dispatch_compute_makespan_s"], 6e-6, delta=1e-18)
        singleton = run_dispatch_compute(replace(config, batching=BatchingConfig(0, 3)))
        self.assertEqual(singleton.summary["weight_reads"], 5)
        self.assertAlmostEqual(singleton.summary["dispatch_compute_makespan_s"], 10e-6, delta=1e-18)

    def test_early_cap_stale_timer_does_not_extend_makespan(self):
        base = load_config(ROOT / "configs/stage2_batching.json")
        config = replace(base, workload=replace(base.workload, num_tokens=2, interarrival_s=0),
                         batching=BatchingConfig(100e-6, 2))
        result = run_dispatch_compute(config)
        self.assertEqual(len(result.executions), 1)
        self.assertEqual(result.summary["ignored_batch_deadlines"], 1)
        self.assertEqual(result.events[-1].kind, E.BATCH_DEADLINE)
        self.assertAlmostEqual(result.events[-1].time_s, 100e-6, delta=1e-18)
        self.assertAlmostEqual(result.summary["dispatch_compute_makespan_s"], 2e-6, delta=1e-18)

    def test_arrivals_exactly_at_deadline_do_not_enter_previous_batch(self):
        base = load_config(ROOT / "configs/stage2_batching.json")
        config = replace(base, workload=replace(base.workload, num_tokens=2, interarrival_s=1e-6),
                         batching=BatchingConfig(1e-6, 4))
        result = run_dispatch_compute(config)
        self.assertEqual([x.batch.request_ids for x in result.executions], [(0,), (1,)])
        self.assertAlmostEqual(result.summary["dispatch_compute_makespan_s"], 5e-6, delta=1e-18)

    def test_open_window_does_not_block_other_ready_expert(self):
        rows = [{"token_id": i, "source_die": 0, "arrival_s": arrival,
                 "expert_ids": [expert], "gate_weights": [1]}
                for i, arrival, expert in ((0, 0, 0), (1, 1e-6, 1), (2, 2e-6, 1))]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.json"
            path.write_text(json.dumps({"schema_version": 1, "tokens": rows}))
            config = replace(self.config, model=replace(self.config.model, num_experts=2, top_k=1),
                             topology=replace(self.config.topology, num_dies=1), batching=BatchingConfig(10e-6, 2),
                             workload=replace(self.config.workload, num_tokens=3, routing="trace", source_policy="trace",
                                              trace_file="trace.json", expert_ids=(), gate_weights=()))
            result = run_dispatch_compute(config, trace_base_dir=temp)
        self.assertEqual([x.batch.expert_id for x in result.executions], [1, 0])
        np.testing.assert_allclose([(x.start_s, x.end_s) for x in result.executions],
                                   [(2e-6, 4e-6), (10e-6, 12e-6)], rtol=0, atol=1e-18)

    def test_arrival_offset_does_not_inflate_latency(self):
        config = replace(self.config, workload=replace(self.config.workload, arrival_s=0.001))
        result = run_dispatch_compute(config)
        self.assertAlmostEqual(result.summary["dispatch_compute_makespan_s"], 12e-6, delta=1e-15)
        self.assertAlmostEqual(result.completions[-1].completed_s, 0.001012, delta=1e-15)

    def test_same_total_bandwidth_for_different_subcarrier_counts(self):
        values = []
        for count in (16, 32, 64, 128, 256):
            result = run_dispatch_compute(replace(self.config, phy=replace(self.config.phy, num_subcarriers=count)))
            values.append(result.summary["dispatch_compute_makespan_s"])
            self.assertTrue(all(t.rate_bps == 16e6 for t in result.transmissions))
        self.assertEqual(len(set(values)), 1)

    def test_pdf_shape_top8_assignment_conservation(self):
        config = load_config(ROOT / "configs/stage2_zipf.json")
        result = run_dispatch_compute(config)
        self.assertEqual(len(result.inputs), 128 * 8)
        counts = Counter(r.token_id for r in result.inputs)
        self.assertEqual(set(counts.values()), {8})
        self.assertEqual(len(result.completions), 128)
        self.assertEqual(sum(len(x.batch.request_ids) for x in result.executions), 1024)
        self.assertEqual(result.summary["total_bandwidth_hz"], 32e9)
        self.assertLessEqual(result.summary["pepd"], 1)

    def test_seeded_parameter_corner_cases(self):
        # Small cross-product exercises all-local, all-experts, skew and multiple windows.
        for dies in (1, 2, 8):
            for k, skew, window, cap in ((1, 0, 0, 1), (4, 1, 2e-6, 3), (16, 2, 1e-6, 16)):
                with self.subTest(dies=dies, k=k, skew=skew, window=window):
                    config = replace(self.config, model=replace(self.config.model, num_experts=16, top_k=k),
                        topology=replace(self.config.topology, num_dies=dies), batching=BatchingConfig(window, cap),
                        workload=replace(self.config.workload, num_tokens=16, source_policy="round_robin",
                                         routing="zipf", zipf_skew=skew, expert_ids=(), gate_weights=()))
                    result = run_dispatch_compute(config)
                    self.assertEqual(len(result.inputs), 16 * k)
                    self.assertEqual(len(result.completions), 16)
                    if dies == 1:
                        self.assertEqual(len(result.transmissions), 0)

    def test_causal_and_accounting_corruption_rejected(self):
        result = run_dispatch_compute(self.config)
        tx = result.transmissions[0]
        execution = result.executions[0]
        bad = [replace(result, inputs=result.inputs[:-1]),
               replace(result, inputs=(*result.inputs, result.inputs[0])),
               replace(result, executions=(*result.executions, execution)),
               replace(result, events=result.events[:-1]),
               replace(result, intervals=(*result.intervals, result.intervals[0])),
               replace(result, transmissions=(replace(tx, flow=replace(tx.flow, dsts=())), *result.transmissions[1:])),
               replace(result, executions=(replace(execution, start_s=execution.end_s), *result.executions[1:])),
               replace(result, completions=(replace(result.completions[0], completed_s=2e-6), result.completions[1]))]
        remote = next(r for r in result.inputs if not r.local)
        bad.append(replace(result, inputs=tuple(replace(r, ready_s=0) if r == remote else r for r in result.inputs)))
        for index, corrupted in enumerate(bad):
            with self.subTest(index=index), self.assertRaises(ValueError):
                corrupted.audit()


class PipelineExportTests(unittest.TestCase):
    def test_repeated_run_exports_identical(self):
        config = load_config(ROOT / "configs/stage2_tiny.json")
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            for path in (a, b):
                run_dispatch_compute(config).export(path)
            for path in a.iterdir():
                self.assertEqual(path.read_bytes(), (b / path.name).read_bytes(), path.name)
            self.assertEqual(len(pd.read_csv(a / "requests.csv")), 4)
            self.assertEqual(len(pd.read_csv(a / "batches.csv")), 4)
            summary = pd.read_csv(a / "summary.csv")
            self.assertNotIn("layer_makespan_s", summary.columns)
            manifest = json.loads((a / "manifest.json").read_text())
            self.assertIn("before_combine_return", manifest["endpoint"])
            self.assertIn("Return", manifest["not_implemented"])
            self.assertTrue(any(a["id"] == "A05" and "user_accepted" in a["status"] for a in manifest["assumptions"]))

    def test_all_local_empty_flows_csv_has_schema(self):
        config = load_config(ROOT / "configs/stage2_batching.json")
        with tempfile.TemporaryDirectory() as temp:
            result = run_dispatch_compute(config)
            result.export(temp)
            flows = pd.read_csv(Path(temp) / "flows.csv")
            self.assertTrue(flows.empty)
            self.assertIn("payload_bytes", flows.columns)
            compute = pd.read_csv(Path(temp) / "compute_utilization.csv")
            self.assertAlmostEqual(compute.iloc[0]["utilization"], 6 / 7)

    def test_generated_trace_replays_identical_workload_and_timing(self):
        config = load_config(ROOT / "configs/stage2_zipf.json")
        original = run_dispatch_compute(config)
        with tempfile.TemporaryDirectory() as temp:
            original.export(temp)
            replay_config = replace(config, workload=replace(config.workload, routing="trace", source_policy="trace",
                                       zipf_skew=0, trace_file="routing_trace.json"))
            replay = run_dispatch_compute(replay_config, trace_base_dir=temp)
            self.assertEqual(replay.workload.digest, original.workload.digest)
            self.assertEqual(replay.transmissions, original.transmissions)
            self.assertEqual(replay.executions, original.executions)
            self.assertEqual(replay.completions, original.completions)

    def test_trace_row_order_does_not_affect_simulation(self):
        path = ROOT / "configs/stage2_trace.json"
        config = load_config(path)
        original = run_dispatch_compute(config, trace_base_dir=path.parent)
        with tempfile.TemporaryDirectory() as temp:
            data = original.workload.to_trace()
            data["tokens"].reverse()
            (Path(temp) / "trace.json").write_text(json.dumps(data))
            changed = replace(config, workload=replace(config.workload, trace_file="trace.json"))
            replay = run_dispatch_compute(changed, trace_base_dir=temp)
        self.assertEqual(original.events, replay.events)
        self.assertEqual(original.workload.digest, replay.workload.digest)

    def test_cli_resolves_trace_relative_to_config_from_another_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            process = subprocess.run([sys.executable, "-m", "vdie_moe_sim", "--stage", "stage2", "--config",
                                      str(ROOT / "configs/stage2_trace.json"), "--out", str(Path(temp) / "run")],
                                     cwd=temp, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)["stage"], "stage2_dispatch_compute")

    def test_timeline_renders_multicast_and_batches(self):
        from vdie_moe_sim.plots import plot_pipeline_timeline
        result = run_dispatch_compute(load_config(ROOT / "configs/stage2_tiny.json"))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "timeline.png"
            plot_pipeline_timeline(result, path)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(path.stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
