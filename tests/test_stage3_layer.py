from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from vdie_moe_sim.combine import synthetic_expert_output
from vdie_moe_sim.config import BatchingConfig, CombineConfig, load_config
from vdie_moe_sim.engine import LayerEventType as L
from vdie_moe_sim.layer import run_layer
from vdie_moe_sim.pipeline import run_dispatch_compute
from vdie_moe_sim.simulator import run_reference

ROOT = Path(__file__).resolve().parents[1]


class LayerTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/stage3_tiny.json")

    def test_singleton_matches_stage1_22_us(self):
        config = load_config(ROOT / "configs/stage1_tiny.json")
        reference, layer = run_reference(config), run_layer(config)
        self.assertEqual(reference.completions, layer.completions)
        self.assertEqual(reference.transmissions, layer.transmissions)
        self.assertAlmostEqual(layer.summary["layer_makespan_s"], 22e-6, delta=1e-18)

    def test_local_combine_on_hand_oracle(self):
        result = run_layer(self.config)
        # Dispatch 0..5; remote experts 5..7 / 7..9; Local Reduce 9..10;
        # Return 10..15; Source Reduce 15..16; Base 16..18 microseconds.
        self.assertAlmostEqual(result.summary["layer_makespan_s"], 18e-6, delta=1e-18)
        np.testing.assert_allclose([(t.start_s, t.end_s) for t in result.transmissions],
                                   [(0, 5e-6), (10e-6, 15e-6)], rtol=0, atol=1e-18)
        self.assertEqual(result.summary["return_flows"], 1)
        self.assertEqual(result.summary["return_payload_bytes"], 8)
        self.assertEqual(result.summary["return_bytes_saved_by_local_combine"], 8)
        self.assertEqual(sorted(p.expert_ids for p in result.partials), [(0,), (2, 3)])
        self.assertEqual(len([j for j in result.jobs if j.kind == "weight"]), 3)
        self.assertEqual(len([j for j in result.jobs if j.kind == "local_reduce"]), 1)
        self.assertAlmostEqual(result.outputs[0].ready_s, 16e-6, delta=1e-18)

    def test_local_combine_off_bytes_and_numeric_equivalence(self):
        on = run_layer(self.config)
        off = run_layer(replace(self.config, combine=replace(self.config.combine, local_combine=False)))
        self.assertEqual(off.summary["return_flows"], 2)
        self.assertEqual(off.summary["return_payload_bytes"], 16)
        self.assertEqual(off.summary["return_bytes_saved_by_local_combine"], 0)
        self.assertAlmostEqual(off.summary["layer_makespan_s"], 20e-6, delta=1e-18)
        self.assertFalse(any(j.kind == "local_reduce" for j in off.jobs))
        self.assertEqual(on.workload.digest, off.workload.digest)
        self.assertEqual(on.config.phy, off.config.phy)
        np.testing.assert_allclose(on.outputs[0].vector, off.outputs[0].vector, rtol=1e-14, atol=1e-14)

    def test_high_reduce_cost_can_make_combine_slower(self):
        slow = run_layer(replace(self.config, combine=replace(self.config.combine, local_reduce_s=100e-6)))
        off = run_layer(replace(self.config, combine=replace(self.config.combine, local_combine=False)))
        self.assertAlmostEqual(slow.summary["layer_makespan_s"], 117e-6, delta=1e-18)
        self.assertGreater(slow.summary["layer_makespan_s"], off.summary["layer_makespan_s"])

    def test_return_feedback_changes_future_dispatch_and_compute(self):
        config = load_config(ROOT / "configs/stage3_return_feedback.json")
        partial = run_dispatch_compute(config)
        full = run_layer(config)
        self.assertEqual([t.flow.flow_type for t in full.transmissions], ["dispatch", "return", "dispatch", "return"])
        np.testing.assert_allclose([(t.start_s, t.end_s) for t in full.transmissions],
                                   [(0, 5e-6), (7e-6, 12e-6), (12e-6, 17e-6), (19e-6, 24e-6)], rtol=0, atol=1e-18)
        self.assertAlmostEqual(partial.completions[1].completed_s, 15e-6, delta=1e-18)
        self.assertAlmostEqual(full.front.completions[1].completed_s, 19e-6, delta=1e-18)
        self.assertAlmostEqual(full.summary["layer_makespan_s"], 24e-6, delta=1e-18)

    def test_source_waits_for_slowest_relevant_die_and_return(self):
        config = replace(self.config, model=replace(self.config.model, num_experts=6),
                         topology=replace(self.config.topology, num_dies=3),
                         workload=replace(self.config.workload, expert_ids=(2, 3, 4)),
                         combine=replace(self.config.combine, local_reduce_s=0),
                         timing=replace(self.config.timing, source_combine_s=0, source_to_base_s=0))
        result = run_layer(config)
        returns = [t for t in result.transmissions if t.flow.flow_type == "return"]
        self.assertEqual([t.flow.src for t in returns], [2, 1])
        np.testing.assert_allclose([(t.start_s, t.end_s) for t in returns],
                                   [(7e-6, 12e-6), (12e-6, 17e-6)], rtol=0, atol=1e-18)
        source_job = next(j for j in result.jobs if j.kind == "source_reduce")
        self.assertAlmostEqual(source_job.ready_s, 17e-6, delta=1e-18)

    def test_all_local_has_no_rf_or_self_return(self):
        config = replace(self.config, topology=replace(self.config.topology, num_dies=1))
        result = run_layer(config)
        self.assertEqual(result.transmissions, ())
        self.assertTrue(all(d.flow_id is None for d in result.deliveries))
        self.assertEqual(result.summary["wireless_payload_bytes"], 0)
        self.assertAlmostEqual(result.summary["layer_makespan_s"], 10e-6, delta=1e-18)

    def test_no_global_barrier_between_tokens(self):
        base = load_config(ROOT / "configs/stage2_batching.json")
        config = replace(base, workload=replace(base.workload, num_tokens=2, interarrival_s=100e-6),
                         batching=BatchingConfig(0, 16),
                         timing=replace(base.timing, source_combine_s=1e-6, source_to_base_s=2e-6))
        result = run_layer(config)
        np.testing.assert_allclose([c.completed_s for c in result.completions], [5e-6, 105e-6], rtol=0, atol=1e-18)
        self.assertLess(result.completions[0].completed_s, result.completions[1].token.arrival_s)

    def test_shared_combine_unit_is_separate_from_expert_compute(self):
        base = load_config(ROOT / "configs/stage2_batching.json")
        config = replace(base, workload=replace(base.workload, num_tokens=2, interarrival_s=0),
                         batching=BatchingConfig(0, 16), combine=CombineConfig(weight_s=4e-6),
                         timing=replace(base.timing, source_combine_s=1e-6))
        result = run_layer(config)
        np.testing.assert_allclose([(x.start_s, x.end_s) for x in result.executions],
                                   [(0, 2e-6), (2e-6, 4e-6)], rtol=0, atol=1e-18)
        np.testing.assert_allclose([c.completed_s for c in result.completions], [11e-6, 12e-6], rtol=0, atol=1e-18)
        source0 = next(j for j in result.jobs if j.kind == "source_reduce" and j.token_id == 0)
        self.assertAlmostEqual(source0.start_s - source0.ready_s, 4e-6, delta=1e-18)
        self.assertAlmostEqual(result.combine_utilization()[0]["utilization"], 10 / 12)
        ideal = run_layer(replace(config, combine=replace(config.combine, resource_model="independent")))
        np.testing.assert_allclose([c.completed_s for c in ideal.completions], [7e-6, 9e-6], rtol=0, atol=1e-18)
        self.assertLessEqual(ideal.combine_utilization()[0]["utilization"], 1)
        self.assertGreater(ideal.combine_utilization()[0]["mean_active_jobs"], 1)

    def test_gate_weights_applied_once_and_not_renormalized(self):
        path = ROOT / "configs/stage2_trace.json"
        result = run_layer(load_config(path), trace_base_dir=path.parent)
        token = next(t for t in result.workload.tokens if t.token_id == 11)
        self.assertEqual(token.gate_weights, (2.5, 0.5))
        raw0 = synthetic_expert_output(11, 1, 4, 0)
        raw1 = synthetic_expert_output(11, 2, 4, 0)
        actual = next(o.vector for o in result.outputs if o.token_id == 11)
        np.testing.assert_allclose(actual, 2.5 * raw0 + 0.5 * raw1, rtol=0, atol=1e-14)

    def test_zero_weight_retains_assignment_and_contribution_count(self):
        config = replace(self.config, workload=replace(self.config.workload, gate_weights=(0, 0, 1)))
        result = run_layer(config)
        self.assertEqual(result.summary["expert_assignments"], 3)
        self.assertEqual(result.summary["return_flows"], 1)
        np.testing.assert_array_equal(result.outputs[0].vector, synthetic_expert_output(0, 3, 4, 7))

    def test_wrong_double_weighting_is_detected_by_independent_oracle(self):
        def wrong(vector, weight, dimension):
            return np.asarray(vector) * weight * weight
        with patch("vdie_moe_sim.layer.weighted_vector", side_effect=wrong):
            with self.assertRaisesRegex(ValueError, "weighted-sum oracle"):
                run_layer(self.config)

    def test_fp8_affects_bytes_but_not_synthetic_numeric_semantics(self):
        bf16 = run_layer(self.config)
        fp8 = run_layer(replace(self.config, model=replace(self.config.model, activation_precision="fp8")))
        self.assertEqual(fp8.summary["wireless_payload_bytes"], 8)
        self.assertAlmostEqual(fp8.summary["layer_makespan_s"], 14e-6, delta=1e-18)
        np.testing.assert_array_equal(bf16.outputs[0].vector, fp8.outputs[0].vector)

    def test_arrival_offset_and_full_latency_metrics(self):
        result = run_layer(replace(self.config, workload=replace(self.config.workload, arrival_s=0.125)))
        s = result.summary
        self.assertAlmostEqual(s["layer_makespan_s"], 18e-6, delta=1e-15)
        self.assertAlmostEqual(s["token_latency_p99_s"], 18e-6, delta=1e-15)
        self.assertAlmostEqual(s["thz_utilization"], 10 / 18, places=9)
        self.assertAlmostEqual(s["throughput_tokens_per_s"], 1 / 18e-6, places=5)

    def test_stale_deadline_does_not_extend_final_base_window(self):
        base = load_config(ROOT / "configs/stage2_batching.json")
        config = replace(base, workload=replace(base.workload, num_tokens=2, interarrival_s=0),
                         batching=BatchingConfig(100e-6, 2))
        result = run_layer(config)
        self.assertGreater(result.events[-1].time_s, result.completions[-1].completed_s)
        self.assertAlmostEqual(result.summary["layer_makespan_s"], 2e-6, delta=1e-18)

    def test_subcarrier_count_does_not_duplicate_dispatch_return_budget(self):
        times = []
        for count in (16, 32, 128):
            result = run_layer(replace(self.config, phy=replace(self.config.phy, num_subcarriers=count)))
            times.append(result.summary["layer_makespan_s"])
            self.assertTrue(all(t.rate_bps == 16e6 for t in result.transmissions))
            self.assertLessEqual(result.summary["pepd"], 1)
        self.assertEqual(len(set(times)), 1)

    def test_paper_shape_128_top8_numeric_and_count_conservation(self):
        config = load_config(ROOT / "configs/stage3_zipf.json")
        result = run_layer(config)
        self.assertEqual(len(result.front.inputs), 1024)
        self.assertEqual(len(result.completions), 128)
        expected_returns = sum(len({e // 16 for e in t.expert_ids} - {t.source_die}) for t in result.workload.tokens)
        self.assertEqual(result.summary["return_flows"], expected_returns)
        self.assertEqual(result.summary["return_payload_bytes"], expected_returns * 14336)
        self.assertLess(result.summary["verification_max_abs_error"], 1e-10)
        self.assertEqual(np.stack([o.vector for o in result.outputs]).shape, (128, 7168))

    def test_small_corner_cases_do_not_drop_experts_or_partials(self):
        for dies, k, combine_on in ((1, 4, True), (1, 4, False), (2, 4, False), (4, 1, True), (4, 4, True)):
            with self.subTest(dies=dies, k=k, combine_on=combine_on):
                config = replace(self.config, model=replace(self.config.model, num_experts=8, top_k=k),
                    topology=replace(self.config.topology, num_dies=dies), batching=BatchingConfig(1e-6, 3),
                    workload=replace(self.config.workload, num_tokens=8, source_policy="round_robin", routing="uniform",
                                     expert_ids=(), gate_weights=()),
                    combine=replace(self.config.combine, local_combine=combine_on, weight_s=0.1e-6))
                result = run_layer(config)
                self.assertEqual(len(result.front.inputs), 8 * k)
                self.assertEqual(len(result.outputs), 8)

    def test_corrupt_partials_deliveries_outputs_and_final_events_rejected(self):
        result = run_layer(self.config)
        remote = next(p for p in result.partials if p.die_id != 0)
        returned = next(r for r in result.flow_ready if r.partial_id == remote.partial_id)
        source = next(j for j in result.jobs if j.kind == "source_reduce")
        corruptions = [replace(result, partials=(*result.partials, result.partials[0])),
            replace(result, deliveries=result.deliveries[:-1]),
            replace(result, outputs=(replace(result.outputs[0], vector=result.outputs[0].vector + 1),)),
            replace(result, completions=(replace(result.completions[0], completed_s=16e-6),)),
            replace(result, events=result.events[:-1]),
            replace(result, intervals=(*result.intervals, result.intervals[-1])),
            replace(result, jobs=tuple(replace(j, ready_s=0) if j == source else j for j in result.jobs)),
            replace(result, flow_ready=tuple(replace(r, ready_s=0) if r == returned else r for r in result.flow_ready)),
            replace(result, partials=tuple(replace(p, vector=p.vector + 1) if p.partial_id == remote.partial_id else p
                                          for p in result.partials))]
        for i, corrupted in enumerate(corruptions):
            with self.subTest(case=i), self.assertRaises(ValueError):
                corrupted.audit()


class LayerExportTests(unittest.TestCase):
    def test_repeatable_csv_json_array_exports(self):
        config = load_config(ROOT / "configs/stage3_tiny.json")
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            for path in (a, b):
                run_layer(config).export(path)
            self.assertEqual({p.name for p in a.iterdir()}, {p.name for p in b.iterdir()})
            for path in a.iterdir():
                self.assertEqual(path.read_bytes(), (b / path.name).read_bytes(), path.name)
            self.assertEqual(np.load(a / "outputs.npy", allow_pickle=False).shape, (1, 4))
            self.assertEqual(len(pd.read_csv(a / "partials.csv")), 2)
            self.assertEqual(len(pd.read_csv(a / "output_validation.csv")), 1)
            manifest = json.loads((a / "manifest.json").read_text())
            self.assertEqual(manifest["endpoint"], "all_tokens_returned_to_base")
            self.assertNotIn("Return", manifest["not_implemented"])
            self.assertIn("quantization", " ".join(manifest["not_implemented"]))
            self.assertAlmostEqual(pd.read_csv(a / "compute_utilization.csv").iloc[1]["utilization"], 4 / 18)

    def test_all_local_empty_rf_csv_keeps_schema(self):
        config = load_config(ROOT / "configs/stage2_batching.json")
        with tempfile.TemporaryDirectory() as temp:
            run_layer(config).export(temp)
            self.assertTrue(pd.read_csv(Path(temp) / "flows.csv").empty)
            self.assertEqual(len(pd.read_csv(Path(temp) / "tokens.csv")), 5)

    def test_trace_replay_reproduces_schedule_and_values(self):
        config = load_config(ROOT / "configs/stage3_return_feedback.json")
        original = run_layer(config)
        with tempfile.TemporaryDirectory() as temp:
            original.export(temp)
            replay_config = replace(config, workload=replace(config.workload, routing="trace", source_policy="trace",
                interarrival_s=0, expert_ids=(), gate_weights=(), trace_file="routing_trace.json", seed=999))
            replay = run_layer(replay_config, trace_base_dir=temp)
        self.assertEqual(original.workload.digest, replay.workload.digest)
        self.assertEqual(original.events, replay.events)
        self.assertEqual(original.transmissions, replay.transmissions)
        for a, b in zip(original.outputs, replay.outputs):
            np.testing.assert_array_equal(a.vector, b.vector)

    def test_cli_stage3_and_timeline(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            process = subprocess.run([sys.executable, "-m", "vdie_moe_sim", "--stage", "stage3", "--config",
                str(ROOT / "configs/stage3_tiny.json"), "--out", str(out), "--timeline"],
                cwd=temp, capture_output=True, text=True, timeout=60)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)["stage"], "stage3_end_to_end")
            self.assertEqual((out / "timeline.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
