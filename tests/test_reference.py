from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from vdie_moe_sim.config import config_from_dict, load_config
from vdie_moe_sim.engine import EventType
from vdie_moe_sim.metrics import audit_transmissions, summarize
from vdie_moe_sim.moe import Flow, Interval, Token, TokenCompletion, Transmission
from vdie_moe_sim.simulator import run_reference
from vdie_moe_sim.thz_phy import THzLink

ROOT = Path(__file__).resolve().parents[1]


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/stage1_tiny.json")
        self.result = run_reference(self.config)

    def test_hand_calculated_latency_and_complete_causal_chain(self):
        # Independent oracle, microseconds: base 2 + gate 3 + RF 5 + compute 2
        # + singleton bypass 0 + RF 5 + source combine 1 + base 4 = 22.
        expected_us = [2, 5, 5, 10, 12, 12, 12, 17, 18, 22]
        self.assertEqual([e.kind for e in self.result.events], list(EventType))
        np.testing.assert_allclose([e.time_s for e in self.result.events],
                                   np.array(expected_us) * 1e-6, rtol=0, atol=1e-18)
        self.assertAlmostEqual(self.result.summary["layer_makespan_s"], 22e-6, delta=1e-18)
        self.assertEqual(len(self.result.transmissions), 2)
        self.assertEqual(self.result.transmissions[0].flow.src, 0)
        self.assertEqual(self.result.transmissions[1].flow.dsts, (0,))

    def test_hand_calculated_integral_metrics(self):
        metrics = self.result.summary
        self.assertAlmostEqual(metrics["thz_utilization"], 10 / 22)
        self.assertAlmostEqual(metrics["pepd"], 10 / 22)
        self.assertAlmostEqual(metrics["pepd_payload_only"], 8 / 22)
        self.assertAlmostEqual(metrics["throughput_tokens_per_s"], 1 / 22e-6)
        self.assertEqual(metrics["wireless_payload_bytes"], 16)
        self.assertAlmostEqual(metrics["allocated_capacity_bits"], 160)
        self.assertAlmostEqual(metrics["bandwidth_waste"], 0.2)
        self.assertEqual(metrics["peak_concurrent_flows"], 1)
        for percentile in (50, 95, 99):
            self.assertAlmostEqual(metrics[f"token_latency_p{percentile}_s"], 22e-6, delta=1e-18)
        _, compute = summarize(self.result.completions, self.result.transmissions, self.result.intervals,
                                THzLink(self.config.phy), 2)
        self.assertEqual(compute[0]["utilization"], 0)
        self.assertAlmostEqual(compute[1]["utilization"], 2 / 22)

    def test_arrival_offset_is_not_token_latency(self):
        config = replace(self.config, workload=replace(self.config.workload, arrival_s=0.125))
        result = run_reference(config)
        self.assertAlmostEqual(result.completions[0].completed_s, 0.125022, delta=1e-15)
        self.assertAlmostEqual(result.summary["layer_makespan_s"], 22e-6, delta=1e-15)
        self.assertAlmostEqual(result.summary["token_latency_p50_s"], 22e-6, delta=1e-15)

    def test_zero_delay_stages_preserve_events(self):
        config = replace(self.config, timing=replace(self.config.timing, base_to_source_s=0,
                          gate_s=0, source_combine_s=0, source_to_base_s=0))
        result = run_reference(config)
        self.assertEqual([e.kind for e in result.events], list(EventType))
        self.assertEqual([e.time_s for e in result.events[:3]], [0, 0, 0])
        self.assertAlmostEqual(result.summary["layer_makespan_s"], 12e-6, delta=1e-18)

    def test_fp8_halves_payload_not_whole_layer(self):
        result = run_reference(replace(self.config, model=replace(self.config.model, activation_precision="fp8")))
        self.assertEqual(result.summary["wireless_payload_bytes"], 8)
        self.assertAlmostEqual(result.summary["layer_makespan_s"], 18e-6, delta=1e-18)

    def test_pdf_dimensions_reference_hand_calculation(self):
        result = run_reference(load_config(ROOT / "configs/stage1_reference.json"))
        # Memory: 44,040,192 / 1e12 s; two RF packets: 2 * (0.1 + 2.24) us.
        self.assertAlmostEqual(result.summary["layer_makespan_s"], 48.720192e-6, delta=1e-18)
        self.assertEqual(result.summary["wireless_payload_bytes"], 28672)

    def test_uniform_run_is_repeatable_without_resampling_local_hits(self):
        # Seed 42 in the pinned NumPy gives expert 0 of 2; source 1 makes it remote.
        workload = replace(self.config.workload, routing="uniform", expert_ids=(), gate_weights=(), source_die=1)
        config = replace(self.config, workload=workload)
        self.assertEqual(run_reference(config), run_reference(config))
        with self.assertRaisesRegex(ValueError, "remote expert"):
            run_reference(replace(config, workload=replace(workload, source_die=0)))

    def test_reference_limits_are_errors_not_silent_fallbacks(self):
        invalid = [replace(self.config, workload=replace(self.config.workload, num_tokens=2)),
                   replace(self.config, model=replace(self.config.model, top_k=2),
                           workload=replace(self.config.workload, expert_ids=(0, 1), gate_weights=(0.5, 0.5))),
                   replace(self.config, workload=replace(self.config.workload, expert_ids=(0,)))]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                run_reference(config)

    def test_float_time_resolution_failure_is_explicit(self):
        config = replace(self.config, workload=replace(self.config.workload, arrival_s=1e100))
        with self.assertRaisesRegex(ValueError, "time precision"):
            run_reference(config)

    def test_corrupt_causal_trace_is_detected(self):
        bad_events = list(self.result.events)
        bad_events[4] = replace(bad_events[4], time_s=9e-6)  # compute finishes before dispatch
        invalid = [replace(self.result, events=tuple(bad_events)),
                   replace(self.result, events=self.result.events[:-1]),
                   replace(self.result, completions=(replace(self.result.completions[0], completed_s=18e-6),)),
                   replace(self.result, intervals=self.result.intervals[:-1])]
        for result in invalid:
            with self.subTest(result=result), self.assertRaises(ValueError):
                _ = result.summary

    def test_exports_are_reproducible_and_self_describing(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "a", Path(temp) / "b"
            self.result.export(a)
            run_reference(self.config).export(b)
            self.assertEqual({p.name for p in a.iterdir()}, {p.name for p in b.iterdir()})
            for path in a.iterdir():
                self.assertEqual(path.read_bytes(), (b / path.name).read_bytes(), path.name)
            summary = pd.read_csv(a / "summary.csv")
            self.assertEqual(len(summary), 1)
            self.assertAlmostEqual(summary.iloc[0]["layer_makespan_s"], 22e-6, delta=1e-18)
            self.assertEqual(len(pd.read_csv(a / "events.csv")), 10)
            self.assertEqual(len(pd.read_csv(a / "flows.csv")), 2)
            self.assertEqual(len(pd.read_csv(a / "tokens.csv")), 1)
            manifest = json.loads((a / "manifest.json").read_text())
            self.assertEqual(manifest["config_sha256"], self.config.digest)
            self.assertEqual(manifest["workload_sha256"], self.result.workload_digest)
            self.assertIn("uncalibrated", manifest["model_status"])
            self.assertIn("CA-OFDMA", manifest["not_implemented"])
            self.assertTrue(any(a["id"] == "A06" for a in manifest["assumptions"]))
            with self.assertRaises(FileExistsError):
                self.result.export(a)

    def test_cli_runs_from_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            process = subprocess.run([sys.executable, "-m", "vdie_moe_sim", "--config",
                                      "configs/stage1_tiny.json", "--out", str(Path(temp) / "run")],
                                     cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertAlmostEqual(json.loads(process.stdout)["layer_makespan_s"], 22e-6, delta=1e-18)

    def test_timeline_is_rendered(self):
        from vdie_moe_sim.plots import plot_timeline
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "timeline.png"
            plot_timeline(self.result, path)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertGreater(path.stat().st_size, 10000)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / "configs/stage1_tiny.json").read_text())

    def test_unknown_fields_rejected(self):
        for section in (None, "phy", "compute"):
            with self.subTest(section=section):
                raw = json.loads(json.dumps(self.raw))
                target = raw if section is None else raw[section]
                target["misspelled_parameter"] = 1
                with self.assertRaises(ValueError):
                    config_from_dict(raw)

    def test_missing_required_hardware_values_rejected(self):
        del self.raw["compute"]["memory_bytes_per_s"]
        with self.assertRaises(ValueError):
            config_from_dict(self.raw)

    def test_invalid_fields_rejected(self):
        cases = [("model", "top_k", 3), ("model", "activation_precision", "int4"),
                 ("compute", "compute_flops_per_s", 0), ("compute", "weight_reuse_tokens", 0),
                 ("workload", "expert_ids", [2]), ("workload", "source_die", 2),
                 ("workload", "gate_weights", [0.2]), ("workload", "seed", True),
                 ("workload", "arrival_s", -1), ("model", "d_model", 4.0)]
        for section, name, value in cases:
            with self.subTest(section=section, name=name):
                raw = json.loads(json.dumps(self.raw))
                raw[section][name] = value
                with self.assertRaises(ValueError):
                    config_from_dict(raw)

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text('{"name":"a","name":"b"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                load_config(path)

    def test_config_hash_roundtrip_and_change(self):
        config = config_from_dict(self.raw)
        self.assertEqual(config.digest, config_from_dict(config.to_dict()).digest)
        self.raw["phy"]["total_bandwidth_hz"] *= 2
        self.assertNotEqual(config.digest, config_from_dict(self.raw).digest)


class MetricAuditTests(unittest.TestCase):
    def setUp(self):
        config = load_config(ROOT / "configs/stage1_tiny.json")
        self.link = THzLink(replace(config.phy, setup_s=0))
        self.flow = Flow(0, 0, 0, (1,), 8, "dispatch")
        self.tx = Transmission(self.flow, 0, 0, 8e-6, tuple(range(8)), 8e6)

    def test_ideal_same_die_multiflow_uses_disjoint_spectrum(self):
        other = replace(self.tx, flow=replace(self.flow, flow_id=1), subcarriers=tuple(range(8, 16)))
        self.assertEqual(audit_transmissions((self.tx, other), self.link, 2), 2)
        token = Token(0, 0, 0, 8, (1,), (1.0,))
        metrics, _ = summarize((TokenCompletion(token, 8e-6),), (self.tx, other), (), self.link, 2)
        self.assertAlmostEqual(metrics["pepd"], 2)
        self.assertAlmostEqual(metrics["thz_utilization"], 1)
        self.assertAlmostEqual(metrics["bandwidth_waste"], 0)

    def test_time_collision_is_rejected(self):
        other = replace(self.tx, flow=replace(self.flow, flow_id=1))
        with self.assertRaisesRegex(ValueError, "collide"):
            audit_transmissions((self.tx, other), self.link, 2)

    def test_touching_intervals_are_not_collisions(self):
        other = replace(self.tx, flow=replace(self.flow, flow_id=1), start_s=8e-6,
                        payload_start_s=8e-6, end_s=16e-6)
        self.assertEqual(audit_transmissions((self.tx, other), self.link, 2), 1)

    def test_capacity_and_lifetime_corruption_rejected(self):
        for tx in (replace(self.tx, rate_bps=16e6), replace(self.tx, end_s=4e-6),
                   replace(self.tx, flow=replace(self.flow, dsts=(2,)))):
            with self.subTest(tx=tx), self.assertRaises(ValueError):
                audit_transmissions((tx,), self.link, 2)
        token = Token(0, 0, 0, 8, (1,), (1.0,))
        with self.assertRaisesRegex(ValueError, "lifetime"):
            summarize((TokenCompletion(token, 4e-6),), (self.tx,), (), self.link, 2)

    def test_compute_resource_overlap_rejected(self):
        token = Token(0, 0, 0, 8, (1,), (1.0,))
        spans = (Interval(0, "expert_compute", "Compute die 1", 0, 2, 1),
                 Interval(0, "expert_compute", "Compute die 1", 1, 3, 1))
        with self.assertRaisesRegex(ValueError, "overbooked"):
            summarize((TokenCompletion(token, 4),), (), spans, self.link, 2)

    def test_percentiles_on_nondegenerate_samples(self):
        completions = tuple(TokenCompletion(Token(i, 0, 0, 8, (1,), (1.0,)), i + 1) for i in range(4))
        metrics, _ = summarize(completions, (), (), self.link, 2)
        self.assertEqual(metrics["layer_makespan_s"], 4)
        self.assertEqual(metrics["throughput_tokens_per_s"], 1)
        self.assertAlmostEqual(metrics["token_latency_p50_s"], 2.5)
        self.assertAlmostEqual(metrics["token_latency_p95_s"], 3.85)
        self.assertAlmostEqual(metrics["token_latency_p99_s"], 3.97)

    def test_zero_token_run_is_not_a_success(self):
        with self.assertRaises(ValueError):
            summarize((), (), (), self.link, 2)


if __name__ == "__main__":
    unittest.main()
