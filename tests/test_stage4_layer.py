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

from vdie_moe_sim.config import load_config, SchedulingConfig
from vdie_moe_sim.experiments import audit_comparison, run_comparison
from vdie_moe_sim.layer import run_layer
from vdie_moe_sim.scheduled_layer import effective_config, run_scheduled, SCHEMES

ROOT = Path(__file__).resolve().parents[1]


class ScheduledTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/stage3_tiny.json")
        self.concurrent = load_config(ROOT / "configs/stage2_tiny.json")

    def run_scheme(self, scheme, config=None):
        return run_scheduled(effective_config(config or self.config, scheme), trace_base_dir=ROOT / "configs")

    def test_five_hand_calculated_latencies_and_bytes(self):
        # A serializes eight 5-us hops (compute overlaps), then Source/Base 3 us.
        # B serializes four 5-us hops, then Source/Base 3 us.
        # C/D/Proposed singleton broadcast: D 0..5, compute 5..9, reduce 9..10,
        # Return 10..15, Source/Base 15..18. One ready flow gives no OFDMA gain.
        expected = {"A": (43, 64), "B": (23, 32), "C": (18, 16), "D": (18, 16), "Proposed": (18, 16)}
        results = []
        for scheme, (microseconds, payload) in expected.items():
            r = self.run_scheme(scheme)
            self.assertAlmostEqual(r.summary["layer_makespan_s"], microseconds*1e-6, delta=1e-18)
            self.assertEqual(r.summary["wireless_payload_bytes"], payload)
            results.append(r)
        self.assertTrue(audit_comparison(results)["numerically_equal"])

    def test_tdma_c_matches_stage3_full_pipeline(self):
        for name in ("stage3_tiny", "stage3_return_feedback", "stage2_batching"):
            cfg = load_config(ROOT / f"configs/{name}.json")
            before, after = run_layer(cfg), self.run_scheme("C", cfg)
            np.testing.assert_allclose([c.completed_s for c in before.completions], [c.completed_s for c in after.completions], atol=1e-18, rtol=0)
            self.assertEqual(before.summary["wireless_payload_bytes"], after.summary["wireless_payload_bytes"])

    def test_static_concurrency_hand_integrals_and_possible_slowdown(self):
        r = self.run_scheme("D", self.concurrent)
        # Dispatch both 0..9; Returns 11..19 and 13..20.5 us.
        np.testing.assert_allclose([(f.admitted_s, f.completed_s) for f in r.flows],
                                  [(0,9e-6),(0,9e-6),(11e-6,19e-6),(13e-6,20.5e-6)], rtol=0, atol=1e-18)
        s = r.summary
        self.assertAlmostEqual(s["layer_makespan_s"], 20.5e-6, delta=1e-18)
        self.assertAlmostEqual(s["pepd"], 33.5/20.5)
        self.assertAlmostEqual(s["occupied_subcarrier_seconds"], 16*18.5e-6, delta=1e-18)
        self.assertAlmostEqual(s["thz_utilization"], 18.5/20.5)
        self.assertAlmostEqual(s["allocated_capacity_bits"], 296)
        self.assertAlmostEqual(s["bandwidth_waste"], 1-256/296)
        self.assertGreater(s["layer_makespan_s"], self.run_scheme("C", self.concurrent).summary["layer_makespan_s"])
        self.assertEqual(s["peak_concurrent_flows"], 2)  # Both flows share the same source AND destination.

    def test_ofdma_pepd_not_always_greater_than_one(self):
        cfg = replace(self.concurrent, compute=replace(self.concurrent.compute, compute_flops_per_s=48000))
        r = self.run_scheme("D", cfg)
        self.assertEqual(r.summary["peak_concurrent_flows"], 2)
        self.assertLess(r.summary["pepd"], 1)

    def test_p2p_is_per_expert_and_relay_charges_every_hop(self):
        a, b, c = [self.run_scheme(s) for s in ("A", "B", "C")]
        self.assertEqual([len(r.flows) for r in (a,b,c)], [8,4,2])
        self.assertEqual([r.summary["return_payload_bytes"] for r in (a,b,c)], [32,16,8])
        self.assertEqual(a.summary["relay_extra_hops"], 4)
        for r in a.flows:
            if r.parent_id is not None:
                self.assertEqual(r.ready_s, a.flows[r.parent_id].completed_s)
                self.assertEqual(r.flow.src, a.config.topology.num_dies)

    def test_broadcast_multidestination_dedup_and_slow_fanin(self):
        cfg = replace(self.config, model=replace(self.config.model, num_experts=6, top_k=4),
                      topology=replace(self.config.topology, num_dies=3),
                      workload=replace(self.config.workload, expert_ids=(0,2,3,4), gate_weights=()))
        for scheme in ("C", "D", "Proposed"):
            r = self.run_scheme(scheme, cfg)
            dispatch = [f for f in r.flows if f.flow.flow_type == "dispatch"]
            self.assertEqual(len(dispatch), 1)
            self.assertEqual(dispatch[0].flow.dsts, (1,2))
            self.assertEqual(r.summary["return_flows"], 2)
            source = next(j for j in r.jobs if j.kind == "source_reduce")
            self.assertEqual(source.ready_s, max(d.available_s for d in r.deliveries))

    def test_all_local_empty_rf_exports(self):
        cfg = replace(self.config, topology=replace(self.config.topology, num_dies=1))
        for scheme in SCHEMES:
            r = self.run_scheme(scheme, cfg)
            self.assertEqual(r.flows, ())
            self.assertEqual(r.summary["thz_utilization"], 0)
            with tempfile.TemporaryDirectory() as d:
                r.export(d)
                self.assertTrue(pd.read_csv(Path(d)/"flows.csv").empty)
                self.assertTrue(pd.read_csv(Path(d)/"radio_segments.csv").empty)

    def test_more_flows_than_subcarriers_drain_without_loss(self):
        cfg = replace(self.concurrent, workload=replace(self.concurrent.workload, num_tokens=7),
                      phy=replace(self.concurrent.phy, num_subcarriers=2))
        for scheme in ("D", "Proposed"):
            r = self.run_scheme(scheme, cfg)
            self.assertEqual(len(r.completions), 7)
            self.assertEqual(len(r.inputs), 14)
            self.assertEqual(len(r.flows), 14)
            self.assertLessEqual(r.summary["peak_concurrent_flows"], 2)
            self.assertTrue(any(f.admitted_s > f.ready_s for f in r.flows))

    def test_one_subcarrier_c_d_ca_agree(self):
        cfg = replace(self.concurrent, phy=replace(self.concurrent.phy, num_subcarriers=1))
        results = [self.run_scheme(s, cfg) for s in ("C", "D", "Proposed")]
        for r in results:
            self.assertAlmostEqual(r.summary["layer_makespan_s"], 20e-6, delta=1e-18)
            self.assertLessEqual(r.summary["pepd"], 1)

    def test_ca_features_include_return_age_and_compute_idle(self):
        cfg = replace(self.concurrent, scheduling=SchedulingConfig(epoch_s=0.5e-6, q_ref_bytes=8, w_ref_s=1e-6))
        r = self.run_scheme("Proposed", cfg)
        self.assertTrue(any(d["idle"] == 1 for d in r.decisions))
        self.assertTrue(any(d["critical"] == 1 for d in r.decisions))
        self.assertTrue(any(d["w"] > 1 for d in r.decisions))
        self.assertTrue(r.ignored_radio)
        self.assertEqual(len([e for e in r.events if e.kind.value == "RADIO_FLOW_DONE"]), len(r.flows))

    def test_fractional_bits_across_many_epochs(self):
        cfg = replace(self.concurrent, scheduling=SchedulingConfig(epoch_s=0.137e-6))
        r = self.run_scheme("Proposed", cfg)
        self.assertTrue(any(not s.sent_bits.is_integer() for s in r.segments if s.phase == "payload"))
        for f in r.flows:
            self.assertAlmostEqual(sum(s.sent_bits for s in r.segments if s.flow_id == f.flow.flow_id), 64, delta=1e-7)

    def test_arrival_offset_changes_no_latency(self):
        a = self.run_scheme("D", self.concurrent)
        b = self.run_scheme("D", replace(self.concurrent, workload=replace(self.concurrent.workload, arrival_s=.01)))
        self.assertAlmostEqual(a.summary["layer_makespan_s"], b.summary["layer_makespan_s"], delta=1e-15)

    def test_false_numeric_output_is_rejected(self):
        r = self.run_scheme("Proposed", self.concurrent)
        with self.assertRaises(ValueError):
            replace(r, outputs=(replace(r.outputs[0], vector=r.outputs[0].vector*2), *r.outputs[1:])).audit()

    def test_spectrum_progress_and_priority_corruptions_are_rejected(self):
        r = self.run_scheme("D", self.concurrent)
        first = r.segments[0]
        other = next(s for s in r.segments if s.flow_id != first.flow_id and s.start_s == first.start_s)
        corruptions = [replace(r, segments=(replace(first, sent_bits=1), *r.segments[1:])),
            replace(r, segments=(replace(first, subcarriers=other.subcarriers), *r.segments[1:])),
            replace(r, decisions=({**r.decisions[0], "priority": -1}, *r.decisions[1:])),
            replace(r, flows=(replace(r.flows[0], admitted_s=1e-6), *r.flows[1:]))]
        for index, broken in enumerate(corruptions):
            with self.subTest(corruption=index), self.assertRaises(ValueError):
                broken.audit()

    def test_relay_and_combine_dependency_corruptions_are_rejected(self):
        r = self.run_scheme("A")
        i = next(i for i,f in enumerate(r.flows) if f.parent_id is not None)
        bad_flows = list(r.flows)
        bad_flows[i] = replace(bad_flows[i], parent_id=None)
        source = next(j for j in r.jobs if j.kind == "source_reduce")
        bad_jobs = tuple(replace(j, ready_s=0) if j == source else j for j in r.jobs)
        for bad in (replace(r, flows=tuple(bad_flows)), replace(r, jobs=bad_jobs), replace(r, events=r.events[:-1])):
            with self.assertRaises(ValueError):
                bad.audit()

    def test_fairness_audit_rejects_extra_bandwidth_or_changed_workload(self):
        results = [self.run_scheme(s) for s in SCHEMES]
        for bad in (replace(results[-1], config=replace(results[-1].config, phy=replace(results[-1].config.phy, total_bandwidth_hz=32e6))),
                    replace(results[-1], workload=replace(results[-1].workload, tokens=(replace(results[-1].workload.tokens[0], arrival_s=1),)))):
            with self.assertRaises(ValueError):
                audit_comparison([*results[:-1], bad])

    def test_trace_replay_preserves_non_normalized_weights(self):
        with tempfile.TemporaryDirectory() as d:
            trace = {"schema_version":1,"tokens":[{"token_id":19,"source_die":0,"arrival_s":0,"expert_ids":[0,2,3],"gate_weights":[2,0,4]}]}
            path = Path(d)/"trace.json"
            path.write_text(json.dumps(trace))
            cfg = replace(self.config, workload=replace(self.config.workload, routing="trace", source_policy="trace", expert_ids=(), gate_weights=(), trace_file=str(path)))
            results = [self.run_scheme(s, cfg) for s in SCHEMES]
            self.assertTrue(audit_comparison(results)["numerically_equal"])
            self.assertEqual(results[-1].workload.tokens[0].gate_weights, (2,0,4))

    def test_default_paper_size_five_schemes(self):
        cfg = load_config(ROOT / "configs/stage3_zipf.json")
        results = [self.run_scheme(s, cfg) for s in SCHEMES]
        self.assertTrue(audit_comparison(results)["same_total_phy_budget"])
        for r in results:
            self.assertEqual(len(r.inputs), 1024)
            self.assertEqual(np.stack([o.vector for o in r.outputs]).shape, (128,7168))
        self.assertGreater(results[-1].summary["peak_concurrent_flows"], 1)

    def test_export_repeated_bytes_and_trace_replay(self):
        r = self.run_scheme("Proposed", self.concurrent)
        with tempfile.TemporaryDirectory() as d:
            a,b = Path(d)/"a", Path(d)/"b"
            r.export(a)
            self.run_scheme("Proposed", self.concurrent).export(b)
            for f in a.iterdir():
                self.assertEqual(f.read_bytes(), (b/f.name).read_bytes(), f.name)
            cfg = replace(r.config, workload=replace(r.config.workload, routing="trace", source_policy="trace", expert_ids=(), gate_weights=(), trace_file=str(a/"routing_trace.json"), seed=999))
            replay = run_scheduled(cfg)
            self.assertEqual(r.flows, replay.flows)
            self.assertEqual(r.segments, replay.segments)
            np.testing.assert_array_equal(np.load(a/"outputs.npy"), np.stack([o.vector for o in replay.outputs]))

    def test_cli_comparison_and_plots(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)/"comparison"
            process = subprocess.run([sys.executable, "-m", "vdie_moe_sim", "--stage", "stage4", "--compare", "--config",
                                      str(ROOT/"configs/stage2_tiny.json"), "--out", str(out), "--timeline"], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(list(pd.read_csv(out/"comparison.csv")["scheme"]), list(SCHEMES))
            self.assertTrue((out/"comparison.png").read_bytes().startswith(b"\x89PNG"))
            for scheme in SCHEMES:
                self.assertTrue((out/scheme/"timeline.png").read_bytes().startswith(b"\x89PNG"))
            self.assertTrue(json.loads((out/"fairness.json").read_text())["same_total_phy_budget"])

    def test_invalid_cli_stage_or_conflicting_options(self):
        for args in (("--stage","stage3","--compare"),("--stage","stage4","--compare","--scheme","A")):
            p = subprocess.run([sys.executable,"-m","vdie_moe_sim",*args,"--config",str(ROOT/"configs/stage3_tiny.json"),"--out","unused"], capture_output=True)
            self.assertNotEqual(p.returncode,0)


if __name__ == "__main__":
    unittest.main()
