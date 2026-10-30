from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vdie_moe_sim.batching import ExpertBatcher, ExpertInput
from vdie_moe_sim.config import BatchingConfig, load_config
from vdie_moe_sim.dispatch import plan_dispatch
from vdie_moe_sim.moe import Token
from vdie_moe_sim.placement import ExpertPlacement
from vdie_moe_sim.routing import zipf_probabilities, zipf_route
from vdie_moe_sim.workload import build_workload

ROOT = Path(__file__).resolve().parents[1]


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/stage1_tiny.json")

    def test_fixed_multitoken_round_robin_and_arrivals(self):
        config = replace(self.config, workload=replace(self.config.workload, num_tokens=5,
                          source_policy="round_robin", source_die=1, arrival_s=2, interarrival_s=0.5))
        workload = build_workload(config)
        self.assertEqual([t.source_die for t in workload.tokens], [1, 0, 1, 0, 1])
        self.assertEqual([t.arrival_s for t in workload.tokens], [2, 2.5, 3, 3.5, 4])
        self.assertTrue(all(t.expert_ids == (1,) for t in workload.tokens))

    def test_uniform_seeded_no_replacement(self):
        model = replace(self.config.model, num_experts=32, top_k=8)
        w = replace(self.config.workload, num_tokens=64, routing="uniform", expert_ids=(), gate_weights=())
        config = replace(self.config, model=model, workload=w)
        a, b = build_workload(config), build_workload(config)
        self.assertEqual(a, b)
        self.assertTrue(all(len(set(t.expert_ids)) == 8 for t in a.tokens))
        self.assertNotEqual(a.digest, build_workload(replace(config, workload=replace(w, seed=43))).digest)

    def test_zipf_formula_and_zero_skew(self):
        np.testing.assert_allclose(zipf_probabilities(3, 1), [6 / 11, 3 / 11, 2 / 11])
        np.testing.assert_array_equal(zipf_probabilities(4, 0), np.full(4, 0.25))

    def test_zipf_is_seeded_unique_and_skewed(self):
        model = replace(self.config.model, num_experts=32, top_k=4)
        w = replace(self.config.workload, num_tokens=512, routing="zipf", zipf_skew=2,
                    expert_ids=(), gate_weights=())
        config = replace(self.config, model=model, workload=w)
        a = build_workload(config)
        self.assertEqual(a, build_workload(config))
        counts = np.bincount([e for t in a.tokens for e in t.expert_ids], minlength=32)
        self.assertGreater(counts[0], 5 * counts[-1])
        self.assertTrue(all(len(set(t.expert_ids)) == 4 for t in a.tokens))
        self.assertTrue(all(t.gate_weights == (0.25,) * 4 for t in a.tokens))

    def test_invalid_zipf(self):
        for skew in (-1, float("nan"), float("inf")):
            with self.subTest(skew=skew), self.assertRaises(ValueError):
                zipf_probabilities(4, skew)
        with self.assertRaises(ValueError):
            zipf_route(4, 5, 1, np.random.default_rng(1))

    def trace_config(self, rows, directory):
        path = Path(directory) / "trace.json"
        path.write_text(json.dumps({"schema_version": 1, "tokens": rows}))
        workload = replace(self.config.workload, num_tokens=len(rows), routing="trace", source_policy="trace",
                           trace_file=path.name, expert_ids=(), gate_weights=())
        return replace(self.config, workload=workload)

    def test_trace_preserves_ids_weights_and_sorts_arrivals(self):
        rows = [{"token_id": 9, "source_die": 1, "arrival_s": 1, "expert_ids": [0], "gate_weights": [2.5]},
                {"token_id": 3, "source_die": 0, "arrival_s": 0, "expert_ids": [1], "gate_weights": [0.3]}]
        with tempfile.TemporaryDirectory() as temp:
            config = self.trace_config(rows, temp)
            a = build_workload(config, base_dir=temp)
            self.assertEqual([t.token_id for t in a.tokens], [3, 9])
            self.assertEqual([t.gate_weights for t in a.tokens], [(0.3,), (2.5,)])
            self.assertEqual(len(a.trace_sha256), 64)
            path = Path(temp) / "trace.json"
            path.write_text(json.dumps(a.to_trace()))
            self.assertEqual(a.digest, build_workload(config, base_dir=temp).digest)
            # A byte-level input hash changes with formatting; semantic workload digest does not.
            self.assertNotEqual(a.trace_sha256, build_workload(config, base_dir=temp).trace_sha256)

    def test_trace_rejects_malformed_or_duplicate_assignments(self):
        row = {"token_id": 0, "source_die": 0, "arrival_s": 0, "expert_ids": [1], "gate_weights": [1]}
        changes = [{"expert_ids": [2]}, {"gate_weights": []}, {"gate_weights": [-1]},
                   {"arrival_s": -1}, {"source_die": 2}, {"token_id": True}, {"unknown": 1}]
        with tempfile.TemporaryDirectory() as temp:
            for change in changes:
                with self.subTest(change=change), self.assertRaises(ValueError):
                    build_workload(self.trace_config([{**row, **change}], temp), base_dir=temp)
            with self.assertRaisesRegex(ValueError, "duplicate token"):
                build_workload(self.trace_config([row, row], temp), base_dir=temp)
            config = self.trace_config([row], temp)
            with self.assertRaisesRegex(ValueError, "token count"):
                build_workload(replace(config, workload=replace(config.workload, num_tokens=2)), base_dir=temp)
            with self.assertRaisesRegex(ValueError, "duplicate expert"):
                two = {**row, "expert_ids": [1, 1], "gate_weights": [0.2, 0.8]}
                cfg = self.trace_config([two], temp)
                build_workload(replace(cfg, model=replace(cfg.model, top_k=2)), base_dir=temp)

    def test_conflicting_config_fields_rejected(self):
        for changes in ({"routing": "trace"}, {"trace_file": "trace.json"},
                        {"source_policy": "invalid"}, {"zipf_skew": 1},
                        {"routing": "uniform"}, {"interarrival_s": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.config.workload, **changes)


class DispatchTests(unittest.TestCase):
    def test_one_broadcast_and_local_group(self):
        token = Token(0, 0, 0, 14336, (0, 1, 2, 3, 4, 5), (1 / 6,) * 6)
        plan = plan_dispatch(token, ExpertPlacement.contiguous(8, 4))
        self.assertEqual(plan.local_experts, (0, 1))
        self.assertEqual(plan.remote_dies, (1, 2))
        self.assertEqual(plan.remote_groups, ((1, (2, 3)), (2, (4, 5))))

    def test_all_local_needs_no_transmission(self):
        token = Token(7, 1, 0, 8, (2, 3), (0.2, 0.8))
        plan = plan_dispatch(token, ExpertPlacement.contiguous(4, 2))
        self.assertEqual(plan.local_experts, (2, 3))
        self.assertEqual(plan.remote_dies, ())


class BatchingTests(unittest.TestCase):
    def request(self, request_id, time, expert=0, die=0):
        return ExpertInput(request_id, request_id, expert, die, time, True)

    def test_zero_window_does_not_merge_same_time_arrivals(self):
        batcher = ExpertBatcher(BatchingConfig(0, 16))
        first, timers = batcher.admit(self.request(0, 0))
        second, _ = batcher.admit(self.request(1, 0))
        self.assertEqual(timers, ())
        self.assertEqual(first[0].request_ids, (0,))
        self.assertEqual(second[0].request_ids, (1,))
        self.assertEqual(batcher.pending, 0)

    def test_window_closes_with_weight_reuse_membership(self):
        batcher = ExpertBatcher(BatchingConfig(2, 4))
        first, deadlines = batcher.admit(self.request(0, 0))
        second, _ = batcher.admit(self.request(1, 1))
        self.assertEqual((first, second), ((), ()))
        self.assertEqual(deadlines, ((2, 0),))
        closed = batcher.expire(0, 2)
        self.assertEqual(closed.request_ids, (0, 1))
        self.assertEqual(closed.ready_s, 2)
        self.assertEqual(closed.close_reason, "window")

    def test_capacity_closes_early_and_timer_is_stale(self):
        batcher = ExpertBatcher(BatchingConfig(10, 2))
        batcher.admit(self.request(0, 0))
        closed, _ = batcher.admit(self.request(1, 1))
        self.assertEqual(closed[0].request_ids, (0, 1))
        self.assertEqual(closed[0].ready_s, 1)
        self.assertEqual(closed[0].close_reason, "cap")
        self.assertIsNone(batcher.expire(0, 10))
        self.assertEqual(batcher.pending, 0)

    def test_deadline_tie_is_half_open_independent_of_event_order(self):
        results = []
        for expire_first in (True, False):
            batcher = ExpertBatcher(BatchingConfig(1, 8))
            batcher.admit(self.request(0, 0))
            closed = []
            if expire_first:
                closed.append(batcher.expire(0, 1))
            newly_closed, timers = batcher.admit(self.request(1, 1))
            closed.extend(newly_closed)
            if not expire_first:
                self.assertIsNone(batcher.expire(0, 1))
            closed.append(batcher.expire(timers[0][1], 2))
            results.append([(b.request_ids, b.ready_s) for b in closed])
        self.assertEqual(results, [[((0,), 1), ((1,), 2)]] * 2)

    def test_experts_have_separate_windows(self):
        batcher = ExpertBatcher(BatchingConfig(1, 8))
        batcher.admit(self.request(0, 0, expert=0))
        batcher.admit(self.request(1, 0, expert=1))
        self.assertEqual(batcher.expire(0, 1).request_ids, (0,))
        self.assertEqual(batcher.expire(1, 1).request_ids, (1,))

    def test_early_deadline_and_duplicate_request_rejected(self):
        batcher = ExpertBatcher(BatchingConfig(1, 8))
        batcher.admit(self.request(0, 0))
        with self.assertRaises(ValueError):
            batcher.expire(0, 0.5)
        with self.assertRaises(ValueError):
            batcher.admit(self.request(0, 0))

    def test_bad_batch_config_rejected(self):
        for window, cap in ((-1, 8), (0, 0), (float("inf"), 8), (0, True)):
            with self.subTest(window=window, cap=cap), self.assertRaises(ValueError):
                BatchingConfig(window, cap)


if __name__ == "__main__":
    unittest.main()
