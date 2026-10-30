import math
import unittest

from vdie_moe_sim.config import PHYConfig, SchedulingConfig
from vdie_moe_sim.engine import EventLoop, PipelineEventType as E, RadioEventType as R
from vdie_moe_sim.radio import SharedRadio, allocate_counts


class AllocationTests(unittest.TestCase):
    def test_tdma_and_static_paper_floor(self):
        self.assertEqual(allocate_counts("tdma", 8, [0]), ((8,), False))
        self.assertEqual(allocate_counts("static", 8, [3, 2, 1]), ((2, 2, 2), False))

    def test_ca_paper_formula_and_overflow_repair(self):
        self.assertEqual(allocate_counts("ca", 8, [1, 2, 1]), ((2, 4, 2), False))
        self.assertEqual(allocate_counts("ca", 4, [100, 0, 0, 0]), ((1, 1, 1, 1), True))
        self.assertEqual(allocate_counts("ca", 8, [100, 0, 0]), ((6, 1, 1), True))

    def test_zero_weights_ties_and_huge_weights(self):
        self.assertEqual(allocate_counts("ca", 8, [0, 0, 0]), ((3, 3, 2), False))
        self.assertEqual(allocate_counts("ca", 4, [1e308, 1e308]), ((2, 2), False))

    def test_allocation_budget_corners(self):
        for total in (1, 2, 3, 8, 16, 64):
            for count in range(1, total + 1):
                for weights in ([0]*count, [1]*count, [1e9]+[1]*(count-1)):
                    values, _ = allocate_counts("ca", total, weights)
                    self.assertEqual(sum(values), total)
                    self.assertGreaterEqual(min(values), 1)

    def test_bad_allocation_and_configuration(self):
        for policy, total, weights in [("x", 2, [1]), ("static", 1, [1, 1]), ("tdma", 2, [1, 1]), ("ca", 2, [math.nan])]:
            with self.assertRaises(ValueError):
                allocate_counts(policy, total, weights)
        for kwargs in ({"epoch_s": 0}, {"q_ref_bytes": 0}, {"beta": -1}, {"scheme": "wrong"}):
            with self.assertRaises(ValueError):
                SchedulingConfig(**kwargs)


class RadioTests(unittest.TestCase):
    def run_radio(self, scheme, arrivals, setup=0, subcarriers=4, epoch=0.5, **weights):
        loop = EventLoop()
        completed = []
        radio = SharedRadio(loop, PHYConfig(8, subcarriers, 1, 1, setup),
                            SchedulingConfig(scheme=scheme, epoch_s=epoch, q_ref_bytes=1, w_ref_s=1, **weights),
                            lambda s: completed.append((s.flow.flow_id, loop.now_s)))
        for i, (at, payload) in enumerate(arrivals):
            loop.schedule(at, E.TOKEN_ARRIVAL, i)
        def handle(event):
            if event.kind == E.TOKEN_ARRIVAL:
                radio.enqueue(event.token_id, 0, (1,), arrivals[event.token_id][1], "dispatch")
            else:
                radio.handle(event)
        loop.run(handle)
        self.assertFalse(radio.unfinished)
        self.assertEqual(len(completed), len(arrivals))
        for state in radio.states.values():
            pieces = [p for p in radio.segments if p.flow_id == state.flow.flow_id]
            self.assertAlmostEqual(sum(p.sent_bits for p in pieces), state.flow.payload_bytes * 8)
            self.assertAlmostEqual(sum(p.end_s-p.start_s for p in pieces if p.phase == "setup"), setup)
        return radio, dict(completed)

    def test_tdma_hand_calculation(self):
        _, done = self.run_radio("C", [(0, 1), (0, 1)], setup=1)
        self.assertEqual(done, {0: 2, 1: 4})

    def test_same_time_arrivals_share_static_spectrum(self):
        radio, done = self.run_radio("D", [(0, 1), (0, 1)], setup=1)
        self.assertEqual(done, {0: 3, 1: 3})
        self.assertEqual([len(d["subcarriers"]) for d in radio.decisions[:2]], [2, 2])

    def test_late_arrival_resizes_rate_without_resending(self):
        radio, done = self.run_radio("D", [(0, 2), (1, 1)])
        self.assertEqual(done, {0: 3, 1: 3})
        self.assertTrue(radio.ignored)

    def test_setup_survives_reallocation(self):
        _, done = self.run_radio("D", [(0, 1), (0.5, 1)], setup=1)
        self.assertEqual(done, {0: 3, 1: 3.25})

    def test_fifo_admission_more_flows_than_subcarriers(self):
        radio, done = self.run_radio("D", [(0, 1)] * 5, subcarriers=2)
        self.assertEqual(done, {0: 2, 1: 2, 2: 4, 3: 4, 4: 5})
        self.assertEqual([s.admitted_s for s in radio.states.values()], [0, 0, 2, 2, 4])

    def test_ca_epoch_progress_zero_priority_fallback(self):
        radio, done = self.run_radio("Proposed", [(0, 1), (0, 1)], epoch=0.3, alpha=0, beta=0, gamma=0, delta=0)
        self.assertAlmostEqual(done[0], 2)
        self.assertAlmostEqual(done[1], 2)
        self.assertGreater(len(radio.decisions), 4)

    def test_ca_positive_service_and_finite_queue_drain(self):
        radio, _ = self.run_radio("Proposed", [(0, 20)] + [(0.1*i, 1) for i in range(1, 8)], epoch=0.2)
        self.assertTrue(all(len(d["subcarriers"]) >= 1 for d in radio.decisions))

    def test_same_time_arbitration_after_zero_time_chain(self):
        loop, seen = EventLoop(), []
        def handle(e):
            seen.append(e.kind)
            if e.kind == E.TOKEN_ARRIVAL:
                loop.schedule(0, E.GATE_DONE)
        loop.schedule(0, R.REBALANCE, priority=1)
        loop.schedule(0, E.TOKEN_ARRIVAL)
        loop.run(handle)
        self.assertEqual(seen, [E.TOKEN_ARRIVAL, E.GATE_DONE, R.REBALANCE])


if __name__ == "__main__":
    unittest.main()
