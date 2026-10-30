from dataclasses import replace
import math
import unittest

import numpy as np

from vdie_moe_sim.compute import estimate_expert
from vdie_moe_sim.config import ComputeConfig, ModelConfig, PHYConfig
from vdie_moe_sim.engine import EventLoop, EventType
from vdie_moe_sim.moe import Flow
from vdie_moe_sim.ofdma import Allocation, validate_allocations
from vdie_moe_sim.placement import ExpertPlacement
from vdie_moe_sim.routing import uniform_route
from vdie_moe_sim.scheduler import IsolatedFullBandAllocator
from vdie_moe_sim.thz_phy import THzLink


class FormulaTests(unittest.TestCase):
    def test_spec_model_sizes(self):
        model = ModelConfig()
        self.assertEqual(model.activation_bytes, 14336)
        self.assertEqual(model.weight_bytes, 44040192)
        self.assertEqual(replace(model, activation_precision="fp8").activation_bytes, 7168)

    def test_compute_memory_bound_hand_oracle(self):
        estimate = estimate_expert(ModelConfig(4, 2, 2, 1), ComputeConfig(48e6, 12e6), 1)
        self.assertEqual((estimate.flops, estimate.weight_bytes, estimate.weight_reads), (48, 24, 1))
        self.assertAlmostEqual(estimate.compute_s, 1e-6, delta=1e-18)
        self.assertAlmostEqual(estimate.memory_s, 2e-6, delta=1e-18)
        self.assertEqual(estimate.duration_s, estimate.memory_s)

    def test_compute_bound_and_reuse(self):
        model = ModelConfig(4, 2, 2, 1)
        compute = ComputeConfig(48e6, 12e6, 4)
        estimate = estimate_expert(model, compute, 9)
        self.assertEqual(estimate.weight_reads, 3)
        self.assertAlmostEqual(estimate.memory_s, 6e-6, delta=1e-18)
        self.assertAlmostEqual(estimate.duration_s, 9e-6, delta=1e-18)
        self.assertEqual(estimate_expert(model, replace(compute, weight_reuse_tokens=None), 9).weight_reads, 1)

    def test_zero_batch_and_reuse_boundaries(self):
        model, compute = ModelConfig(), ComputeConfig(1e12, 1e12, 4)
        for count, reads in ((0, 0), (1, 1), (4, 1), (5, 2), (8, 2)):
            with self.subTest(count=count):
                self.assertEqual(estimate_expert(model, compute, count).weight_reads, reads)
        self.assertEqual(estimate_expert(model, compute, 0).duration_s, 0)
        with self.assertRaises(ValueError):
            estimate_expert(model, compute, -1)

    def test_phy_byte_to_bit_and_setup(self):
        link = THzLink(PHYConfig(16e6, 16, 2, 0.5, 1e-6))
        tx = link.transfer(8, 16)
        self.assertEqual(tx.payload_bits, 64)
        self.assertEqual(tx.rate_bps, 16e6)
        self.assertAlmostEqual(tx.payload_s, 4e-6, delta=1e-18)
        self.assertAlmostEqual(tx.duration_s, 5e-6, delta=1e-18)
        self.assertAlmostEqual(link.transfer(8, 8).payload_s, 8e-6, delta=1e-18)
        self.assertEqual(link.transfer(0, 16).duration_s, 0)

    def test_subcarrier_count_does_not_create_bandwidth(self):
        full_band_times = []
        for count in (16, 32, 64, 128, 256, 512, 1024):
            link = THzLink(PHYConfig(32e9, count, 2, 0.8, 1e-7))
            self.assertAlmostEqual(count * link.subcarrier_rate_bps, 51.2e9)
            full_band_times.append(link.transfer(14336, count).duration_s)
        self.assertEqual(len(set(full_band_times)), 1)

    def test_invalid_parameters(self):
        for args in ((0, 16, 1, 1, 0), (1, 0, 1, 1, 0), (1, 16, 1, 1.01, 0),
                     (1, 16, 1, 1, -1), (math.nan, 16, 1, 1, 0), (math.inf, 16, 1, 1, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                PHYConfig(*args)
        link = THzLink(PHYConfig(1, 16, 1, 1, 0))
        for size, carriers in ((1, 17), (1, 0), (-1, 1), (1.5, 1), (True, 1)):
            with self.subTest(size=size, carriers=carriers), self.assertRaises(ValueError):
                link.transfer(size, carriers)


class EngineTests(unittest.TestCase):
    def test_same_time_fifo_with_new_events(self):
        loop, seen = EventLoop(), []
        loop.schedule(1, EventType.GATE_DONE, 0)
        loop.schedule(1, EventType.GATE_DONE, 1)

        def handler(event):
            seen.append(event.token_id)
            if event.token_id == 0:
                loop.schedule(1, EventType.GATE_DONE, 2)

        loop.run(handler)
        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(loop.processed, 3)

    def test_chronological_order(self):
        loop, seen = EventLoop(), []
        for time in (3, 1, 2):
            loop.schedule(time, EventType.GATE_DONE)
        loop.run(lambda e: seen.append(e.time_s))
        self.assertEqual(seen, [1, 2, 3])

    def test_past_and_nonfinite_rejected(self):
        loop = EventLoop()
        loop.schedule(1, EventType.GATE_DONE)
        loop.run(lambda e: None)
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                loop.schedule(value, EventType.GATE_DONE)

    def test_instantaneous_cycle_stopped(self):
        loop = EventLoop()
        loop.schedule(0, EventType.GATE_DONE)
        with self.assertRaisesRegex(RuntimeError, "event limit"):
            loop.run(lambda e: loop.schedule(e.time_s, e.kind), max_events=5)
        self.assertEqual(loop.processed, 5)


class AllocationTests(unittest.TestCase):
    def test_valid_disjoint_groups(self):
        validate_allocations((Allocation(0, (0, 1)), Allocation(1, (2, 3))), 4)
        validate_allocations((), 4)

    def test_invalid_allocations(self):
        invalid = ((Allocation(0, (0, 1)), Allocation(1, (1, 2))),
                   (Allocation(0, (4,)),), (Allocation(0, (0, 0)),),
                   (Allocation(0, ()),), (Allocation(0, (-1,)),),
                   (Allocation(0, (0,)), Allocation(0, (1,))))
        for allocations in invalid:
            with self.subTest(allocations=allocations), self.assertRaises(ValueError):
                validate_allocations(allocations, 4)

    def test_isolated_allocator_rejects_competition(self):
        allocator = IsolatedFullBandAllocator(16)
        flow = Flow(0, 0, 0, (1,), 8, "dispatch")
        self.assertEqual(allocator.allocate((flow,))[0].subcarriers, tuple(range(16)))
        with self.assertRaises(ValueError):
            allocator.allocate((flow, replace(flow, flow_id=1)))


class RoutingPlacementTests(unittest.TestCase):
    def test_placement_and_grouping(self):
        placement = ExpertPlacement.contiguous(256, 8)
        self.assertEqual(placement.die_for(0), 0)
        self.assertEqual(placement.die_for(31), 0)
        self.assertEqual(placement.die_for(32), 1)
        self.assertEqual(placement.die_for(255), 7)
        self.assertEqual(placement.group((2, 5, 66, 67)), {0: (2, 5), 2: (66, 67)})
        for expert in (-1, 256):
            with self.assertRaises(ValueError):
                placement.die_for(expert)

    def test_invalid_placement(self):
        for experts, dies in ((256, 3), (0, 8), (1, 2), (8, 0)):
            with self.subTest(experts=experts, dies=dies), self.assertRaises(ValueError):
                ExpertPlacement.contiguous(experts, dies)

    def test_uniform_is_seeded_unique_and_equal_weighted(self):
        a = uniform_route(256, 8, np.random.default_rng(42))
        b = uniform_route(256, 8, np.random.default_rng(42))
        self.assertEqual(a, b)
        self.assertEqual(len(set(a[0])), 8)
        self.assertEqual(sum(a[1]), 1)
        self.assertNotEqual(a, uniform_route(256, 8, np.random.default_rng(43)))


if __name__ == "__main__":
    unittest.main()
