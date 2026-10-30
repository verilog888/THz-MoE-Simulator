from dataclasses import replace
import unittest
import numpy as np

from vdie_moe_sim.combine import PartialVector, sum_vectors, synthetic_expert_output, weighted_vector
from vdie_moe_sim.config import CombineConfig


class CombineSemanticsTests(unittest.TestCase):
    def test_weighted_local_then_source_matches_hand_oracle(self):
        # Die A contains e0/e1; Die B contains e2. Non-normalized weights are deliberate.
        a = sum_vectors([weighted_vector([1, 2, -3], 0.25, 3),
                         weighted_vector([4, -2, 1], 0.5, 3)], 3)
        b = weighted_vector([-1, 3, 2], 2.0, 3)
        result = sum_vectors([a, b], 3)
        np.testing.assert_array_equal(a, [2.25, -0.5, -0.25])
        np.testing.assert_array_equal(result, [0.25, 5.5, 3.75])

    def test_zero_weight_is_not_renormalized(self):
        np.testing.assert_array_equal(weighted_vector([100, -100], 0.0, 2), [0, 0])
        np.testing.assert_array_equal(sum_vectors([np.array([1., 2.]), np.zeros(2)], 2), [1, 2])

    def test_singleton_preserves_already_weighted_vector(self):
        vector = weighted_vector([1, 2], 0.3, 2)
        np.testing.assert_array_equal(sum_vectors([vector], 2), vector)
        self.assertFalse(np.allclose(sum_vectors([vector], 2), vector * 0.3))

    def test_combining_does_not_mutate_inputs(self):
        first, second = np.array([1., 2.]), np.array([3., 4.])
        sum_vectors([first, second], 2)
        weighted_vector(first, 0.4, 2)
        np.testing.assert_array_equal(first, [1, 2])
        np.testing.assert_array_equal(second, [3, 4])

    def test_shape_nonfinite_and_empty_rejected(self):
        for vector in ([1], [[1, 2]], [np.nan, 1], [np.inf, 1]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                weighted_vector(vector, 1, 2)
        with self.assertRaises(ValueError):
            sum_vectors([], 2)
        with self.assertRaises(ValueError):
            weighted_vector([1, 2], -1, 2)

    def test_synthetic_vectors_are_assignment_stable(self):
        a = synthetic_expert_output(7, 9, 32, 42)
        synthetic_expert_output(9, 7, 32, 42)
        np.testing.assert_array_equal(a, synthetic_expert_output(7, 9, 32, 42))
        self.assertFalse(np.array_equal(a, synthetic_expert_output(7, 8, 32, 42)))
        self.assertFalse(np.array_equal(a, synthetic_expert_output(7, 9, 32, 43)))

    def test_partials_own_readonly_storage(self):
        source = np.array([1., 2.])
        partial = PartialVector(0, 0, 0, (0,), 0, source)
        source[:] = 0
        np.testing.assert_array_equal(partial.vector, [1, 2])
        with self.assertRaises(ValueError):
            partial.vector[0] = 10

    def test_bad_combine_configuration(self):
        for change in ({"local_combine": 1}, {"weight_s": -1}, {"local_reduce_s": float("nan")},
                       {"resource_model": "unknown"}, {"verification_seed": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(CombineConfig(), **change)


if __name__ == "__main__":
    unittest.main()
