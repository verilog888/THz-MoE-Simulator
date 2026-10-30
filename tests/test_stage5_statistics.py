import unittest

from vdie_moe_sim.statistics import describe, paired_observations
from vdie_moe_sim.sweep_plan import StatisticsConfig


class StatisticsTests(unittest.TestCase):
    def test_hand_mean_sample_standard_deviation(self):
        s = describe([1,2,3], StatisticsConfig(bootstrap_resamples=0))
        self.assertEqual((s['mean'],s['std'],s['min'],s['max']), (2,1,1,3))
        self.assertIsNone(s['ci_low'])

    def test_one_seed_has_no_invented_uncertainty(self):
        s = describe([17], StatisticsConfig())
        self.assertEqual(s['mean'],17)
        self.assertIsNone(s['std'])
        self.assertIsNone(s['ci_low'])
        self.assertEqual(s['ci_method'],'not_estimated_single_seed')

    def test_bootstrap_deterministic_and_analytic_endpoints(self):
        config = StatisticsConfig(seed=7)
        a = describe([0,2],config,key='x')
        self.assertEqual(a,describe([0,2],config,key='x'))
        self.assertEqual((a['ci_low'],a['ci_high']), (0,2))
        self.assertEqual(describe([4,4,4],config)['std'],0)
        self.assertEqual(describe([4,4,4],config)['ci_low'],4)

    def test_bad_statistics_inputs(self):
        for values in ([],[float('nan')],[float('inf')],[[1,2]]):
            with self.assertRaises(ValueError):
                describe(values,StatisticsConfig())
        for kw in ({'confidence':1},{'bootstrap_resamples':1},{'seed':-1}):
            with self.assertRaises(ValueError):
                StatisticsConfig(**kw)

    def rows(self):
        return [dict(point_id='p',routing_seed=seed,series=series,layer_makespan_s=latency,
                     workload_sha256=str(seed),controlled_config_sha256='same')
                for seed, series, latency in [(1,'C',10),(1,'D',1),(2,'C',100),(2,'D',20)]]

    def test_speedup_is_mean_of_matched_ratios(self):
        observations = paired_observations(self.rows(),('C','D'))
        values = [r['speedup_vs_C'] for r in observations if r['series']=='D']
        self.assertEqual(values,[10,5])
        self.assertEqual(describe(values,StatisticsConfig())['mean'],7.5)
        self.assertNotEqual(7.5,110/21)

    def test_incomplete_duplicate_or_unfair_pairs_rejected(self):
        rows=self.rows()
        for broken in (rows[:-1],rows+[rows[0]], [*rows[:-1],{**rows[-1],'workload_sha256':'wrong'}],
                       [*rows[:-1],{**rows[-1],'controlled_config_sha256':'wrong'}]):
            with self.assertRaises(ValueError):
                paired_observations(broken,('C','D'))


if __name__ == '__main__':
    unittest.main()
