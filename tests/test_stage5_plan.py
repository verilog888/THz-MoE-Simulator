from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from vdie_moe_sim.config import load_config
from vdie_moe_sim.scheduled_layer import run_scheduled, effective_config
from vdie_moe_sim.sweep_plan import SweepSpec, compile_plan, load_plan, spec_from_dict

ROOT = Path(__file__).resolve().parents[1]


class PlanTests(unittest.TestCase):
    def setUp(self):
        base=load_config(ROOT/'configs/stage4_limited.json')
        self.base=replace(base, workload=replace(base.workload,num_tokens=2))

    def spec(self, axes, **kw):
        return SweepSpec('test','base.json',axes,(11,23),**kw)

    def test_one_factor_deduplicates_shared_base(self):
        s=self.spec({'workload.num_tokens':[2,4],'phy.num_subcarriers':[2,4]})
        plan=compile_plan(s,self.base)
        self.assertEqual(len(plan.points),3)
        self.assertEqual(len(plan.views),4)
        self.assertEqual(plan.run_count,30)
        self.assertEqual(plan.views[0]['point_id'],plan.views[2]['point_id'])
        self.assertEqual(plan,compile_plan(s,self.base))

    def test_grid_has_cartesian_count_and_preflight_limit(self):
        s=self.spec({'workload.num_tokens':[2,4],'phy.num_subcarriers':[2,4]},mode='grid')
        self.assertEqual(compile_plan(s,self.base).run_count,40)
        with self.assertRaisesRegex(ValueError,'max_runs'):
            compile_plan(replace(s,max_runs=39),self.base)

    def test_integer_and_float_base_coordinate_is_one_point(self):
        base=replace(self.base,phy=replace(self.base.phy,total_bandwidth_hz=64000000))
        plan=compile_plan(self.spec({'phy.total_bandwidth_hz':[64e6],'phy.num_subcarriers':[2]}),base)
        self.assertEqual(len(plan.points),1)
        self.assertEqual(plan.views[0]['point_id'],plan.views[1]['point_id'])

    def test_reject_bad_fields_duplicates_and_seed_axis(self):
        for axes in ({'workload.seed':[1,2]},{'phy.total_bandwidth_hz':[1,1.]},{'phy.num_subcarriers':[]}, {'name':['oops']}):
            with self.assertRaises(ValueError): self.spec(axes)
        with self.assertRaises(ValueError):
            SweepSpec('x','base.json',{'workload.num_tokens':[2]},(1,1))
        with self.assertRaises(ValueError):
            spec_from_dict({'name':'x','base_config':'x','axes':{},'seeds':[1],'unknown':True})

    def test_invalid_effective_configuration_rejected_before_run(self):
        for axes in ({'topology.num_dies':[3]},{'model.top_k':[20]},{'phy.num_subcarriers':[0]}, {'model.activation_precision':['fp32']}):
            with self.assertRaises(ValueError): compile_plan(self.spec(axes),self.base)

    def test_fixed_or_trace_seeds_not_fake_independent_trials(self):
        base=load_config(ROOT/'configs/stage3_tiny.json')
        s=self.spec({'phy.num_subcarriers':[2,4]})
        with self.assertRaisesRegex(ValueError,'pseudo-replication'): compile_plan(s,base)
        self.assertEqual(compile_plan(replace(s,seeds=(1,)),base).run_count,10)

    def test_zipf_axis_requires_zipf_routing(self):
        with self.assertRaisesRegex(ValueError,'Zipf'):
            compile_plan(self.spec({'workload.zipf_skew':[0,1]}),self.base)

    def test_base_path_is_relative_to_plan_and_strict_json(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'plans').mkdir()
            (root/'base.json').write_text(json.dumps(self.base.to_dict()))
            path=root/'plans/plan.json'
            path.write_text(json.dumps({**self.spec({'phy.num_subcarriers':[2]}).to_dict(),'base_config':'../base.json'}))
            plan=load_plan(path)
            self.assertEqual(plan.base,self.base)
            self.assertEqual(plan.trace_base_dir,root)
            path.write_text('{"name":"a","name":"b"}')
            with self.assertRaisesRegex(ValueError,'duplicate'): load_plan(path)

    def test_ablation_axes_cannot_modify_named_baselines(self):
        with self.assertRaisesRegex(ValueError,'ablation'):
            self.spec({'ablation.broadcast':[False,True]})
        plan=compile_plan(self.spec({'ablation.broadcast':[False,True], 'ablation.local_combine':[False,True]},study='ablation',mode='grid'),self.base)
        self.assertEqual(len(plan.points),4)
        self.assertEqual(plan.spec.series,('C','D','Proposed'))

    def test_independent_broadcast_and_local_combine_traffic_and_values(self):
        base=load_config(ROOT/'configs/stage3_tiny.json')
        spec=SweepSpec('ablation','base.json',{'ablation.broadcast':[False,True],'ablation.local_combine':[False,True]},(1,),mode='grid',study='ablation')
        plan=compile_plan(spec,base)
        outputs=[]
        for point in plan.points:
            r=run_scheduled(effective_config(point.config,'C'))
            a=r.config.ablation
            self.assertEqual(r.summary['dispatch_flows'],1 if a.broadcast else 2)
            self.assertEqual(r.summary['return_flows'],1 if a.local_combine else 2)
            self.assertTrue(r.summary['scheme'].startswith('Ablation-'))
            outputs.append(r.outputs[0].vector)
        for output in outputs:
            np.testing.assert_allclose(output,outputs[0],rtol=1e-14,atol=1e-14)


if __name__=='__main__': unittest.main()
