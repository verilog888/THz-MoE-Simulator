from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from vdie_moe_sim.config import load_config
from vdie_moe_sim.scheduled_layer import run_scheduled
from vdie_moe_sim.sweep import audit_sweep,run_sweep
from vdie_moe_sim.sweep_plan import SweepSpec,StatisticsConfig,compile_plan

ROOT=Path(__file__).resolve().parents[1]


def contents(root):
    return {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}


class SweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.tmp.name)/'reference'
        cfg=load_config(ROOT/'configs/stage2_tiny.json')
        cls.base=replace(cfg,workload=replace(cfg.workload,routing='uniform',expert_ids=(),gate_weights=()),
                         phy=replace(cfg.phy,num_subcarriers=2))
        cls.spec=SweepSpec('sweep test','base.json',{'workload.num_tokens':[1,2],'phy.num_subcarriers':[2,4]},
                           (11,23),timelines='none',statistics=StatisticsConfig(bootstrap_resamples=200))
        cls.plan=compile_plan(cls.spec,cls.base)
        run_sweep(cls.plan,cls.root)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_complete_matrix_and_offline_aggregate_reconstruction(self):
        result=audit_sweep(self.root)
        self.assertEqual(result['simulation_runs'],30)
        self.assertEqual(result['parameter_points'],3)
        self.assertEqual(result['matched_seed_groups'],6)
        self.assertEqual(result['aggregate_rows'],210)
        raw=pd.read_csv(self.root/'raw_results.csv')
        self.assertFalse(raw.duplicated(['point_id','routing_seed','series']).any())
        self.assertTrue((pd.read_csv(self.root/'aggregate.csv')['n']==2).all())
        for _,group in raw.groupby(['point_id','routing_seed']):
            self.assertEqual(group['workload_sha256'].nunique(),1)
            self.assertEqual(group['total_bandwidth_hz'].nunique(),1)

    def test_different_phy_points_share_route_trace(self):
        raw=pd.read_csv(self.root/'raw_results.csv')
        for seed in self.spec.seeds:
            rows=raw[(raw.routing_seed==seed)&(raw.num_tokens==2)]
            self.assertEqual(rows['routing_sha256'].nunique(),1)

    def test_all_exports_repeat_including_svg(self):
        with tempfile.TemporaryDirectory() as d:
            other=Path(d)/'run'
            run_sweep(self.plan,other)
            self.assertEqual(contents(self.root),contents(other))
            self.assertTrue(list(other.glob('plots/*.svg')))

    def test_resume_reuses_verified_runs_without_simulation(self):
        with tempfile.TemporaryDirectory() as d:
            copy=Path(d)/'run';shutil.copytree(self.root,copy)
            before=contents(copy)
            with patch('vdie_moe_sim.sweep.run_scheduled',side_effect=AssertionError('unexpected rerun')) as mocked:
                run_sweep(self.plan,copy,resume=True)
                mocked.assert_not_called()
            self.assertEqual(before,contents(copy))

    def test_interruption_resume_preserves_completed_runs(self):
        spec=replace(self.spec,axes={'phy.num_subcarriers':(2,)},seeds=(11,))
        plan=compile_plan(spec,self.base)
        calls=[]
        def interrupted(config,**kwargs):
            calls.append(config.scheduling.scheme)
            if len(calls)==3: raise RuntimeError('injected interruption')
            return run_scheduled(config,**kwargs)
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run'
            with patch('vdie_moe_sim.sweep.run_scheduled',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'injected'):run_sweep(plan,out)
            self.assertEqual(json.loads((out/'sweep_manifest.json').read_text())['status'],'failed')
            with self.assertRaisesRegex(ValueError,'incomplete'):audit_sweep(out)
            with patch('vdie_moe_sim.sweep.run_scheduled',wraps=run_scheduled) as actual:
                run_sweep(plan,out,resume=True)
                self.assertEqual(actual.call_count,3)
            self.assertEqual(audit_sweep(out)['simulation_runs'],5)

    def test_resume_changed_plan_or_corrupt_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';shutil.copytree(self.root,out)
            before=(out/'sweep_manifest.json').read_bytes()
            changed=compile_plan(replace(self.spec,seeds=(11,29)),self.base)
            with self.assertRaisesRegex(ValueError,'fingerprint changed'):run_sweep(changed,out,resume=True)
            self.assertEqual(before,(out/'sweep_manifest.json').read_bytes())
            path=next(out.glob('runs/*/seed_*/A/summary.csv'))
            path.write_text(path.read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):run_sweep(self.plan,out,resume=True)

    def test_aggregate_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run';shutil.copytree(self.root,out)
            path=out/'aggregate.csv'
            table=pd.read_csv(path,float_precision='round_trip');table.loc[0,'mean']*=2
            table.to_csv(path,index=False,float_format='%.17g')
            with self.assertRaisesRegex(ValueError,'aggregate differs'):audit_sweep(out)

    def test_nonempty_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/'user-file').write_text('preserve')
            with self.assertRaises(FileExistsError):run_sweep(self.plan,out)
            self.assertEqual((out/'user-file').read_text(),'preserve')

    def test_ablation_grid_runs_three_distinct_phy_series(self):
        base=load_config(ROOT/'configs/stage3_tiny.json')
        spec=SweepSpec('grid','base.json',{'ablation.broadcast':[False,True],'ablation.local_combine':[False,True]},
                       (7,),study='ablation',mode='grid',timelines='none')
        plan=compile_plan(spec,base)
        with tempfile.TemporaryDirectory() as d:
            run_sweep(plan,d)
            self.assertEqual(audit_sweep(d)['simulation_runs'],12)
            rows=pd.read_csv(Path(d)/'raw_results.csv')
            self.assertTrue(rows.scheme.str.startswith('Ablation-').all())
            self.assertEqual(set(rows.series),{'C','D','Proposed'})
            aggregate=pd.read_csv(Path(d)/'aggregate.csv')
            self.assertTrue(aggregate['std'].isna().all())
            self.assertTrue(aggregate['ci_low'].isna().all())

    def test_cli_dry_run_and_saved_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'base.json').write_text(json.dumps(self.base.to_dict()))
            (root/'plan.json').write_text(json.dumps(self.spec.to_dict()))
            out=root/'must-not-exist'
            p=subprocess.run([sys.executable,'-m','vdie_moe_sim','--stage','stage5','--plan',str(root/'plan.json'),
                              '--dry-run','--out',str(out)],capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(json.loads(p.stdout)['simulation_runs'],30)
            self.assertFalse(out.exists())
        p=subprocess.run([sys.executable,'-m','vdie_moe_sim','--stage','stage5','--audit','--out',str(self.root)],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)['status'],'passed')

    def test_cli_execution_and_conflicting_options(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'base.json').write_text(json.dumps(self.base.to_dict()))
            spec=replace(self.spec,axes={'phy.num_subcarriers':(2,)},seeds=(11,))
            (root/'plan.json').write_text(json.dumps(spec.to_dict()))
            args=[sys.executable,'-m','vdie_moe_sim','--stage','stage5','--plan',str(root/'plan.json'),'--out',str(root/'run')]
            p=subprocess.run(args,capture_output=True,text=True)
            self.assertEqual(p.returncode,0,p.stderr)
            self.assertEqual(audit_sweep(root/'run')['simulation_runs'],5)
            p=subprocess.run([*args,'--dry-run','--resume'],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)


if __name__=='__main__':unittest.main()
