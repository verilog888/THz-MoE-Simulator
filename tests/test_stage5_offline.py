from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from vdie_moe_sim.config import load_config
from vdie_moe_sim.offline import audit_run
from vdie_moe_sim.scheduled_layer import run_scheduled,effective_config
from vdie_moe_sim.sweep_plan import load_plan
from vdie_moe_sim.sweep import run_config

ROOT=Path(__file__).resolve().parents[1]


class OfflineTests(unittest.TestCase):
    def result(self):
        return run_scheduled(effective_config(load_config(ROOT/'configs/stage4_concurrent.json'),'D'))

    def test_hand_integrals_from_csv_and_npy(self):
        with tempfile.TemporaryDirectory() as d:
            self.result().export(d)
            a=audit_run(d)
            self.assertEqual(a['status'],'passed')
            self.assertAlmostEqual(a['summary']['pepd'],33.5/20.5)
            self.assertAlmostEqual(a['mean_compute_utilization'],4/20.5)

    def test_all_local_empty_radio_tables(self):
        c=load_config(ROOT/'configs/stage3_tiny.json')
        r=run_scheduled(replace(c,topology=replace(c.topology,num_dies=1)))
        with tempfile.TemporaryDirectory() as d:
            r.export(d)
            self.assertEqual(audit_run(d)['summary']['wireless_payload_bytes'],0)

    def test_csv_summary_corruption_rejected_without_engine(self):
        with tempfile.TemporaryDirectory() as d:
            self.result().export(d)
            path=Path(d)/'summary.csv'
            table=pd.read_csv(path,float_precision='round_trip')
            table.loc[0,'pepd']*=2
            table.to_csv(path,index=False,float_format='%.17g')
            with self.assertRaisesRegex(ValueError,'offline summary pepd'): audit_run(d)

    def test_segment_capacity_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.result().export(d)
            path=Path(d)/'radio_segments.csv'
            table=pd.read_csv(path,float_precision='round_trip')
            table.loc[0,'sent_bits']=1
            table.to_csv(path,index=False,float_format='%.17g')
            with self.assertRaisesRegex(ValueError,'bit conservation'): audit_run(d)

    def test_output_array_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.result().export(d)
            path=Path(d)/'outputs.npy'
            array=np.load(path,allow_pickle=False);array[0]*=2
            np.save(path,array,allow_pickle=False)
            with self.assertRaisesRegex(ValueError,'output row/hash'): audit_run(d)

    def test_priority_ties_survive_independent_floating_sum_order(self):
        plan=load_plan(ROOT/'configs/sweeps/stage5_ablation.json')
        result=run_scheduled(run_config(plan.points[0],23,'Proposed'))
        tied={d['flow_id']:d for d in result.decisions if abs(d['time_s']-80e-6)<1e-15}
        self.assertEqual(tied[30]['priority'],tied[47]['priority'])
        self.assertEqual(len(tied[30]['subcarriers']),2)
        self.assertEqual(len(tied[47]['subcarriers']),1)
        with tempfile.TemporaryDirectory() as d:
            result.export(d)
            self.assertEqual(audit_run(d)['status'],'passed')


if __name__=='__main__': unittest.main()
