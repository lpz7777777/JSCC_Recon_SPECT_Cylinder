"""Regression tests for optional quality selection and original kernel arithmetic."""
import csv
from dataclasses import fields
from pathlib import Path
import subprocess
import sys
import types
import unittest

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT),str(HERE)]
import compton_event_response as current
from detector_csv import load_detector_coordinates


class ResponseMismatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.old=types.ModuleType('baseline_response_for_test')
        sys.modules[cls.old.__name__]=cls.old
        source=subprocess.check_output(['git','show','87d1713:compton_event_response.py'],cwd=ROOT)
        exec(compile(source,'baseline_response.py','exec'),cls.old.__dict__)
        cls.coords=torch.tensor(np.load(HERE/'generated/Geometry/geometry.npz')['coordinates_mm'],dtype=torch.float32)
        cls.detector=torch.tensor(load_detector_coordinates(HERE/'generated/Diagnostics/process_list_audit/Detector.csv',10496))
        rows=[]
        with (HERE/'generated/Diagnostics/process_list_audit/events.csv').open() as stream:
            for r in csv.DictReader(stream):
                if r['variant']=='current' and len(rows)<24:
                    rows.append([float(r['cp1']),float(r['e1']),float(r['cp2']),float(r['e2'])])
        rows.append([7629,.128853,1936,.316147])
        cls.events=torch.tensor(rows)
        cls.settings=current.ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35)
        cls.var=current.build_detector_position_variance(cls.detector,0)
        cls.prepared,_=current.prepare_compton_events(cls.events,cls.settings,cls.detector,cls.var,cls.var,input_energies_already_smeared=True)

    def test_disabled_kernel_is_bitwise_baseline(self):
        settings=self.old.ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35)
        prepared,_=self.old.prepare_compton_events(self.events,settings,self.detector,self.var,self.var,input_energies_already_smeared=True)
        before=self.old.build_compton_cone_weights(prepared,self.coords,settings)
        after=current.build_compton_cone_weights(self.prepared,self.coords,self.settings)
        self.assertTrue(torch.equal(before,after))

    def test_quality_is_independent_of_event_batch_partition(self):
        whole=current.min_standardized_compton_arm(self.prepared,self.coords,self.settings)
        pieces=[]
        for start in range(0,self.prepared.count,7):
            part=current.PreparedComptonEvents(**{f.name:getattr(self.prepared,f.name)[start:start+7] for f in fields(self.prepared)})
            pieces.append(current.min_standardized_compton_arm(part,self.coords,self.settings))
        self.assertTrue(torch.equal(whole,torch.cat(pieces)))
        self.assertGreater(float(whole[-1]),9)

    def test_threshold_inclusive_and_original_invalid_rows_not_new_rejections(self):
        norm=torch.tensor([[.5,.5],[.8,.2],[.3,.7]])
        keep,low,cut=current.select_normalized_response_rows(norm,1,torch.tensor([3.,3.0001,9.]),3.)
        self.assertEqual(keep.tolist(),[True,False,False]); self.assertEqual((low,cut),(0,2))
        keep,low,cut=current.select_normalized_response_rows(norm,1)
        self.assertEqual(keep.tolist(),[True,True,True]); self.assertEqual(cut,0)
        keep,low,cut=current.select_normalized_response_rows(norm,1.9,torch.tensor([3.,9.,9.]),3.)
        self.assertEqual((low,cut),(2,0))

    def test_read_only_checkpoint_keeps_original_joint_mlem_bitwise(self):
        import torch_active_operator as op
        old=types.ModuleType('baseline_operator_for_test')
        source=subprocess.check_output(['git','show','87d1713:experiments/ELLIPSE500x300_H120/torch_active_operator.py'],cwd=ROOT)
        exec(compile(source,'baseline_operator.py','exec'),old.__dict__)
        geometry=op.ActiveGeometry(np.arange(3),np.array([1.,.1,.7]),np.array([[0,1],[1,2],[2,0]]))
        response=op.ViewResponse(torch.tensor([[.2,.1,.5],[.4,.6,.2]]),geometry)
        projection=torch.tensor([[13.,15.],[23.,24.]])
        blocks=[[torch.tensor([[.3,.1,.2],[.2,.4,.1]])], [torch.tensor([[.1,.2,.5]])]]
        sensi=response.sensitivity(); sensid=torch.tensor([[.2],[.1],[.3]])
        before=old.compton_and_joint_mlem(response,projection,blocks,sensi,sensid,50,10)
        snapshots=[]
        def checkpoint(iteration,history_d,history_j):
            snapshots.append((iteration,history_d[-1].clone(),history_j[-1].clone()))
        after=op.compton_and_joint_mlem(response,projection,blocks,sensi,sensid,50,10,checkpoint_callback=checkpoint)
        for a,b in zip(before,after):
            for aa,bb in zip(a,b): self.assertTrue(torch.equal(aa,bb))
        self.assertEqual([r[0] for r in snapshots],[10,20,30,40,50])


if __name__=='__main__': unittest.main()
