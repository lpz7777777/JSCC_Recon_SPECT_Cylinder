"""Bounded 5e9 contracts and original-core checkpoint invariance, not imaging proof."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from energy_5e9_v5_contract import load_contract,write_checkpoint,verify_checkpoints,CHANNELS
from verify_energy_5e9_v5 import validate_authority,verify_support_record
from run_reconstruction import digest
from torch_active_operator import ActiveGeometry,ViewResponse,compton_and_joint_mlem

class FiveBillionFormalSafeguards(unittest.TestCase):
    def geometry(self):
        return ActiveGeometry([0,2],[1.,0.,1.],np.arange(3)[:,None])

    def test_bounded_entry_rejects_10000_before_reading_data(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'contract.json';p.write_text('{}')
            for mode,n,save in [('formal',10000,50),('formal',2000,10),('validation',50,50)]:
                with self.assertRaises(ValueError):load_contract(p,mode,n,save)

    def test_formal_requires_pinned_actual_5e9_validation(self):
        with self.assertRaises(ValueError):validate_authority(None,None,Path('contract.json'))

    def test_active_support_guard_retains_small_positive_rows(self):
        record=dict(minimum_active_event_mass=1e-20,initial_forward_floor_events=3,accepted_events=10)
        verify_support_record(record)
        for minimum in (0.,-1.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):verify_support_record(dict(record,minimum_active_event_mass=minimum))
        for count in (-1,11,True,1.5):
            with self.assertRaises(ValueError):verify_support_record(dict(record,initial_forward_floor_events=count))

    def test_actual_frozen_calibration_must_pass_at_deployment(self):
        # A read-only contract test is not evidence of a new numerical run.
        p=Path(__file__).parent/'contract.json'
        if not p.exists():self.skipTest('No deployed actual legacy calibration fixture yet')
        cfg=load_contract(p,'validation',10,10)
        self.assertEqual(cfg['event_policy'],'legacy')
        self.assertEqual(sum(cfg['events_per_view']),cfg['accepted_events'])

    def test_checkpoint_callback_does_not_change_original_mlem_images_or_histories(self):
        g=self.geometry();response=ViewResponse(torch.tensor([[.2,.3,.4],[.5,.1,.3]]),g)
        projection=torch.tensor([[8.],[11.]]);blocks=[[torch.tensor([[.3,.2],[.2,.4]])]]
        single=response.sensitivity();compton=torch.tensor([[.4],[.5]])
        baseline=compton_and_joint_mlem(response,projection,blocks,single,compton,100,50)
        with tempfile.TemporaryDirectory() as d:
            def save(i,hd,hj):write_checkpoint(Path(d),'angular',i,hd,hj,g,'contract')
            actual=compton_and_joint_mlem(response,projection,blocks,single,compton,100,50,checkpoint_callback=save)
            for a,b in zip(actual,baseline):
                for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertTrue((Path(d)/'checkpoint_000050/checkpoint_manifest.json').exists())
            self.assertTrue((Path(d)/'checkpoint_000100/checkpoint_manifest.json').exists())

    def test_all_forty_snapshots_required_and_hash_corruption_rejected(self):
        g=self.geometry();hd=[];hj=[]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for frame in range(40):
                hd.append(torch.tensor([[frame+1.],[frame+2.]]));hj.append(hd[-1]*2)
                write_checkpoint(root,'angular',(frame+1)*50,hd,hj,g,'sha')
            history={CHANNELS[0]:torch.stack(hd).numpy().reshape(40,2),CHANNELS[1]:torch.stack(hj).numpy().reshape(40,2)}
            self.assertEqual(len(verify_checkpoints(root,history,np.array([0,2]),3,'angular','sha')),40)
            p=root/'checkpoint_001000'/f'Image_{CHANNELS[0]}_active.float32';p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):verify_checkpoints(root,history,np.array([0,2]),3,'angular','sha')

    def test_invalid_checkpoint_is_unpublished_and_existing_frame_cannot_be_overwritten(self):
        g=self.geometry()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):write_checkpoint(root,'angular',50,[torch.tensor([[float('nan')],[1.]])],[torch.ones(2,1)],g,'sha')
            self.assertEqual(list(root.iterdir()),[])
            write_checkpoint(root,'angular',50,[torch.ones(2,1)],[torch.ones(2,1)],g,'sha')
            with self.assertRaises(ValueError):write_checkpoint(root,'angular',50,[torch.ones(2,1)],[torch.ones(2,1)],g,'sha')


if __name__=='__main__':unittest.main()
