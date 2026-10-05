"""Pilot safeguards: complete identities, no silent drops, unchanged angular rows."""
from pathlib import Path
import unittest
import numpy as np
import torch
from run_energy_preflight_v5 import (partition_indices,prepare_frozen,rows_for,
    settings,validate_whole_geometry,digest)
from compton_event_response import build_detector_position_variance,build_compton_cone_weights
from compton_energy_probability_v5 import ContinuousTransferLaw

HERE=Path(__file__).parent


class PilotSafeguards(unittest.TestCase):
    def test_ragged_rank_partitions_preserve_every_identity_once(self):
        for n in (0,3,97):
            indices=np.arange(n,dtype=np.int64)*3
            for world in (4,8):
                parts=[partition_indices(indices,r,world)[0] for r in range(world)]
                np.testing.assert_array_equal(np.concatenate(parts),indices)
        for bad in ([3,2],[1,1],[1.,2.]):
            with self.assertRaises(ValueError):partition_indices(np.array(bad),0,4)

    def fixture(self):
        detector=torch.tensor([[0.,-300.,0.],[90.,-330.,0.],[0.,-360.,0.],[0.,-390.,0.]])
        variance=build_detector_position_variance(detector,0.)
        raw=np.array([[1,.14,2,.28],[1,.13,2,.29]],np.float32)
        coords=torch.tensor([[0.,0.,0.],[60.,0.,0.],[-60.,0.,0.],[0.,0.,45.]])
        return raw,detector,variance,coords,torch.ones((4,4))

    def test_invalid_frozen_events_are_rejected_not_silently_dropped(self):
        raw,d,v,_,_=self.fixture();raw[0,3]=.01
        with self.assertRaises(ValueError):prepare_frozen(raw,d,v,'cpu')

    def test_angular_path_retains_existing_kernel_and_normalization(self):
        raw,d,v,x,B=self.fixture();p=prepare_frozen(raw,d,v,'cpu')
        prior=build_compton_cone_weights(p,x,settings())*B[p.cpnum1-1]
        prior/=prior.sum(1,keepdim=True)
        actual=rows_for(raw,d,v,x,B,None,'angular')
        torch.testing.assert_close(actual,prior,rtol=0,atol=0)
        torch.testing.assert_close(actual.sum(1),torch.ones(len(raw)))

    def test_continuous_candidate_order_and_chunk_do_not_change_identity(self):
        raw,d,v,x,B=self.fixture();path=HERE/'transfer_training_summary.json'
        if not path.exists():path=HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4/evidence/energy_energy_probability_summary.json'
        law=ContinuousTransferLaw.load(path)
        full=rows_for(raw,d,v,x,B,law,'continuous_energy')
        chunks=torch.cat([rows_for(r[None],d,v,x,B,law,'continuous_energy') for r in raw])
        reverse=rows_for(raw[::-1].copy(),d,v,x,B,law,'continuous_energy').flip(0)
        torch.testing.assert_close(chunks,full,rtol=1e-6,atol=1e-7)
        torch.testing.assert_close(reverse,full,rtol=1e-6,atol=1e-7)

    def test_whole_cell_geometry_is_frozen_and_has_no_fractional_columns(self):
        path=HERE/'whole_geometry.npz'
        if not path.exists():path=HERE/'generated/process_list_global_audit_v4/WholeCellGeometry/geometry.npz'
        g=validate_whole_geometry(path,digest(path));self.assertEqual(len(g['active_indices']),78920)
        with self.assertRaises(ValueError):validate_whole_geometry(path,'0'*64)


if __name__=='__main__':unittest.main()
