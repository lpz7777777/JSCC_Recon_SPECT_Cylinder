"""Pilot safeguards: complete identities, no silent drops, unchanged angular rows."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import Mock,patch
import numpy as np
import torch
from run_energy_preflight_v5 import (partition_indices,prepare_frozen,rows_for,
    settings,validate_whole_geometry,digest,original_sparse_projector)
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

    def test_original_projector_moves_coordinates_to_cpu_before_construction(self):
        path=HERE/'whole_geometry.npz'
        if not path.exists():path=HERE/'generated/process_list_global_audit_v4/WholeCellGeometry/geometry.npz'
        coords=torch.tensor(np.load(path)['coordinates_mm'],dtype=torch.float32)
        source=Mock();source.detach.return_value.cpu.return_value=coords
        from run_energy_preflight_v5 import build_compton_sparse_projector
        with patch('run_energy_preflight_v5.build_compton_sparse_projector',wraps=build_compton_sparse_projector) as builder:
            projector=original_sparse_projector(source,torch.device('cpu'))
            source.detach.return_value.cpu.assert_called_once_with()
            self.assertEqual(builder.call_args.args[0].device.type,'cpu')
            self.assertEqual(projector.coor_coarse.device.type,'cpu')
        if torch.cuda.is_available():
            actual=original_sparse_projector(coords.cuda(),torch.device('cuda:0'))
            self.assertEqual(actual.coor_coarse.device.type,'cuda')
            torch.testing.assert_close(actual.coor_coarse.cpu(),projector.coor_coarse,rtol=0,atol=0)

    def test_reuse_accepts_file_receipt_from_legacy_verifier_with_no_return_value(self):
        from verify_energy_preflight_v5 import verify_reused_regression
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);result=root/'passed';result.mkdir()
            (result/'run_manifest.json').write_text('{}');allocation=root/'allocation.txt';allocation.write_text('allocation')
            receipt=dict(passed=True,accepted_events=91231,iterations=50)
            (result/'verification.json').write_text(json.dumps(receipt))
            reuse=dict(job=1666205,result=str(result),allocation=str(allocation),baseline=str(root/'baseline'),
                run_manifest_sha256=digest(result/'run_manifest.json'),allocation_sha256=digest(allocation),
                verification_sha256=digest(result/'verification.json'))
            contract=root/'contract.json';contract.write_text(json.dumps(dict(files={},regression_reuse=reuse)))
            with patch('verify_energy_preflight_v5.verify_first_scatter',return_value=None) as verifier:
                self.assertEqual(verify_reused_regression(contract),receipt)
                verifier.assert_called_once()
            (result/'run_manifest.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):verify_reused_regression(contract)


if __name__=='__main__':unittest.main()
