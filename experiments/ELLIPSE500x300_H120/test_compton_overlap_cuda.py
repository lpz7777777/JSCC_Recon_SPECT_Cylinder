"""Device/CPU interpolation and segmented reduction correctness tests."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from compton_overlap_cuda import DeviceARows,segment_sum
from compton_boundary_quadrature import GuardedPolarResponseField
from compton_cartesian_patch import CartesianPatch


class DeviceTests(unittest.TestCase):
    def test_device_interpolation_matches_both_independent_original_formulas(self):
        xy=np.array([[-5.,-5.],[5.,-5.],[0.,5.]])
        guard=GuardedPolarResponseField(xy,3,np.array([-60.,0.,60.]))
        patch=CartesianPatch([np.array([-5.,5.])]*3)
        points=np.array([[0.,0.,-1.],[-1.,1.,2.],[.3,.4,.5]])
        base=np.arange(27,dtype=float).reshape(3,3,3)/10
        fine=np.arange(24,dtype=float).reshape(3,2,2,2)/10
        provider=SimpleNamespace(base=base,patches=[dict(values=fine)],selected=np.array([2,0,1]),scales=np.array([.8,1.1,1.3]))
        cp=np.array([2,0,2]);compiled=dict(count=3,groups={0:(np.array([0,2]),guard.cache(points[[0,2]])),
            1:(np.array([1]),patch.cache(points[[1]]))})
        expected=np.empty((len(cp),3))
        for row,c in enumerate(cp):
            expected[row,[0,2]]=guard.evaluate(base[c],compiled['groups'][0][1])
            expected[row,1]=(patch.evaluate(fine[provider.selected[c]],compiled['groups'][1][1])*provider.scales[c]).item()
        for device in ('cpu',*(('cuda:0',) if torch.cuda.is_available() else ())):
            actual=DeviceARows(provider,cp,torch.device(device)).evaluate(compiled).cpu().numpy()
            np.testing.assert_allclose(actual,expected,rtol=1e-14,atol=1e-14)

    def test_contiguous_reduction_is_independent_of_node_chunk_and_empty_slots(self):
        rng=np.random.default_rng(23);values=rng.random((5,237))
        slots=np.repeat([0,2,5,7],[13,97,101,26]);expected=np.stack([
            np.bincount(slots,weights=row,minlength=9) for row in values])
        for device in ('cpu',*(('cuda:0',) if torch.cuda.is_available() else ())):
            x=torch.tensor(values,device=device)
            for block in (17,237):
                actual=sum((segment_sum(x[:,lo:lo+block],slots[lo:lo+block],9)
                            for lo in range(0,237,block)),torch.zeros((5,9),dtype=torch.float64,device=device))
                np.testing.assert_allclose(actual.cpu().numpy(),expected,rtol=1e-13,atol=1e-12)
        with self.assertRaises(ValueError):segment_sum(torch.ones(1,3),np.array([0,2,1]),3)


if __name__=='__main__':unittest.main()
