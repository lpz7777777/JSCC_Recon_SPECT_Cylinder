"""Tests of production-mode stable geometry against independent derivatives."""
import sys
from pathlib import Path
import unittest
from dataclasses import replace
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from compton_event_response import (ComptonEventSettings,PreparedComptonEvents,
    stable_compton_geometry,build_compton_cone_weights,min_standardized_compton_arm)
import test_compton_geometry_stability as reference_tests


class ProductionGeometryTests(reference_tests.StableGeometryTests):
    def setUp(self):
        super().setUp()
        # Exercise production geometry with the inherited independent tests.
        original=reference_tests.stable_geometry
        self.addCleanup(setattr,reference_tests,'stable_geometry',original)
        reference_tests.stable_geometry=stable_compton_geometry

    def prepared(self,device='cpu'):
        t=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
        return PreparedComptonEvents(torch.tensor([1],device=device),torch.tensor([2],device=device),
            t([.204927]),t([.179933]),t([[0.,300.,0.]]),t([[0.,360.,.003]]),
            t([[.75,.75,.75]]),t([[1/3,3.,1/3]]))

    def test_default_mode_is_explicit_legacy(self):
        s=ComptonEventSettings(.440,.13,.277,.05,.35)
        p=self.prepared();c=torch.tensor([[20.,10.,4.],[-180.,-105.,-4.5]])
        torch.testing.assert_close(build_compton_cone_weights(p,c,s),
            build_compton_cone_weights(p,c,replace(s,geometry_mode='legacy')),rtol=0,atol=0)

    def test_stable_kernel_and_quality_are_finite_across_collinearity(self):
        p=self.prepared();s=ComptonEventSettings(.440,.13,.277,.05,.35,geometry_mode='stable_float64')
        c=torch.tensor([[0.,0.,0.],[0.,600.,.02],[1e-5,600.,.02],[-10.,20.,.001]])
        k=build_compton_cone_weights(p,c,s);q=min_standardized_compton_arm(p,c,s)
        self.assertEqual(k.dtype,torch.float32);self.assertEqual(q.dtype,torch.float64)
        self.assertTrue(bool(torch.isfinite(k).all() and (k>=0).all() and torch.isfinite(q).all()))
        split=torch.cat([build_compton_cone_weights(p,c[i:i+1],s) for i in range(len(c))],1)
        torch.testing.assert_close(k,split,rtol=0,atol=0)

    def test_invalid_mode_and_coincident_positions_fail_explicitly(self):
        with self.assertRaises(ValueError):ComptonEventSettings(.440,.13,.277,.05,.35,geometry_mode='typo')
        with self.assertRaises(ValueError):
            stable_compton_geometry(torch.zeros(1,1,3),torch.ones(1,1,3),torch.ones(1,3),torch.ones(1,3))

    def test_frozen_real_event_quality_after_stable_mode_selection(self):
        s=ComptonEventSettings(.440,.13*np.sqrt(511/440),.277,.05,.35,geometry_mode='stable_float64')
        for device in (['cpu','cuda:0'] if torch.cuda.is_available() else ['cpu']):
            t=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
            p=PreparedComptonEvents(torch.tensor([9554],device=device),torch.tensor([1091],device=device),
                t([.204927]),t([.179933]),t([[103.95,390.,-30.45]]),t([[69.3,330.,-27.3]]),
                t([[1/3,3.,1/3]]),t([[.75,.75,.75]]))
            c=t([[-181.86533479473212,-105.,-4.5],[227.0441547,109.3387023,58.5]])
            q=min_standardized_compton_arm(p,c,s)
            self.assertAlmostEqual(float(q),3.35468137168,places=5)


if __name__=='__main__':unittest.main()
