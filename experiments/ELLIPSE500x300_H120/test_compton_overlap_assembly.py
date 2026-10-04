"""Synthetic mathematical contracts, not transport or reconstruction evidence."""
import unittest
import numpy as np
import torch
from compton_overlap_assembly import OverlapAssembly,sensitivity_contribution


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.assembly=OverlapAssembly([0,1,2],[1],np.array([[0,2],[1,0],[2,3],[3,1]]),4)
        self.raw=torch.tensor([[2.,3.,5.,7.],[1.,2.,4.,8.]])
        self.obj=torch.tensor([[.2],[.4]]);self.ref=torch.tensor([[6.],[5.]])

    def test_reference_keeps_outside_ellipse_and_integrals_do_not_get_f_again(self):
        rows,norm=self.assembly.assemble(self.raw,self.obj,self.ref,0)
        torch.testing.assert_close(norm,torch.tensor([20.,18.],dtype=torch.float64))
        torch.testing.assert_close(rows,torch.tensor([[2/20,.2/20,5/20],[1/18,.4/18,4/18]]))

    def test_rotation_and_exact_transpose(self):
        rows,norm=self.assembly.assemble(self.raw,self.obj,self.ref,1)
        torch.testing.assert_close(norm,torch.tensor([21.,19.],dtype=torch.float64))
        torch.testing.assert_close(rows,torch.tensor([[5/21,.2/21,7/21],[4/19,.4/19,8/19]]))
        x=torch.tensor([[.3],[.5],[.8]]);y=torch.tensor([[.7],[.2]])
        self.assertLess(float(abs(y.T@(rows@x)-x.T@(rows.T@y))),1e-7)

    def test_matching_sensitivity_keeps_emission_denominator(self):
        rows,_=self.assembly.assemble(self.raw,self.obj,self.ref,0)
        expected=rows.double().sum(0)*120/1000/20
        torch.testing.assert_close(sensitivity_contribution(rows,1000,120,20),expected)

    def test_zero_active_and_invalid_layout_hold_without_dropping_events(self):
        with self.assertRaises(ValueError):self.assembly.assemble(self.raw,self.obj,self.ref,0,input_layout='normalized')
        raw=torch.tensor([[0.,0.,0.,1.]])
        with self.assertRaises(ValueError):self.assembly.assemble(raw,torch.zeros(1,1),torch.ones(1,1),0)
        with self.assertRaises(ValueError):self.assembly.assemble(self.raw,-self.obj,self.ref,0)


if __name__=='__main__':unittest.main()
