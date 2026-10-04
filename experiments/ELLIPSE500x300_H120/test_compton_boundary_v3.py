"""Meaningful operator, measure and interpolation tests for R2 interfaces."""
import unittest
import numpy as np
import torch
from compton_boundary_quadrature import cell_quadrature,rotate_to_detector,PolarResponseField
from torch_active_operator import ActiveGeometry

class BoundaryTests(unittest.TestCase):
    def test_constant_volume_and_true_ellipse_intersection(self):
        cell=(145.,155.,-.2,.2)
        p,w=cell_quadrature(cell,1.5,16,8,8)
        self.assertTrue(np.all((p[:,0]/250)**2+(p[:,1]/150)**2<=1+1e-14))
        self.assertTrue(np.all(w>0))
        q,v=cell_quadrature(cell,1.5,32,12,12)
        self.assertLess(abs(v.sum()/w.sum()-1),1e-9)
        _,full=cell_quadrature(cell,1.5,8,4,4,ellipse=False)
        self.assertAlmostEqual(full.sum(),(155**2-145**2)*.4/2*3,places=9)

    def test_rotation_and_interpolation_of_affine_density(self):
        points=np.array([[2.,1.,-59.],[1.,-2.,59.]])
        restored=rotate_to_detector(rotate_to_detector(points,3),17)
        np.testing.assert_allclose(restored,points,atol=1e-14)
        xy=np.array([[-3.,-3.],[-3.,3.],[3.,-3.],[3.,3.]])
        field=np.array([100+xy[:,0]+2*xy[:,1]+z*.1 for z in np.arange(40)*3-58.5])
        interpolation=PolarResponseField(xy)
        result=interpolation.evaluate(field,interpolation.cache(points))
        np.testing.assert_allclose(result,100+points[:,0]+2*points[:,1]+points[:,2]*.1,atol=1e-12)
        with self.assertRaises(ValueError):interpolation.cache(np.array([[4.,0.,0.]]))

    def test_integrated_layout_does_not_rotate_or_multiply_fraction_twice(self):
        active=np.array([0,2]);fraction=np.array([1.,0.,.01]);rotation=np.array([[0,2],[1,1],[2,0]])
        geometry=ActiveGeometry(active,fraction,rotation)
        rows=torch.tensor([[2.,3.],[4.,1.]])
        direct=geometry.compact(rows,1,'object_active_integrated')
        torch.testing.assert_close(direct,rows,rtol=0,atol=0)
        s=torch.tensor([5.,7.]);torch.testing.assert_close(geometry.compton_sensitivity(s,'object_active_integrated')[:,0],s,rtol=0,atol=0)
        x=torch.tensor([[.3],[.7]]);y=torch.tensor([[.8],[.2]])
        self.assertLess(float(abs(y.T@(direct@x)-x.T@(direct.T@y))),1e-6)
        with self.assertRaises(ValueError):geometry.compact(rows,0,'unknown')
        with self.assertRaises(ValueError):geometry.compton_sensitivity(s)

if __name__=='__main__':unittest.main()
