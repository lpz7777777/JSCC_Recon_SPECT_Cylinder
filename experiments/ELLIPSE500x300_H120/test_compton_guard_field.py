"""Interior compatibility and complete-cell support of sampled halo interpolation."""
import json
from pathlib import Path
import unittest
import numpy as np
from geometry import grid
from compton_boundary_quadrature import (PolarResponseField,GuardedPolarResponseField,
    cell_quadrature,rotate_to_detector)
from build_compton_a_guard_field import interpolate_cartesian


class GuardFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coords,cls.cells,_=grid(json.loads(Path(__file__).with_name('config.json').read_text()))
        cls.xy=cls.coords[:3301,:2]
        a=np.arange(140)*2*np.pi/140
        cls.extended=np.vstack((cls.xy,258*np.column_stack((np.cos(a),np.sin(a)))))
        cls.z=np.r_[-60.,np.arange(40)*3-58.5,60.]
        cls.guard=GuardedPolarResponseField(cls.extended,3301,cls.z)

    def test_original_interior_field_is_unchanged(self):
        rng=np.random.default_rng(8610);old=rng.random((40,3301))
        extended=rng.random((42,3441));extended[1:-1,:3301]=old
        points=self.coords[rng.choice(len(self.coords),100,replace=False)].copy()
        points[:,:2]*=.9
        expected=PolarResponseField.evaluate(old,PolarResponseField(self.xy).cache(points))
        actual=self.guard.evaluate(extended,self.guard.cache(points))
        np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)

    def test_full_circle_partial_reference_is_covered_for_all_views(self):
        old=np.load(Path(__file__).parent/'generated/Geometry/geometry.npz')['ellipse_fraction'][:3301]
        partial=np.flatnonzero((old>1e-12)&(old<1-1e-12))
        points=np.concatenate([cell_quadrature(self.cells[c],58.5,8,4,4,ellipse=False)[0] for c in partial])
        for view in range(20):self.guard.cache(rotate_to_detector(points,view))
        with self.assertRaises(ValueError):self.guard.cache(np.array([[259.,0.,0.]]))
        with self.assertRaises(ValueError):self.guard.cache(np.array([[0.,0.,60.01]]))

    def test_nonuniform_z_and_cartesian_halo_preserve_affine_functions(self):
        points=np.array([[254.9,0.,-59.75],[0.,-254.9,59.75],[20.,30.,0.]])
        values=np.array([1000+self.extended[:,0]+2*self.extended[:,1]+.5*z for z in self.z])
        np.testing.assert_allclose(self.guard.evaluate(values,self.guard.cache(points)),
            1000+points[:,0]+2*points[:,1]+.5*points[:,2],atol=1e-12)
        yy,xx=np.meshgrid(np.arange(87)*6-258,np.arange(87)*6-258,indexing='ij')
        np.testing.assert_allclose(interpolate_cartesian(1000+xx+2*yy,points[:,:2]),
            1000+points[:,0]+2*points[:,1],atol=1e-12)


if __name__=='__main__':unittest.main()
