"""Frozen real geometry tests for the 12 independent physical A controls."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import unittest
import numpy as np
from geometry import grid
from compton_boundary_quadrature import cell_quadrature,rotate_to_detector
from generate_compton_a_guard import axes,digest


class RegionalPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        here=Path(__file__).resolve().parent
        cls.plan=json.loads((here/'reports/NEMA_Body_H60/compton_response_geometry_v3/regional_a_plan.json').read_text())
        cls.geo=np.load(here/'generated/Geometry/geometry.npz')
        cls.coords,cls.cells,_=grid(json.loads((here/'config.json').read_text()))
        cls.rotation=cls.geo['inverse_rotation'];cls.n=3301

    def test_actual_object_cell_view_maps_to_declared_detector_cell(self):
        for c in self.plan['cases']:
            i=c['layer']*self.n+c['object_xy_index'];v=c['view']-1
            self.assertEqual(int(self.rotation[i,v]%self.n),c['detector_xy_index'])
            np.testing.assert_allclose(rotate_to_detector(self.coords[i:i+1],v)[0,:2],
                c['actual_detector_xy_mm'],atol=1e-10,rtol=0)
            self.assertGreater(self.geo['ellipse_fraction'][i],0)
            self.assertLess(self.geo['ellipse_fraction'][i],1)

    def test_entire_reference_and_intersection_quadrature_have_support(self):
        for c in self.plan['cases']:
            for ellipse in (False,True):
                points,_=cell_quadrature(self.cells[c['object_xy_index']],c['z_mm'],32,12,12,ellipse=ellipse)
                points=rotate_to_detector(points,c['view']-1)
                for s in c['parts']:
                    for col,axis in enumerate(axes(s)):
                        self.assertGreaterEqual(float(points[:,col].min()),float(axis[0])-1e-10)
                        self.assertLessEqual(float(points[:,col].max()),float(axis[-1])+1e-10)

    def test_independent_grids_share_every_coarse_point_and_safe_index_budget(self):
        self.assertEqual(len(self.plan['cases']),12)
        for c in self.plan['cases']:
            for coarse,fine in zip(axes(c['parts'][0]),axes(c['parts'][1])):
                np.testing.assert_array_equal(coarse,fine[::2])
            for s in c['parts']:self.assertLess(11520*int(np.prod(s['shape'])),2**31)


if __name__=='__main__':unittest.main()
