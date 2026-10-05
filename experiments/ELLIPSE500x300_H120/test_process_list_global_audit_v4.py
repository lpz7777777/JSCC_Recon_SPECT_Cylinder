"""Physical partition, Jacobian and rotation tests for bounded diagnostics."""
import json
from pathlib import Path
import unittest
import numpy as np
from geometry import grid, rotations
from process_list_global_audit_v4 import map_source_cell, full_sector, transfer, fine_labels
from energy_probability_probe_v4 import conditioned_pdf_pit
from scipy.integrate import quad
from scipy.stats import truncnorm


class AuditGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((Path(__file__).parent/'config.json').read_text())
        cls.coords,cls.cells,cls.slices=grid(cls.config)

    def test_emission_cell_rotation_matches_permutation(self):
        rot,_=rotations(self.config,self.slices)
        indices=np.arange(3301)
        for view in range(20):
            mapped=map_source_cell(self.coords[:3301],self.slices,view)
            np.testing.assert_array_equal(mapped,rot[:,view])

    def test_radial_and_axial_boundaries_assign_volume(self):
        p=np.array([[0,0,-60],[2.99,0,-57.01],[3.01,0,-57],[254.99,0,59.99]])
        mapped=map_source_cell(p,self.slices)
        self.assertEqual(mapped[0],0)
        self.assertEqual(mapped[1],0)
        self.assertEqual(mapped[2],3302)
        self.assertEqual(mapped[3],39*3301+self.slices[-1][0])
        labels=fine_labels(self.coords)
        self.assertEqual(len(np.unique(labels)),192)

    def test_sector_quadrature_has_correct_measure_and_moments(self):
        cell=self.cells[300]
        points,w=full_sector(cell,1.5,8)
        self.assertAlmostEqual(w.sum(),1.)
        self.assertAlmostEqual(w@points[:,2],1.5)
        self.assertAlmostEqual(w@((points[:,2]-1.5)**2),.75)
        self.assertAlmostEqual(w@np.sum(points[:,:2]**2,axis=1),(cell[0]**2+cell[1]**2)/2)
        radial_centroid=2/3*(cell[1]**3-cell[0]**3)/(cell[1]**2-cell[0]**2)
        expected_x=radial_centroid*(np.sin(cell[3])-np.sin(cell[2]))/(cell[3]-cell[2])
        self.assertAlmostEqual(w@points[:,0],expected_x,places=8)

    def test_free_compton_limits(self):
        source=np.array([0.,0.,0.]);first=np.array([1.,0.,0.])
        energy,angle=transfer(first,np.array([2.,0.,0.]),source)
        self.assertAlmostEqual(float(energy),0.)
        self.assertAlmostEqual(float(angle),0.)
        energy,angle=transfer(first,np.array([0.,0.,0.]),source)
        self.assertAlmostEqual(float(energy),2*.440**2/(.511+2*.440))
        self.assertAlmostEqual(float(angle),np.pi)

    def test_truncated_energy_mixture_is_a_density(self):
        means=np.array([.08,.14,.23]);sigma=np.array([.012,.017,.020])
        lo,hi=.09,.245
        area,_=quad(lambda energy:np.exp(conditioned_pdf_pit(energy,lo,hi,means,sigma)[0]),lo,hi,epsabs=1e-10)
        self.assertAlmostEqual(area,1.,places=9)
        self.assertAlmostEqual(conditioned_pdf_pit(lo,lo,hi,means,sigma)[1],0.)
        self.assertAlmostEqual(conditioned_pdf_pit(hi,lo,hi,means,sigma)[1],1.)

    def test_extreme_selection_tail_remains_finite(self):
        mean,sd,lo,hi,energy=.003,.004,.05,.25,.051
        ll,pit=conditioned_pdf_pit(energy,lo,hi,np.array([mean]),np.array([sd]))
        a,b=(lo-mean)/sd,(hi-mean)/sd
        self.assertAlmostEqual(ll,truncnorm.logpdf(energy,a,b,loc=mean,scale=sd),places=9)
        self.assertAlmostEqual(pit,truncnorm.cdf(energy,a,b,loc=mean,scale=sd),places=9)


if __name__=='__main__':unittest.main()
