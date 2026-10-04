"""Physical volume, field selection and bounded accumulation contracts."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from compton_event_response import ComptonEventSettings, PreparedComptonEvents
from compton_overlap_integrator import OverlapIntegrator, reference_cell_bounds
from prepare_compton_overlap_measure import precise_cell_volume
from compton_boundary_quadrature import GuardedPolarResponseField, cell_quadrature, rotate_to_detector
from compton_cell_response_field import GuardedCellCache


class AnalyticalField:
    def __init__(self, negative=False):
        self.selections = []; self.negative = negative

    def assert_production_ready(self):
        raise ValueError('Synthetic field is not production certified')

    def choose(self, bounds):
        self.selections.append(bounds); return 0

    def cache(self, points, fields):
        return dict(points=points.copy(), fields=fields.copy())

    def evaluate(self, crystal, cache):
        if self.negative:
            return np.full(len(cache['points']), -1.)
        return np.full(len(cache['points']), crystal+1.)

    def field_counts(self, cache):
        return {'analytical': len(cache['points'])}


def constant_kernel(prepared, points, settings):
    return torch.ones((prepared.count, len(points)), dtype=torch.float64)


class IntegratorTests(unittest.TestCase):
    def setUp(self):
        self.cells = np.array([[147.,153.,np.pi/2-.05,np.pi/2+.05],
                               [243.,249.,-.03,.03]])
        self.coords = np.array([[0.,150.,1.5],[246.,0.,1.5],
                                [0.,150.,58.5],[246.,0.,58.5]])
        self.partial = np.arange(4)
        self.settings = ComptonEventSettings(.440,.14,.27,.05,.35,geometry_mode='stable_float64')
        values = torch.ones(2)
        self.prepared = PreparedComptonEvents(torch.tensor([1,2]),torch.tensor([3,4]),values,values,
            torch.zeros(2,3),torch.ones(2,3),torch.ones(2,3),torch.ones(2,3))

    def make(self, field=None, **kwargs):
        return OverlapIntegrator(self.cells,self.coords,self.partial,2,field or AnalyticalField(),
            self.settings,diagnostic_only=True,kernel=constant_kernel,**kwargs)

    def test_object_volume_independent_integral_and_full_reference(self):
        provider = AnalyticalField(); integrator = self.make(provider)
        obj, ref = integrator.integrate(self.prepared,7,(24,8,8))
        expected = np.array([precise_cell_volume(self.cells[i%2])[0] for i in range(4)])
        full = np.array([(self.cells[i%2,1]**2-self.cells[i%2,0]**2)*
                         (self.cells[i%2,3]-self.cells[i%2,2])/2*3 for i in range(4)])
        np.testing.assert_allclose(obj.numpy(),np.outer([1,2],expected),rtol=1e-11)
        np.testing.assert_allclose(ref.numpy(),np.outer([1,2],full),rtol=1e-12)
        self.assertEqual(len(provider.selections),4)
        self.assertTrue(np.all(obj.numpy() <= ref.numpy()*(1+1e-12)))

    def test_node_and_cell_blocking_preserve_every_contribution(self):
        a = self.make(node_chunk=17,cells_per_block=1,cache_bytes=0)
        b = self.make(node_chunk=997,cells_per_block=4)
        for x,y in zip(a.integrate(self.prepared,0,(16,4,4)),b.integrate(self.prepared,0,(16,4,4))):
            torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-10)
        self.assertEqual(a.cache_size,0)
        self.assertLessEqual(b.cache_size,b.limit)

    def test_rotated_bounds_cover_the_entire_sector_not_just_representative_point(self):
        cell = (249.,255.,-.04,.04)
        for view in range(20):
            lo,hi = reference_cell_bounds(cell,58.5,view)
            theta=np.linspace(cell[2],cell[3],901)-view*np.pi/10
            xy=np.concatenate([np.column_stack((r*np.cos(theta),r*np.sin(theta))) for r in (249,255)])
            self.assertTrue(np.all(xy >= lo[:2]-1e-10) and np.all(xy <= hi[:2]+1e-10))
            np.testing.assert_array_equal((lo[2],hi[2]),(57,60))

    def test_uncertified_field_cannot_enter_production(self):
        with self.assertRaises(ValueError):
            OverlapIntegrator(self.cells,self.coords,self.partial,2,AnalyticalField(),self.settings)

    def test_invalid_A_holds_without_silently_removing_an_event(self):
        with self.assertRaises(ValueError):
            self.make(AnalyticalField(negative=True)).integrate(self.prepared,0,(8,4,4))

    def test_factorized_guard_cache_matches_every_original_stencil_at_two_layers(self):
        xy=np.array([[-500.,-500.],[500.,-500.],[0.,500.]])
        guard=GuardedPolarResponseField(xy,3,np.array([-60.,0.,60.]))
        cache=GuardedCellCache(guard)
        for z in (-58.5,58.5):
            points,_=cell_quadrature(self.cells[0],z,16,4,4)
            points=rotate_to_detector(points,3)
            actual=cache.compile(points,(3,0,0,(16,4,4)),4)
            expected=guard.cache(points)
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        self.assertEqual(len(cache.xy),1)
        self.assertLessEqual(cache.bytes,cache.limit)


if __name__=='__main__': unittest.main()
