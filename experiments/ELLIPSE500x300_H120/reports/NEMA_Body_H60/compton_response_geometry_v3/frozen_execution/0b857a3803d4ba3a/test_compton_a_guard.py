"""Halo coordinate coverage, common-point rejection and precise measure tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import generate_compton_a_guard as guard
from prepare_compton_overlap_measure import precise_cell_volume
from compton_boundary_quadrature import cell_quadrature


class GuardTests(unittest.TestCase):
    def test_halo_exactly_covers_new_cartesian_domain(self):
        locations=set()
        for spec in guard.specs()[1:7]:
            x,y,z=guard.axes(spec)
            self.assertLess(11520*np.prod(spec['shape']),2**31)
            for zv in z:
                for yv in y:
                    for xv in x: locations.add((xv,yv,zv))
        expected=set()
        for z in [-60., *np.arange(40)*3-58.5, 60.]:
            for y in np.arange(87)*6-258:
                for x in np.arange(87)*6-258:
                    if z in (-60,60) or abs(x)==258 or abs(y)==258: expected.add((x,y,z))
        self.assertEqual(locations,expected)
        self.assertEqual(len(locations),28898)

    def test_anchor_reproduces_original_grid_points_and_fails_changed_bin(self):
        with tempfile.TemporaryDirectory() as name, patch.object(guard,'NDET',1):
            root=Path(name);source=root/'original';folder=root/'anchor';source.mkdir();folder.mkdir()
            spec=guard.specs()[0]
            original=np.arange(40*85*85,dtype=np.float32).reshape(1,40,85,85)+1
            subset=original[:,19:21,41:44,41:44]
            for kind,new in [('PE_SysMat','pe.sysmat'),('PE_Windowed_SysMat','pe_windowed.sysmat'),
                ('Scatter_SysMat','Scatter_SysMat_shift_0.000000_0.000000_0.000000.sysmat'),
                ('SysMat_withScatter','SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat')]:
                suffix='_v4' if kind.startswith('PE_') else ''
                original.tofile(source/f'{kind}_shift_0.000000_0.000000_0.000000{suffix}.sysmat')
                subset.tofile(folder/new)
            self.assertEqual(guard.validate_anchor(folder,source,spec)['status'],'PASSED')
            corrupted=subset.copy();corrupted.flat[0]*=1.01;corrupted.tofile(folder/'pe.sysmat')
            with self.assertRaises(ValueError):guard.validate_anchor(folder,source,spec)
            self.assertEqual(json.loads((folder/'anchor_gate.json').read_text())['status'],'HOLD')

    def test_precise_volume_matches_independent_gauss_intersection(self):
        for cell in [(159.,165.,-.04,.04),(153.,159.,.92,1.0),(249.,255.,-.022,.022)]:
            value,error=precise_cell_volume(cell)
            q,w=cell_quadrature(cell,1.5,32,8,8)
            self.assertGreater(value,0)
            self.assertLess(abs(w.sum()/value-1),1e-10)
            self.assertLess(error,1e-7)
            self.assertTrue(np.all((q[:,0]/250)**2+(q[:,1]/150)**2<=1+1e-12))


if __name__=='__main__':unittest.main()
