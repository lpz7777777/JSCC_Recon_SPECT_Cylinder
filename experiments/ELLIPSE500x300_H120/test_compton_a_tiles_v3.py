import unittest
import tempfile
from pathlib import Path
import numpy as np
from plan_compton_a_tiles_v3 import cover_xy_boxes,tile_spec
from generate_compton_a_guard import axes,digest
from generate_compton_a_tile_pilot_v3 import extract_selected,remove_verified_raw


class TileCoverageTests(unittest.TestCase):
    def test_interface_requires_both_interpolation_tiles(self):
        mask,tiles=cover_xy_boxes([(np.array([-7.,233.,-60]),np.array([7.,247.,60]))])
        for x in np.linspace(-7,7,61):
            for y in np.linspace(233,247,61):
                ix,iy=np.floor((np.array([x,y])+258)/.75).astype(int)
                self.assertTrue(mask[iy,ix])
                self.assertTrue(np.any((tiles==[ix//16,iy//16]).all(axis=1)))

    def test_adjacent_physical_nodes_are_identical(self):
        a,b=axes(tile_spec(21,41,5)),axes(tile_spec(21,42,5))
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[2],b[2])
        self.assertEqual(a[1][-1],b[1][0]);self.assertEqual(a[1][-1],246)
        np.testing.assert_array_equal(a[2],np.arange(0,12.01,.75))

    def test_no_extrapolation_or_unknown_tile(self):
        with self.assertRaises(ValueError):cover_xy_boxes([(np.array([-259,0,0]),np.array([-250,3,0]))])
        with self.assertRaises(ValueError):tile_spec(43,0,0)

    def test_compaction_preserves_detector_identity_and_block_size(self):
        values=np.arange(12*3*4*5,dtype='<f4').reshape(12,3,4,5)/17
        selected=np.array([0,3,6,11])
        with tempfile.TemporaryDirectory() as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b'
            extract_selected(values,a,selected,1);extract_selected(values,b,selected,3)
            self.assertEqual(digest(a),digest(b))
            np.testing.assert_array_equal(np.fromfile(a,dtype='<f4').reshape(4,3,4,5),values[selected])

    def test_cleanup_cannot_escape_new_tile_or_start_with_unverified_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);tile=base/'tile';tile.mkdir();keep=tile/'raw';keep.write_bytes(b'keep')
            outside=base/'outside';outside.write_bytes(b'outside')
            with self.assertRaises(ValueError):
                remove_verified_raw(tile,{'matrices':{'raw':{'sha256':digest(keep)},
                    '../outside':{'sha256':digest(outside)}}})
            self.assertTrue(keep.exists());self.assertTrue(outside.exists())
            with self.assertRaises(ValueError):
                remove_verified_raw(tile,{'matrices':{'raw':{'sha256':'changed'}}})
            self.assertTrue(keep.exists())


if __name__=='__main__':unittest.main()
