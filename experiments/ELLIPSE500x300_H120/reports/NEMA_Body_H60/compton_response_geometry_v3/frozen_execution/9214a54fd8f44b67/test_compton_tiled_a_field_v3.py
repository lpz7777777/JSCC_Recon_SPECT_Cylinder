import unittest
import numpy as np
from plan_compton_a_tiles_v3 import tile_spec
from generate_compton_a_guard import axes
from compton_tiled_a_field import CompactCartesianTiles


class TiledFieldTests(unittest.TestCase):
    def setUp(self):
        self.specs=[tile_spec(21,41,5),tile_spec(21,42,5)];self.arrays=[]
        for s in self.specs:
            x,y,z=axes(s);zz,yy,xx=np.meshgrid(z,y,x,indexing='ij')
            self.arrays.append(np.stack((500+xx+2*yy+3*zz,800+2*xx+yy+zz)).astype('<f4'))

    def test_affine_response_calibration_once_and_order_independent(self):
        f=CompactCartesianTiles(self.specs,self.arrays,[2.,.5])
        p=np.array([[1.123,245.999,3.78],[-3.61,246.001,6.3],[0,246,6]])
        np.testing.assert_allclose(f.evaluate(0,f.cache(p)),2*(500+p[:,0]+2*p[:,1]+3*p[:,2]),atol=1e-11,rtol=0)
        g=CompactCartesianTiles(self.specs[::-1],self.arrays[::-1],[2.,.5])
        np.testing.assert_array_equal(f.evaluate(1,f.cache(p)),g.evaluate(1,g.cache(p)))

    def test_shared_vertex_has_one_owner_despite_raw_roundoff(self):
        # Different duplicate values do not cause a seam or average the data.
        self.arrays[1][:,:,0,:]+=1
        f=CompactCartesianTiles(self.specs,self.arrays,[1.,1.])
        points=np.array([[0,246-1e-9,6],[0,246,6],[0,246+1e-9,6]])
        a=f.evaluate(0,f.cache(points))
        self.assertLess(abs(a[0]-a[1]),1e-7);self.assertLess(abs(a[2]-a[1]),1e-7)
        self.assertEqual(a[1],500+492+18)

    def test_missing_support_and_production_fail_closed(self):
        f=CompactCartesianTiles(self.specs,self.arrays,[1.,1.])
        with self.assertRaises(ValueError):f.cache(np.array([[20,240,6]]))
        with self.assertRaises(ValueError):f.cache(np.array([[0,240,61]]))
        with self.assertRaises(ValueError):f.assert_production_ready()


if __name__=='__main__':unittest.main()
