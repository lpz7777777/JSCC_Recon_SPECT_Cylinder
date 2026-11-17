"""CPU-only I/O regression; synthetic arrays are not physical response validation."""
import unittest
import numpy as np
from repair_matrix import response_name, compact


class RecoveryTests(unittest.TestCase):
    def test_cross_uses_scatter(self):
        self.assertEqual(response_name('C440to218', [1,1,1,0], -56), 'Scatter_SysMat_shift_0.000000_0.000000_-56.000000.sysmat')

    def test_main_uses_combined(self):
        for ch in ['A440','A218']:
            self.assertEqual(response_name(ch, [1,1,1,1], 0), 'SysMat_withScatter_shift_0.000000_0.000000_0.000000.sysmat')

    def test_flags_fail_closed(self):
        for ch, flags in [('C440to218',[1,1,1,1]), ('A440',[1,1,1,0]), ('unknown',[1,1,1,1])]:
            with self.assertRaises(ValueError):
                response_name(ch,flags,0)

    def test_compact_exact_with_tail(self):
        raw=np.arange(70*2*3*4,dtype=np.float32).reshape(70,2,3,4)/100
        selected=np.arange(0,70,2)
        scales=np.linspace(.9,1.3,len(selected))
        actual=np.zeros((35,4,3,4),dtype=np.float32)
        compact(raw,selected,scales,actual,2)
        expected=(raw[selected].astype(np.float64)*scales[:,None,None,None]*64).astype(np.float32)
        np.testing.assert_array_equal(actual[:,2:4],expected)
        self.assertFalse(np.any(actual[:,:2]))

    def test_invalid_values_rejected(self):
        for value in [np.nan,np.inf,-1.]:
            with self.assertRaises(ValueError):
                compact(np.full((1,2,1,1),value),np.array([0]),np.array([1.]),np.zeros((1,2,1,1)),0)


if __name__=='__main__':
    unittest.main()
