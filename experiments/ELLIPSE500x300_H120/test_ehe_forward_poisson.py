"""Scientific invariants for the independent expected-dose Poisson experiment."""
import unittest
import numpy as np
from ehe_common import HERE, read
from ehe_forward_poisson_data import dose_budget, sample_components
from ehe_gpu_pipeline import source_grid


class SyntheticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth=np.load(HERE/'generated/NEMA_Body_H60/truth_3mm.npz')
        cls.config=dict(expected_emitted_photons=5_000_000_000,gamma_yields={'218':.114,'440':.259},
            relative_activity_integral_mm3=read(HERE/'reports/NEMA_Body_H60/manifest.json')['relative_activity_integral_mm3'])

    def test_emitted_not_detected_dose(self):
        b=dose_budget(self.truth,self.config)
        self.assertAlmostEqual(b['expected_emitted_photons'],5e9,places=5)
        self.assertAlmostEqual(b['expected_primary_photons']['218']/5e9,.29380779868182727,places=14)
        for e in ('218','440'):
            self.assertAlmostEqual(b['density_gamma_per_mm3'][e]*b['integrals_mm3'][e],b['expected_primary_photons'][e],places=5)

    def test_all_true_source_mass_and_views(self):
        for e in (218,440):
            for view in range(20):
                source=source_grid(self.truth,e,view)
                self.assertEqual(source.shape,(40,85,85));self.assertTrue(np.all(source>=0))
                self.assertAlmostEqual(source.sum(),1,places=13)

    def test_noise_seeds_components_and_no_brightness_matching(self):
        names=('A218','A440','C440to218');means={n:np.full((2312,20),.31+i) for i,n in enumerate(names)}
        seeds={n:32100101+i for i,n in enumerate(names)}
        a,y=sample_components(means,seeds);b,z=sample_components(means,seeds)
        for n in names:self.assertTrue(np.array_equal(a[n],b[n]))
        self.assertTrue(np.array_equal(y[218],a['A218']+a['C440to218']))
        self.assertTrue(np.array_equal(y[440],a['A440']))
        self.assertNotEqual(int(y[218].sum()+y[440].sum()),5_000_000_000)
        changed={**seeds,'C440to218':32100104};c,_=sample_components(means,changed)
        self.assertFalse(np.array_equal(a['C440to218'],c['C440to218']))
        self.assertTrue(np.array_equal(a['A440'],c['A440']))

    def test_incomplete_or_invalid_mean_rejected(self):
        names=('A218','A440','C440to218');means={n:np.zeros((2312,20)) for n in names};seeds=dict.fromkeys(names,1)
        means['A218']=np.zeros((2311,20))
        with self.assertRaises(ValueError):sample_components(means,seeds)
        means['A218']=np.full((2312,20),-1.)
        with self.assertRaises(ValueError):sample_components(means,seeds)


if __name__=='__main__':unittest.main()
