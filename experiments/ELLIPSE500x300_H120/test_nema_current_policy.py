"""Boundary, shared-background and future separate-output regression checks."""
import json
from pathlib import Path
import unittest
import numpy as np
from nema_roi_policy import build_masks, measure
from reconstruction_output_policy import (EHE_CHANNELS, JSCC_CHANNELS, POLICY_ID,
    require_separate_channels, accepted_channels, RETIRED_SUM_CHANNELS)

H = Path(__file__).resolve().parent


class CurrentPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth = np.load(H/'generated/NEMA_Body_H60/truth_3mm.npz')
        cls.meta = json.loads((H/'reports/NEMA_Body_H60/manifest.json').read_text())
        cls.config = json.loads((H/'nema_body_h60_config.json').read_text())
        cls.masks = build_masks(cls.truth,cls.meta,cls.config)

    def test_registered_counts_and_nonempty_smallest_sphere(self):
        self.assertEqual({d:int(m.sum()) for d,m in self.masks['spheres'].items()},
            {10:8,13:20,17:60,22:136,28:312,37:772})
        self.assertEqual(int(self.masks['background'].sum()),97604)

    def test_every_sphere_center_is_at_least_half_pitch_inside(self):
        z,y,x=np.meshgrid(self.truth['z_mm'],self.truth['y_mm'],self.truth['x_mm'],indexing='ij')
        for s in self.meta['spheres']:
            cx,cy,cz=s['center_mm'];d=int(s['diameter_mm']);m=self.masks['spheres'][d]
            distance=np.sqrt((x[m]-cx)**2+(y[m]-cy)**2+(z[m]-cz)**2)
            self.assertTrue(np.all(distance<=d/2-1.5))

    def test_background_voxel_cannot_touch_any_sphere_or_lung(self):
        z,y,x=np.meshgrid(self.truth['z_mm'],self.truth['y_mm'],self.truth['x_mm'],indexing='ij')
        m=self.masks['background'];x,y,z=x[m],y[m],z[m]
        for s in self.meta['spheres']:
            c=s['center_mm'];closest2=sum(np.maximum(np.abs(v-p)-1.5,0)**2 for v,p in zip((x,y,z),c))
            self.assertTrue(np.all(closest2>(s['diameter_mm']/2)**2))
        self.assertTrue(np.all(np.hypot(np.maximum(abs(x)-1.5,0),np.maximum(abs(y)-1.5,0))>25.5))
        self.assertTrue(np.all(abs(z)+1.5<30))
        for energy in (218,440):self.assertTrue(np.all(self.truth[f'activity_{energy}_zyx'][m]==1))

    def test_constant_image_has_zero_hot_crc_and_undefined_cnr(self):
        base,rows=measure(np.ones(self.masks['background'].shape),self.masks,self.meta,218)
        self.assertEqual(base['background_mean'],1);self.assertEqual(base['background_std'],0)
        for r in rows:self.assertEqual(r['crc'],0);self.assertIsNone(r['cnr'])

    def test_analytic_hot_and_cold_contrast_and_common_noise_denominator(self):
        image=np.ones(self.masks['background'].shape)
        # A deterministic background gradient supplies nonzero spatial std.
        values=np.linspace(.5,1.5,int(self.masks['background'].sum()))
        image[self.masks['background']]=values
        for s in self.meta['spheres']:image[self.masks['spheres'][int(s['diameter_mm'])]]=10 if s['hot_energy_keV']==218 else 0
        base,rows=measure(image,self.masks,self.meta,218)
        self.assertAlmostEqual(base['background_std'],float(values.std(ddof=1)))
        for r in rows:
            self.assertAlmostEqual(r['crc'],1)
            self.assertEqual(r['background_mean'],base['background_mean'])
            self.assertEqual(r['background_std'],base['background_std'])
            self.assertAlmostEqual(r['cnr'],(9 if r['region_kind']=='hot' else -1)/base['background_std'])

    def test_separate_channels_and_440_joint_retained(self):
        self.assertEqual(len(EHE_CHANNELS),2);self.assertEqual(len(JSCC_CHANNELS),4)
        self.assertIn('440_SinglePlusCompton',JSCC_CHANNELS)
        self.assertFalse(set(RETIRED_SUM_CHANNELS)&set(JSCC_CHANNELS))
        for system,channels in (('EHE',EHE_CHANNELS),('JSCC',JSCC_CHANNELS)):
            self.assertEqual(require_separate_channels(dict(output_policy=POLICY_ID,output_channels=list(channels)),system),channels)

    def test_old_contract_cannot_start_a_new_solve(self):
        with self.assertRaises(ValueError):require_separate_channels({})
        with self.assertRaises(ValueError):require_separate_channels(dict(output_policy=POLICY_ID,output_channels=list(EHE_CHANNELS)+[RETIRED_SUM_CHANNELS[0]]))

    def test_acceptance_remains_bound_to_actual_output_scope(self):
        config=dict(output_policy=POLICY_ID,output_channels=list(EHE_CHANNELS))
        run=dict(output_policy=POLICY_ID,channels=list(EHE_CHANNELS))
        self.assertEqual(accepted_channels(run,config,EHE_CHANNELS+RETIRED_SUM_CHANNELS[:1]),EHE_CHANNELS)
        with self.assertRaises(ValueError):accepted_channels(dict(channels=list(EHE_CHANNELS)),config,EHE_CHANNELS)
        legacy=EHE_CHANNELS+RETIRED_SUM_CHANNELS[:1]
        self.assertEqual(accepted_channels(dict(channels=list(legacy)),{},legacy),legacy)


if __name__=='__main__':unittest.main()
