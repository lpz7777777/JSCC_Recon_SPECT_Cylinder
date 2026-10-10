"""New-dose identity and bounded-adaptation failures; no reconstruction surrogate."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ehe_common import HERE, RESPONSES, digest, read, write
from ehe_forward_poisson_5e10_data import STUDY, dose_budget, verify_counts, sample_components
from ehe_forward_poisson_5e10_workflow import (
    REPORT, DATA, DOSE, SEEDS, PREVIOUS_DATA, adapted_data_source)


class NewDoseIdentity(unittest.TestCase):
    def test_only_two_source_constants_change(self):
        old = (PREVIOUS_DATA / 'payload/ehe_forward_poisson_data.py').read_bytes()
        new = (HERE / 'ehe_forward_poisson_5e10_data.py').read_bytes()
        self.assertEqual(adapted_data_source(old), new)
        with self.assertRaises(ValueError):
            adapted_data_source(old + b"\nSTUDY = 'ehe_forward_poisson_5e9_200'")

    def test_budget_and_science_identity(self):
        f = read(REPORT / 'freeze.json')
        cfg = read(DATA / 'payload/config.json')
        budget = dose_budget(np.load(DATA / 'payload/truth_3mm.npz'), cfg)
        self.assertAlmostEqual(budget['expected_emitted_photons'] / DOSE, 1, places=14)
        self.assertAlmostEqual(budget['expected_primary_photons']['218'] / DOSE,
                               .29380779868182727, places=14)
        self.assertEqual(cfg['noise_seeds'], SEEDS)
        old = read(HERE / 'reports/NEMA_Body_H60/ehe_forward_poisson_5e9_200/freeze.json')
        for name in old['sha256']:
            if name not in ('config.json', 'ehe_forward_poisson_data.py'):
                self.assertEqual(f['sha256'][name], old['sha256'][name])

    def fixture(self, root, dose=DOSE, study=STUDY):
        release, counts = root / 'release', root / 'counts'
        release.mkdir(); counts.mkdir()
        cfg = read(DATA / 'payload/config.json')
        cfg['expected_emitted_photons'] = dose
        write(release / 'config.json', cfg)
        for name in ('truth_3mm.npz', 'whole_geometry.npz'):
            shutil.copy2(DATA / 'payload' / name, release / name)
        means = {n: np.full((2312, 20), .41 + i) for i, n in enumerate(RESPONSES)}
        sampled, projection = sample_components(means, SEEDS)
        for name in RESPONSES:
            np.save(counts / ('mean_' + name + '.npy'), means[name])
            np.save(counts / ('sampled_' + name + '.npy'), sampled[name])
        for energy in (218, 440):
            np.save(counts / ('projection_' + str(energy) + '.npy'), projection[energy])
        record = dict(study=study, data_kind='matrix_forward_plus_independent_Poisson',
                      passed=True, physical_calibration_claim=False, transport_performed=False,
                      views=20, bins=2312, config_sha256=digest(release / 'config.json'),
                      truth_sha256=digest(release / 'truth_3mm.npz'),
                      geometry_sha256=digest(release / 'whole_geometry.npz'),
                      factor_sha256=cfg['factor_sha256'], noise_seeds=SEEDS,
                      budget=dose_budget(np.load(release / 'truth_3mm.npz'), cfg),
                      window_counts={str(e):int(projection[e].sum()) for e in (218, 440)},
                      files={p.name:digest(p) for p in counts.glob('*.npy')})
        write(counts / 'collection.json', record)
        return release, counts

    def test_old_dose_or_study_cannot_supply_new_authority(self):
        for dose, study in ((5_000_000_000, STUDY), (DOSE, 'ehe_forward_poisson_5e9_200')):
            with tempfile.TemporaryDirectory() as folder:
                release, counts = self.fixture(Path(folder), dose, study)
                with self.assertRaises(ValueError):
                    verify_counts(counts, release)

    def test_seed_replay_and_archive_sha_reject_changed_component(self):
        with tempfile.TemporaryDirectory() as folder:
            release, counts = self.fixture(Path(folder))
            self.assertTrue(verify_counts(counts, release)['passed'])
            path = counts / 'sampled_C440to218.npy'
            changed = np.load(path); changed[0, 0] += 1; np.save(path, changed)
            with self.assertRaises(ValueError):
                verify_counts(counts, release)
            record = read(counts / 'collection.json')
            record['files'][path.name] = digest(path)
            write(counts / 'collection.json', record)
            with self.assertRaises(ValueError):
                verify_counts(counts, release)


if __name__ == '__main__':
    unittest.main()
