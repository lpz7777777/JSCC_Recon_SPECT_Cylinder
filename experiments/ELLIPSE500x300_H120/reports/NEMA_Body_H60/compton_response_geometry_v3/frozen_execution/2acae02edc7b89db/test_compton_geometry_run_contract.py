"""Regression must retain historical q3; gates and iteration bounds are strict."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from compton_geometry_run_contract import validate_run


class RunContractTests(unittest.TestCase):
    def test_regression_keeps_q3_and_switches_only_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); evidence = root/'spatial.json'
            evidence.write_text(json.dumps(dict(status='PASSED')))
            digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            study = dict(study='compton_response_geometry_v3', variant='R1_stable_point',
                group='ideal_first_scatter_v2', max_min_standardized_arm=3,
                quality_domain='full_circle_132040', iterations=2000, save_step=50,
                geometry_sha256='geometry', kernel_sha256='kernel',
                validation_evidence_sha256={'spatial.json': digest(evidence)})
            args = dict(regression=True, pilot=False, dry_run=False, iterations=50,
                save_step=50, dataset='NEMA_Body_H60', level='1e9', channels='compton-jscc',
                sensitivity=root/'Sensi_d', geometry_sha='geometry', kernel_sha='kernel',
                digest=digest, config_directory=root)
            self.assertEqual(validate_run(study, **args), ('legacy', 3.0))
            args.update(regression=False, pilot=True, iterations=10, save_step=10)
            self.assertEqual(validate_run(study, **args), ('stable_float64', 3.0))
            args.update(pilot=False, iterations=10000, save_step=50)
            with self.assertRaises(ValueError): validate_run(study, **args)
            args.update(iterations=2000); evidence.write_text(json.dumps(dict(status='HOLD')))
            with self.assertRaises(ValueError): validate_run(study, **args)
            study['validation_evidence_sha256']['spatial.json'] = digest(evidence)
            with self.assertRaises(ValueError): validate_run(study, **args)
            evidence.write_text(json.dumps(dict(status='PASSED')))
            study['validation_evidence_sha256']['spatial.json'] = digest(evidence)
            study['variant'] = 'R2_stable_overlap'
            with self.assertRaises(ValueError): validate_run(study, **args)


if __name__ == '__main__': unittest.main()
