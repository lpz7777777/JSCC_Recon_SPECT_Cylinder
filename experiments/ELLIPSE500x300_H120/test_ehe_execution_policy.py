import json
from pathlib import Path
import tempfile
import unittest

from ehe_common import RESPONSES, digest
from ehe_execution_policy import physical_permission


class ContinuationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.physical = self.root / 'physical'; self.physical.mkdir()
        self.counts = self.root / 'counts'; self.counts.mkdir()
        self.responses = self.root / 'responses'; self.responses.mkdir()
        for name in RESPONSES:
            folder = self.responses / name; folder.mkdir()
            (folder / 'factor_manifest.json').write_text('{}')
        (self.counts / 'collection.json').write_text('{}')
        self.gate = dict(passed=False,hold_count=22,undetermined_bins=138720,
                         source_sha256='truth',files={'physical_audit.csv':'audit'})
        (self.physical / 'physical_gate.json').write_text(json.dumps(self.gate))
        self.release = dict(release_key='execution',producer_release_key='science',
                            sha256={'torch_active_operator.py':'operator','single_checkpoint_mlem.py':'mlem'})
        self.policy = dict(kind='explicit_human_original_method_continuation',study='ehe_spect_5e9_200',
                           authorized_modes=['validation','formal'],formal_iterations=200,
                           physical_calibration_passed=False,acknowledges_all_recorded_hold_diagnostics=True,
                           preserve_original_scientific_algorithm=True,producer_release_key='science',
                           physical_gate_sha256=digest(self.physical/'physical_gate.json'),hold_count=22,
                           undetermined_bins=138720,counts_sha256=digest(self.counts/'collection.json'),
                           factor_sha256={n:digest(self.responses/n/'factor_manifest.json') for n in RESPONSES},
                           source_sha256='truth',physical_audit_sha256='audit',baseline_helper_sha256=self.release['sha256'])
        self.path = self.root / 'authorization.json'

    def permission(self):
        self.path.write_text(json.dumps(self.policy))
        return physical_permission(self.physical,self.counts,self.responses,self.release,self.path)

    def test_authorization_does_not_claim_physical_pass(self):
        original=(self.physical/'physical_gate.json').read_bytes()
        value=self.permission()
        self.assertFalse(value['physical_calibration_passed'])
        self.assertTrue(value['execution_authorized_under_known_physics_mismatch'])
        self.assertEqual(value['original_hold_count'],22)
        self.assertEqual(original,(self.physical/'physical_gate.json').read_bytes())

    def test_hold_still_blocks_without_authorization(self):
        with self.assertRaises(ValueError):
            physical_permission(self.physical,self.counts,self.responses,self.release)

    def test_other_observations_are_not_authorized(self):
        (self.counts/'collection.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError): self.permission()

    def test_other_factors_are_not_authorized(self):
        (self.responses/'C440to218'/'factor_manifest.json').write_text('{"changed":true}')
        with self.assertRaises(ValueError): self.permission()

    def test_gate_cannot_be_relabelled_passed(self):
        self.gate['passed']=True
        (self.physical/'physical_gate.json').write_text(json.dumps(self.gate))
        with self.assertRaises(ValueError): self.permission()

    def test_scope_and_scientific_identity_must_match(self):
        for field,value in [('producer_release_key','other'),('formal_iterations',10000),
                            ('physical_calibration_passed',True),('hold_count',20),
                            ('source_sha256','other'),('physical_audit_sha256','other'),
                            ('baseline_helper_sha256',{}),('preserve_original_scientific_algorithm',False)]:
            with self.subTest(field=field):
                original=self.policy[field];self.policy[field]=value
                with self.assertRaises(ValueError): self.permission()
                self.policy[field]=original

    def test_original_reconstruction_body_remains_exact(self):
        from ehe_common import HERE,DATA,REPORT,read
        freeze=read(REPORT/'response_repair_freeze.json')
        original=DATA/freeze['payload_dir']/'run_ehe_reconstruction.py'
        def core(p):
            return p.read_text(encoding='utf-8').split('    out.mkdir',1)[1].split('    record=dict',1)[0]
        self.assertEqual(core(original),core(HERE/'run_ehe_reconstruction.py'))


if __name__ == '__main__': unittest.main()
