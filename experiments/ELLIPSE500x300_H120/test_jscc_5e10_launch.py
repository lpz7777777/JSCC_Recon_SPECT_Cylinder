"""The scheduler-only repair must not authorize changed scientific inputs."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_files(root, files):
    for name, sha in files.items():
        if digest(root/name) != sha:
            raise ValueError('Repair payload changed')


class LaunchRepairIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        report = root/'reports'; report.mkdir()
        data = root/'generated'; data.mkdir()
        payload = data/'launch_control_repair_payload'; payload.mkdir()
        self.name = 'jscc_5e10_reconstruction_workflow.py'
        (root/self.name).write_bytes(b'new local scheduler controller')
        (payload/self.name).write_bytes((root/self.name).read_bytes())
        for name in ('science_preparation.json', 'kernel_freeze.json'):
            (report/name).write_bytes(b'{"original": true}')
        proof = dict(old_controller_sha256='old', new_controller_sha256=digest(root/self.name),
            sha256={self.name: digest(payload/self.name)},
            science_preparation_sha256=digest(report/'science_preparation.json'),
            kernel_freeze_sha256=digest(report/'kernel_freeze.json'))
        (report/'launch_control_repair_freeze.json').write_text(json.dumps(proof))
        source = Path(__file__).with_name(self.name)
        fn = next(x for x in ast.parse(source.read_text()).body
                  if isinstance(x, ast.FunctionDef) and x.name=='verify_prepared_controller_repair')
        env = dict(HERE=root, REPORT=report, DATA=data, read=read,
                   digest=digest, verify_files=verify_files)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), env)
        self.check = env['verify_prepared_controller_repair']
        self.root, self.report, self.payload = root, report, payload

    def test_exact_frozen_local_controller_repair_is_accepted(self):
        self.check(self.name, 'old')

    def test_scientific_source_change_cannot_use_launch_repair(self):
        with self.assertRaises(ValueError):
            self.check('run_jscc_5e10.py', 'old')

    def test_wrong_original_or_tampered_controller_is_rejected(self):
        with self.assertRaises(ValueError):
            self.check(self.name, 'another original')
        (self.root/self.name).write_bytes(b'unregistered controller')
        with self.assertRaises(ValueError):
            self.check(self.name, 'old')

    def test_changed_scientific_freeze_or_repair_payload_is_rejected(self):
        path = self.report/'kernel_freeze.json'; original = path.read_bytes()
        path.write_bytes(b'{"changed": true}')
        with self.assertRaises(ValueError):
            self.check(self.name, 'old')
        path.write_bytes(original)
        (self.payload/self.name).write_bytes(b'unregistered repair')
        with self.assertRaises(ValueError):
            self.check(self.name, 'old')


if __name__=='__main__':
    unittest.main()
