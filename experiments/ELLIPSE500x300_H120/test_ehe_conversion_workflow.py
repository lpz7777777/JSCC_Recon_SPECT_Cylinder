"""Bounded repair routing and immutable source evidence guards."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import ehe_conversion_workflow as controller
from ehe_common import digest, write
from ehe_5e9_workflow import base


class ConversionRoutingTests(unittest.TestCase):
    def binding(self, folder):
        folder = Path(folder)
        key = '1234567890abcdef'
        write(folder / 'source.txt', dict(actual='source job fully exited'))
        stop = dict(passed=True, explicit_human_stop=True, fully_exited=True, job=1672966,
                    files={'source.txt': digest(folder / 'source.txt')})
        write(folder / 'response_stop_acceptance.json', stop)
        b = dict(repair_key=key, producer_release_key='74e129c4460163c5',
                 source_stop_sha256=digest(folder / 'response_stop_acceptance.json'),
                 source_release=base('gpu') + '/releases/74e129c4460163c5',
                 source_responses=base('gpu') + '/responses',
                 repair_root=base('gpu') + '/conversion_releases/' + key,
                 probe_output=base('gpu') + '/conversion_probe_' + key,
                 response_output=base('gpu') + '/responses_conversion_' + key)
        write(folder / 'response_conversion_freeze.json', b)
        return b, stop

    def test_binding_routes_only_new_output_without_redeployment(self):
        with tempfile.TemporaryDirectory() as f, patch.object(controller, 'REPORT', Path(f)):
            b, _ = self.binding(f)
            self.assertEqual(controller.response_root(), b['response_output'])
            with patch('ehe_5e9_workflow.connection', side_effect=AssertionError('No redeployment')):
                self.assertEqual(controller.setup(), b)

    def test_changed_stop_proof_and_outside_output_refused(self):
        with tempfile.TemporaryDirectory() as f, patch.object(controller, 'REPORT', Path(f)):
            b, stop = self.binding(f)
            b['response_output'] = '/another_project/responses'
            write(Path(f) / 'response_conversion_freeze.json', b)
            with self.assertRaises(ValueError):
                controller.repair_binding()
            b, stop = self.binding(f)
            stop['fully_exited'] = False
            write(Path(f) / 'response_stop_acceptance.json', stop)
            with self.assertRaises(ValueError):
                controller.repair_binding()

    def test_conversion_scripts_cannot_launch_physics_or_simulation(self):
        with tempfile.TemporaryDirectory() as f:
            b, _ = self.binding(f)
            for stage in ('probe', 'complete'):
                script = controller.conversion_script(b, stage)
                self.assertIn('ehe_conversion_io.py ' + stage, script)
                self.assertNotIn('PEGen', script)
                self.assertNotIn('ScatterGen', script)
                self.assertNotIn('run_ehe_worker', script)
                self.assertIn('PYTHONDONTWRITEBYTECODE=1', script)


if __name__ == '__main__':
    unittest.main()
