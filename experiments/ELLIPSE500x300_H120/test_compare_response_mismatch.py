"""Analysis smoke test: identical real histories must produce zero differences.

The synthetic verification/result lives in an ignored temporary directory. It is
not a cut3 reconstruction and must never be published as an experiment result.
"""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_response_mismatch import digest


class MatchedComparisonTest(unittest.TestCase):
    def test_identical_history_and_fixed_scale(self):
        baseline = HERE / 'generated/RemoteResults/NEMA_Body_H60_5e9_1644876'
        with tempfile.TemporaryDirectory(prefix='analysis_fixture_', dir=HERE / 'generated') as directory:
            root = Path(directory)
            fixture = root / 'identical_baseline_fixture'
            fixture.mkdir()
            channels = ('440_ComptonOnly', '440_SinglePlusCompton')
            outputs = []
            for channel in channels:
                name = f'Image_{channel}_history.float32'
                os.link(baseline / name, fixture / name)
                outputs.append({'channel': channel, 'sha256': {'history': digest(baseline / name)}})
            (fixture / 'run_manifest.json').write_bytes((baseline / 'run_manifest.json').read_bytes())
            verification = {'passed': True, 'mode': 'formal', 'outputs': outputs,
                            'run_manifest_sha256': digest(fixture / 'run_manifest.json'),
                            'config_sha256': digest(HERE / 'response_mismatch_cut3_v1.json')}
            (fixture / 'verification.json').write_text(json.dumps(verification))
            output = root / 'fixture_figures'
            result = subprocess.run([sys.executable, str(HERE / 'compare_response_mismatch.py'),
                                     '--result', str(fixture), '--output', str(output)],
                                    capture_output=True, text=True, timeout=240)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / 'comparison.json').read_text())
            for verdict in report['verdict'].values():
                self.assertFalse(verdict['extreme_spike_improved'])
                self.assertEqual(set(verdict['reductions'].values()), {0.0})
                self.assertEqual({c['crc_change'] for c in verdict['crc_costs']}, {0.0})
            for name, expected_rows in [('native_iteration_metrics.csv', 800),
                                        ('sphere_iteration_metrics.csv', 2400)]:
                with (output / name).open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), expected_rows)
                old = [{k: v for k, v in r.items() if k != 'group'} for r in rows if r['group'] == 'baseline']
                new = [{k: v for k, v in r.items() if k != 'group'} for r in rows if r['group'] == 'cut3']
                self.assertEqual(old, new)
            self.assertIn('central 72 mm', report['mip'])
            for name in ('iterations_center.png', 'iterations_mip72.png', 'final_multiplanar.png',
                         'spike_noise_leakage_curves.png', 'crc_cnr_curves.png'):
                self.assertGreater((output / name).stat().st_size, 1000)
            # A read-only 2000 snapshot is an observation, never a formal verdict.
            interim = root / 'interim_fixture'; interim.mkdir()
            for channel in channels:
                name = f'Image_{channel}_history.float32'
                with (baseline / name).open('rb') as stream:
                    (interim / name).write_bytes(stream.read(40*82040*4))
            (interim / 'checkpoint_manifest.json').write_bytes((baseline / 'run_manifest.json').read_bytes())
            record = dict(verification, mode='interim',
                          snapshot_manifest_sha256=digest(interim / 'checkpoint_manifest.json'),
                          outputs=[{'channel': c, 'sha256': {'history': digest(interim / f'Image_{c}_history.float32')}} for c in channels])
            (interim / 'verification.json').write_text(json.dumps(record))
            interim_output = root / 'interim_figures'
            result = subprocess.run([sys.executable, str(HERE / 'compare_response_mismatch.py'),
                                     '--result', str(interim), '--output', str(interim_output),
                                     '--through-iteration', '2000'], capture_output=True, text=True, timeout=240)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((interim_output / 'comparison.json').read_text())
            self.assertTrue(report['interim_only'])
            for verdict in report['verdict'].values():
                self.assertIsNone(verdict['extreme_spike_improved'])
                self.assertEqual(set(verdict['reductions'].values()), {0.0})


if __name__ == '__main__':
    unittest.main()
