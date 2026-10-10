"""Protect parallel-trial isolation from the registered production launch."""
import unittest
from unittest.mock import patch
import jscc_5e10_4090_trial as trial
from jscc_5e10_reconstruction_workflow import launcher


class TrialIsolation(unittest.TestCase):
    def test_scientific_invocation_and_topology_are_unchanged(self):
        release = '/immutable/kernel'
        script = trial.trial_launcher(release)
        original = launcher('selection', release, 3600, 10800, 14220)
        self.assertEqual(script.split('timeout --signal=')[1].split('\necho ')[0],
                         original.split('timeout --signal=')[1].split('\necho ')[0])
        self.assertIn('--nnodes=8 --nproc_per_node=4', script)
        self.assertIn('/allocations/selection_4090_trial_', script)
        self.assertIn('/selection_4090_trial_', script)
        self.assertNotIn('/allocations/selection_', script.replace('/allocations/selection_4090_trial_', ''))

    def test_unknown_launch_layout_is_rejected(self):
        with patch.object(trial, 'launcher', return_value='modified launch'):
            with self.assertRaises(ValueError):
                trial.trial_launcher('/immutable/kernel')


if __name__ == '__main__':
    unittest.main()
