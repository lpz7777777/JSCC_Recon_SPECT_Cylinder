"""Completion must retain producer failures and bound diagnostic exemptions."""
import unittest
from ehe_slurm_status import stage_completed


class StatusGuard(unittest.TestCase):
    job = 1672966
    auxiliary = {'1672966.102': ['FAILED', '1:0']}

    def test_running_job_remains_running_after_accepted_diagnostic_failure(self):
        text = '1672966|RUNNING|0:0||23:26:29|mem=94500M\n1672966.102|FAILED|1:0|48K|00:00:00|mem=94500M'
        self.assertFalse(stage_completed(text, self.job, self.auxiliary))
        with self.assertRaises(RuntimeError):
            stage_completed(text, self.job)

    def test_completed_job_requires_successful_producer_steps(self):
        text = '1672966|COMPLETED|0:0||24:00:00|mem=94500M\n1672966.batch|FAILED|1:0|10M|24:00:00|mem=94500M'
        with self.assertRaises(RuntimeError):
            stage_completed(text, self.job, self.auxiliary)

    def test_exact_diagnostic_exit_identity(self):
        text = '1672966|RUNNING|0:0||23:26:29|mem=94500M\n1672966.102|OUT_OF_MEMORY|0:125|48K|00:00:00|mem=94500M'
        with self.assertRaises(RuntimeError):
            stage_completed(text, self.job, self.auxiliary)

    def test_no_new_or_unregistered_failure_is_exempt(self):
        text = '1672966|COMPLETED|0:0||24:00:00|mem=94500M\n1672966.104|TIMEOUT|0:15|48K|00:01:00|mem=94500M'
        with self.assertRaises(RuntimeError):
            stage_completed(text, self.job, self.auxiliary)

    def test_no_parent_batch_other_job_or_wildcard_exemption(self):
        for step in ('1672966', '1672966.batch', '1672966.extern', '1672967.102', '1672966.*'):
            with self.assertRaises(ValueError):
                stage_completed('', self.job, {step: ['FAILED', '1:0']})

    def test_all_200_array_workers_must_exit_successfully(self):
        text = '\n'.join(f'15633840_{i}|COMPLETED|0:0|1M|01:00:00|cpu=1' for i in range(200))
        self.assertTrue(stage_completed(text, 15633840))
        self.assertFalse(stage_completed('\n'.join(text.splitlines()[:-1]), 15633840))
        with self.assertRaises(RuntimeError):
            stage_completed(text.replace('15633840_199|COMPLETED|0:0', '15633840_199|FAILED|1:0'), 15633840)

    def test_successful_parent_with_accepted_auxiliary_failure(self):
        text = '1672966|COMPLETED|0:0||24:00:00|mem=94500M\n1672966.batch|COMPLETED|0:0|10M|24:00:00|mem=94500M\n1672966.102|FAILED|1:0|48K|00:00:00|mem=94500M'
        self.assertTrue(stage_completed(text, self.job, self.auxiliary))


if __name__ == '__main__':
    unittest.main()
