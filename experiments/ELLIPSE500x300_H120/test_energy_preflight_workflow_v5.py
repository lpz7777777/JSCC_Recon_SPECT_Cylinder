"""Recovery safeguards for terminal jobs and persisted verification receipts."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import energy_preflight_v5_workflow as workflow


class RecoverySafeguards(unittest.TestCase):
    def test_completed_job_absent_from_queue_still_reads_accounting(self):
        with tempfile.TemporaryDirectory() as temp:
            report=Path(temp)
            (report/'preflight_job.json').write_text(json.dumps(dict(job=123)))
            calls=[]
            def command(client,text):
                calls.append(text)
                if text.startswith('squeue'):return ''
                if text.startswith('sacct'):return '123|COMPLETED|0:0|1024K|mem=10G,node=4'
                return ''
            with patch.object(workflow,'REPORT',report),patch.object(workflow,'connect',return_value=contextlib.nullcontext(None)),patch.object(workflow,'command',side_effect=command):
                with contextlib.redirect_stdout(io.StringIO()) as output:workflow.status()
            self.assertIn('COMPLETED',output.getvalue())
            self.assertTrue(any(x.startswith('sacct') for x in calls))
            self.assertFalse(any('squeue -j' in x for x in calls))

    def test_repair_refuses_active_attempt_without_archiving_registration(self):
        with tempfile.TemporaryDirectory() as temp:
            report=Path(temp)
            (report/'preflight_job.json').write_text(json.dumps(dict(job=123,release='/frozen')))
            with patch.object(workflow,'REPORT',report),patch.object(workflow,'connect',return_value=contextlib.nullcontext(None)),patch.object(workflow,'command',return_value='123 Energy_v5_preflight RUNNING'):
                with self.assertRaises(ValueError):workflow.repair()
            self.assertTrue((report/'preflight_job.json').exists())
            self.assertFalse((report/'preflight_attempts').exists())

    def test_actual_slurm_peak_requires_measured_units(self):
        self.assertEqual(workflow.peak_rss('123.1|COMPLETED|0:0|1024K|mem=10G'),1024**2)
        with self.assertRaises(ValueError):workflow.peak_rss('123|COMPLETED|0:0||mem=10G')


if __name__=='__main__':unittest.main()
