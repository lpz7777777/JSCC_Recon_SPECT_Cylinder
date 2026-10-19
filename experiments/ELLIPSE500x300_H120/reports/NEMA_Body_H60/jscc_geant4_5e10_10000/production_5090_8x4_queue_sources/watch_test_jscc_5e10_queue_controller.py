"""Exercise actual duplicate-submission and transient read-only timeout guards."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import jscc_5e10_5090_production_queue as queue
from jscc_5e10_common import write, read


class QueueControllerTests(unittest.TestCase):
    def test_registered_job_and_unresolved_intent_never_submit_again(self):
        with tempfile.TemporaryDirectory() as d, patch.object(queue,'connection') as connect:
            root=Path(d)
            with patch.object(queue,'REPORT',root):
                write(root/'formal_job.json',dict(job=123))
                self.assertEqual(queue.submit_script('formal','', '', 1, [], True),dict(job=123))
                write(root/'validation_submission_intent.json',dict(resolved=False))
                with self.assertRaisesRegex(RuntimeError,'Unresolved'):
                    queue.submit_script('validation','', '',1,[],False)
                connect.assert_not_called()

    def test_read_only_timeout_does_not_fetch_or_release(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write(root/'formal_job.json',dict(job=123))
            with patch.object(queue,'REPORT',root), patch.object(queue,'connection',side_effect=TimeoutError('scheduler timeout')), patch.object(queue.workflow,'fetch_stage') as fetch:
                self.assertFalse(queue.advance_release())
                fetch.assert_not_called()
                self.assertTrue(read(root/(queue.PREFIX+'_poll_error.json'))['formal_remains_held'])
                self.assertFalse((root/'formal_authority.json').exists())


if __name__=='__main__':unittest.main()
