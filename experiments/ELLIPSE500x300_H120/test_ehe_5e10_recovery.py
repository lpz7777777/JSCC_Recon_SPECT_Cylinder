"""Reject changed reused receipts, overlapping jobs and incomplete partitions."""
import copy, tempfile, unittest
from pathlib import Path
from ehe_common import read, write, digest
from ehe_5e10_recovered_transport import expected_worker_job
from ehe_5e10_workflow import REPORT


class RecoveryIdentity(unittest.TestCase):
    def setUp(self):
        stop=read(REPORT/'transport_stop_acceptance.json')
        self.recovery=dict(original_failed_job=15683333,original_state='FAILED',
            reused_workers=copy.deepcopy(stop['completed']),reused_count=13,missing_count=987)
        self.index=int(next(iter(stop['completed'])))

    def test_new_worker_requires_recovery_allocation(self):
        missing=next(i for i in range(1000) if str(i) not in self.recovery['reused_workers'])
        self.assertEqual(expected_worker_job({'transport_recovery':self.recovery},{'job':999},missing,Path('.')),'999')

    def test_reused_receipt_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);write(folder/'receipt.json',{'files':{},'changed':True})
            with self.assertRaisesRegex(ValueError,'receipt identity changed'):
                expected_worker_job({'transport_recovery':self.recovery},{'job':999},self.index,folder)

    def test_reuse_preserves_original_job_and_member_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);files={'a':'b'};write(folder/'receipt.json',{'files':files})
            self.recovery['reused_workers'][str(self.index)]={'receipt_sha256':digest(folder/'receipt.json'),'files':files,'allocation_job':'15683333'}
            self.assertEqual(expected_worker_job({'transport_recovery':self.recovery},{'job':999},self.index,folder),'15683333')
            self.recovery['reused_workers'][str(self.index)]['allocation_job']='999'
            with self.assertRaises(ValueError):expected_worker_job({'transport_recovery':self.recovery},{'job':999},self.index,folder)

    def test_failed_original_cannot_be_relabelled_as_recovery_success(self):
        with self.assertRaises(ValueError):expected_worker_job({'transport_recovery':self.recovery},{'job':15683333},0,Path('.'))
        self.recovery['original_state']='COMPLETED'
        with self.assertRaises(ValueError):expected_worker_job({'transport_recovery':self.recovery},{'job':999},0,Path('.'))

    def test_missing_or_invalid_partition_rejected(self):
        for change in ('missing','out_of_range'):
            r=copy.deepcopy(self.recovery)
            if change=='missing':r['missing_count']=986
            else:r['reused_workers']['1000']=r['reused_workers'].pop(str(self.index))
            with self.assertRaises(ValueError):expected_worker_job({'transport_recovery':r},{'job':999},0,Path('.'))


if __name__=='__main__':unittest.main()
