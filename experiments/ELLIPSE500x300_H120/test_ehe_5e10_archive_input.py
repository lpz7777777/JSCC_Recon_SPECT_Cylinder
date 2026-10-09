"""Storage failures must stop before original science; accepted bytes remain exact."""
import argparse,os,tarfile,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
import ehe_5e10_archive_input as bootstrap
from ehe_common import write,read,digest,hashes


class ArchiveInput(unittest.TestCase):
    def test_shared_nested_mount_and_ram_rejected(self):
        mounts='/dev/root / ext4 rw 0 0\nserver:/data /tmp/shared nfs rw 0 0\n'
        with self.assertRaises(ValueError):bootstrap.mount_identity('/tmp/shared/data',mounts)
        with self.assertRaises(ValueError):bootstrap.mount_identity('/tmp','tmpfs / tmpfs rw 0 0')
        self.assertEqual(bootstrap.mount_identity('/tmp/local',mounts)['filesystem'],'ext4')

    def fixture(self,folder,changed=False):
        release=folder/'release';release.mkdir();counts=folder/'original';counts.mkdir()
        write(counts/'collection.json',dict(passed=True,full_worker_marker='exact'))
        (counts/'worker.dat').write_bytes(b'original observation bytes')
        bundle=folder/'counts.tar.gz'
        with tarfile.open(bundle,'w:gz') as t:
            for p in counts.iterdir():t.add(p,arcname=p.name)
        program=release/'run_ehe_5e10_reconstruction.py'
        program.write_text("import sys\nfrom pathlib import Path\np=Path(sys.argv[sys.argv.index('--counts')+1])\nassert (p/'worker.dat').read_bytes()==b'original observation bytes'\n")
        write(release/'config.json',dict(archive_storage=dict(archive_path=str(bundle),archive_bytes=bundle.stat().st_size,
            archive_sha256='0'*64 if changed else digest(bundle),collection_sha256=digest(counts/'collection.json'),disk_allocation_bytes=4096)))
        expected=hashes(release);expected['ehe_5e10_archive_input.py']=digest(Path(bootstrap.__file__))
        (release/'ehe_5e10_archive_input.py').write_bytes(Path(bootstrap.__file__).read_bytes())
        write(release/'release_manifest.json',dict(release_key='fixture',sha256=expected))
        return argparse.Namespace(release=release,program=program,receipt=folder/'receipt.json',arguments=['--','--release',str(release)])

    def test_exact_bytes_and_target_success_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);a=self.fixture(folder)
            with patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(bootstrap,'choose_scratch',return_value=(folder,dict(free_bytes=10**9))):
                previous=sys.argv[:]
                try:bootstrap.launch(a)
                finally:sys.argv=previous
            proof=read(a.receipt);self.assertTrue(proof['passed']);self.assertEqual(proof['status'],'complete')
            self.assertFalse(Path(proof['temporary_counts']).exists())

    def test_corrupt_archive_stops_before_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);a=self.fixture(folder,changed=True)
            with patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(bootstrap,'choose_scratch',return_value=(folder,dict(free_bytes=10**9))):
                with self.assertRaisesRegex(ValueError,'SHA differs'):bootstrap.launch(a)
            self.assertFalse(read(a.receipt)['passed']);self.assertEqual(read(a.receipt)['status'],'failed')

    def test_full_input_space_and_override_fail_before_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=self.fixture(Path(tmp))
            with patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(bootstrap,'choose_scratch',side_effect=RuntimeError('No sufficiently large node-local disk')):
                with self.assertRaisesRegex(RuntimeError,'node-local disk'):bootstrap.launch(a)
            self.assertFalse(a.receipt.exists())
            a.arguments+=['--counts','wrong']
            with patch.dict(os.environ,SLURM_JOB_ID='123'):
                with self.assertRaisesRegex(ValueError,'override'):bootstrap.launch(a)


if __name__=='__main__':unittest.main()
