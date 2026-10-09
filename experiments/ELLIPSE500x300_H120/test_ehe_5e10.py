"""Small failure-oriented tests for independent dose, authority and safe fetching."""
import copy, io, tarfile, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from ehe_common import read, digest, write, policy
from ehe_5e10_transport import registry_identity, extract, TOTAL, WORKERS, PER_VIEW, PER_WORKER, SEED_BASE
from ehe_5e10_workflow import DATA, REPORT
from ehe_slurm_status import stage_completed


class AcquisitionContract(unittest.TestCase):
    def setUp(self):
        self.registry=read(DATA/'simulation/jobs.json')
        self.config=read(DATA/'transport_payload/config.json')

    def verify_registry(self,registry):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);write(folder/'jobs.json',registry)
            cfg={**self.config,'simulation_manifest_sha256':digest(folder/'jobs.json')}
            return registry_identity(folder,cfg)

    def test_full_partition_and_disjoint_seeds(self):
        r=self.verify_registry(self.registry)
        self.assertEqual(sum(v['photons'] for v in r['jobs']),TOTAL)
        self.assertEqual(len({v['seed'] for v in r['jobs']}),WORKERS)
        self.assertTrue(all(sum(v['view']==i for v in r['jobs'])==PER_VIEW for i in range(1,21)))
        self.assertTrue({v['seed'] for v in r['jobs']}.isdisjoint(range(31100101,31100301)))

    def test_repeated_seed_cannot_reuse_old_observation(self):
        r=copy.deepcopy(self.registry);r['jobs'][17]['seed']=r['jobs'][16]['seed']
        with self.assertRaises(ValueError):self.verify_registry(r)

    def test_lost_view_or_changed_worker_dose_rejected(self):
        for field,value in (('view',1),('photons',25000000)):
            r=copy.deepcopy(self.registry);r['jobs'][99][field]=value
            with self.assertRaises(ValueError):self.verify_registry(r)

    def test_all_macros_only_changed_requested_dose(self):
        old=DATA.parent/'ehe_spect_5e9_200/simulation'
        for row in self.registry['macros']:
            original=(old/row['path']).read_bytes()
            current=(DATA/'simulation'/row['path']).read_bytes()
            self.assertEqual(current,original.replace(b'/run/beamOn 25000000',b'/run/beamOn 50000000'))
            self.assertNotIn(b'/gps/ang',current)

    def test_running_worker_source_bytes_remain_frozen(self):
        f=read(REPORT/'transport_freeze.json')
        self.assertEqual(digest(DATA/f['payload_dir']/'ehe_5e10_transport.py'),f['sha256']['ehe_5e10_transport.py'])
        self.assertEqual(digest(DATA.parent.parent/'ehe_5e10_transport.py'),f['sha256']['ehe_5e10_transport.py'])

    def test_original_operator_and_solver_helpers_retained(self):
        f=read(REPORT/'freeze.json');payload=DATA/f['payload_dir']
        old=read(REPORT.parent/'ehe_spect_5e9_200/reconstruction_execution_freeze.json')
        for name in ('torch_active_operator.py','single_checkpoint_mlem.py','whole_geometry.npz','truth_3mm.npz'):
            self.assertEqual(digest(payload/name),old['sha256'][name])

    def test_only_original_iteration_and_saving_contract(self):
        policy('validation',10,10);policy('formal',200,10)
        for args in (('formal',10000,10),('formal',200,1),('validation',1,1)):
            with self.assertRaises(ValueError):policy(*args)

    def test_scientific_failure_not_a_successful_stage(self):
        with self.assertRaises(RuntimeError):stage_completed('1|FAILED|1:0\n1.batch|FAILED|1:0\n',1)
        self.assertFalse(stage_completed('1|RUNNING|0:0\n',1))

    def test_archive_traversal_and_links_rejected(self):
        for name,link in (('../outside',False),('worker/data',True)):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);archive=root/'bad.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    member=tarfile.TarInfo(name)
                    if link:member.type=tarfile.SYMTYPE;member.linkname='../outside';tar.addfile(member)
                    else:member.size=1;tar.addfile(member,io.BytesIO(b'x'))
                with self.assertRaises(ValueError):extract(archive,root/'counts')
                self.assertFalse((root/'outside').exists())

    def test_archive_partial_output_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);counts=root/'counts';counts.mkdir();(counts/'evidence').write_bytes(b'keep')
            with self.assertRaises(FileExistsError):extract(root/'absent.tar.gz',counts)
            self.assertEqual((counts/'evidence').read_bytes(),b'keep')


if __name__=='__main__':unittest.main()
