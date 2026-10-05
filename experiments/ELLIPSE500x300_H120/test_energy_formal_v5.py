"""Formal gates and checkpoint corruption/solver invariance safeguards."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from energy_formal_v5_contract import (load_contract,write_checkpoint,verify_checkpoints,CHANNELS,
    validate_transport_collection)
from verify_energy_formal_v5 import validate_authority,verify
from run_reconstruction import digest
from torch_active_operator import ActiveGeometry,ViewResponse,compton_and_joint_mlem


class FormalSafeguards(unittest.TestCase):
    def test_actual_collection_counts_energy_categories_not_workers(self):
        root=Path(__file__).parent
        path=root/'transport_collection.json'
        if not path.exists():path=root/'generated/compton_first_scatter_v2/recon_inputs/ideal/collections/NEMA_Body_H60_1e9.json'
        c=json.loads(path.read_text());receipt=validate_transport_collection(c)
        self.assertEqual(receipt['primary_by_energy'],[293821153,706178847,0])
        self.assertEqual(receipt['actual_primary_gamma'],1_000_000_000)
        self.assertEqual(receipt['workers'],200)
        for key,value in [('primary_counts',[5_000_000]*200),('primary_counts',[293821152,706178847,0]),
            ('seeds',c['seeds'][:-1]+[c['seeds'][0]]),('worker_indices',[0]*200),('views',list(range(1,20)))]:
            with self.assertRaises(ValueError):validate_transport_collection(dict(c,**{key:value}))

    def geometry(self):
        return ActiveGeometry([0,2],[1.,0.,1.],np.arange(3)[:,None])

    def fixture_contract(self,root):
        (root/'operator.py').write_text('original operator')
        files={'operator.py':digest(root/'operator.py')}
        prior=dict(files=files,input_sha256={'original':'identity'},factor_manifest_sha256={},
            whole_geometry_sha256='geometry',events_per_view=[91225])
        (root/'preflight_contract.json').write_text(json.dumps(prior))
        phases=[]
        for phase in ('regression','angular','continuous_energy'):
            folder=root/'preflight_evidence'/phase;folder.mkdir(parents=True)
            (folder/'verification.json').write_text(json.dumps(dict(passed=True)))
            (folder/'run_manifest.json').write_text('{}')
            phases.append(dict(phase=phase,verification_sha256=digest(folder/'verification.json'),
                run_manifest_sha256=digest(folder/'run_manifest.json')))
        (root/'preflight_summary.json').write_text(json.dumps(dict(status='PASSED',job=7,phases=phases)))
        cfg=dict(prior,study='compton_energy_probability_v5_formal',iterations=2000,save_step=50,
            accepted_events=91225,new_photons=0,models=['angular','continuous_energy'],
            channels=list(CHANNELS),preflight_job=7)
        files.update({p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file()})
        cfg['files']=files;p=root/'contract.json';p.write_text(json.dumps(cfg));return p,cfg

    def test_bounded_entry_refuses_arbitrary_iterations_and_changed_validated_operator(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);path,cfg=self.fixture_contract(root)
            load_contract(path,'formal',2000,50);load_contract(path,'validation',10,10)
            for mode,n,save in [('formal',10000,50),('formal',2000,10),('validation',50,50)]:
                with self.assertRaises(ValueError):load_contract(path,mode,n,save)
            (root/'operator.py').write_text('changed operator')
            with self.assertRaises(ValueError):load_contract(path,'formal',2000,50)

    def test_formal_cannot_start_without_pinned_actual_validation_authority(self):
        with self.assertRaises(ValueError):validate_authority(None,None,Path('contract.json'))

    def test_checkpoint_callback_does_not_change_original_mlem_images_or_histories(self):
        g=self.geometry();response=ViewResponse(torch.tensor([[.2,.3,.4],[.5,.1,.3]]),g)
        projection=torch.tensor([[8.],[11.]]);blocks=[[torch.tensor([[.3,.2],[.2,.4]])]]
        single=response.sensitivity();compton=torch.tensor([[.4],[.5]])
        baseline=compton_and_joint_mlem(response,projection,blocks,single,compton,100,50)
        with tempfile.TemporaryDirectory() as d:
            def save(i,hd,hj):write_checkpoint(Path(d),'angular',i,hd,hj,g,'contract')
            actual=compton_and_joint_mlem(response,projection,blocks,single,compton,100,50,checkpoint_callback=save)
            for a,b in zip(actual,baseline):
                for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=0,atol=0)
            self.assertTrue((Path(d)/'checkpoint_000050/checkpoint_manifest.json').exists())
            self.assertTrue((Path(d)/'checkpoint_000100/checkpoint_manifest.json').exists())

    def test_all_forty_snapshots_required_and_hash_corruption_rejected(self):
        g=self.geometry();hd=[];hj=[]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for frame in range(40):
                hd.append(torch.tensor([[frame+1.],[frame+2.]]));hj.append(hd[-1]*2)
                write_checkpoint(root,'angular',(frame+1)*50,hd,hj,g,'sha')
            history={CHANNELS[0]:torch.stack(hd).numpy().reshape(40,2),CHANNELS[1]:torch.stack(hj).numpy().reshape(40,2)}
            self.assertEqual(len(verify_checkpoints(root,history,np.array([0,2]),3,'angular','sha')),40)
            p=root/'checkpoint_001000'/f'Image_{CHANNELS[0]}_active.float32';p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):verify_checkpoints(root,history,np.array([0,2]),3,'angular','sha')

    def test_invalid_checkpoint_is_unpublished_and_existing_frame_cannot_be_overwritten(self):
        g=self.geometry()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaises(ValueError):write_checkpoint(root,'angular',50,[torch.tensor([[float('nan')],[1.]])],[torch.ones(2,1)],g,'sha')
            self.assertEqual(list(root.iterdir()),[])
            write_checkpoint(root,'angular',50,[torch.ones(2,1)],[torch.ones(2,1)],g,'sha')
            with self.assertRaises(ValueError):write_checkpoint(root,'angular',50,[torch.ones(2,1)],[torch.ones(2,1)],g,'sha')

    def test_strict_verifier_with_actual_pilot_fixture_is_not_new_imaging_evidence(self):
        root=Path(__file__).parent;contract=root/'contract.json'
        if not contract.exists():root=root/'generated/compton_energy_probability_v5/formal_payload';contract=root/'contract.json'
        if not contract.exists():self.skipTest('Frozen actual pilot fixture is required at deployment')
        cfg=json.loads(contract.read_text());active=np.load(root/'whole_geometry.npz')['active_indices']
        with tempfile.TemporaryDirectory() as d:
            result=Path(d)
            # This is a verifier protocol fixture, never a new numerical-path run.
            run=json.loads((root/'preflight_evidence/angular/run_manifest.json').read_text())
            run.update(study=cfg['study'],mode='validation',preflight_job=cfg['preflight_job'],
                actual_primary_gamma=1_000_000_000,workers=200,views=20,
                transport=validate_transport_collection(json.loads((root/'transport_collection.json').read_text())),
                contract_sha256=digest(contract),source_sha256=cfg['files']['run_energy_formal_v5.py'])
            (result/'run_manifest.json').write_text(json.dumps(run))
            for channel in CHANNELS:
                a=np.fromfile(root/f'preflight_evidence/angular/reference/Image_{channel}_active.float32','<f4')
                a.tofile(result/f'Image_{channel}_active.float32');a.tofile(result/f'Image_{channel}_history.float32')
                f=np.zeros(132040,dtype='<f4');f[active]=a;f.tofile(result/f'Image_{channel}_full.float32')
            proof=verify(result,contract,root/'preflight_allocation.txt','validation',write_receipt=False)
            self.assertTrue(proof['passed']);self.assertEqual(proof['outputs'][0]['pilot_frame10_relative_L2'],0.)
            self.assertFalse((result/'verification.json').exists())
            path=result/f'Image_{CHANNELS[0]}_active.float32';a=np.fromfile(path,'<f4');a[0]=float('nan');a.tofile(path)
            with self.assertRaises(ValueError):verify(result,contract,root/'preflight_allocation.txt','validation',write_receipt=False)


if __name__=='__main__':unittest.main()
