"""Policy, additive-MLEM, adjoint and independent worker diagnostic contracts."""
import tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from ehe_common import *
from torch_active_operator import ActiveGeometry,ViewResponse,single_mlem
from single_checkpoint_mlem import single_mlem_checkpointed
from run_ehe_worker import FILES,validate

class Contracts(unittest.TestCase):
    def test_union_patch_is_one_boundary_fix(self):
        import hashlib
        patched=(HERE/'ehe_G4MultiUnion_11_1.cc').read_text()
        self.assertIn('G4VERSION_NUMBER == 1110',patched)
        upstream=patched.split('\n',2)[2].replace('for (std::size_t i = 0; i + 1 < size; ++i)',
                                               'for (std::size_t i = 0; i < size - 1; ++i)')
        self.assertEqual(hashlib.sha256(upstream.encode()).hexdigest(),
                         '98f517c04c37edaa29d9ac5747f1ebf41e545d20ef0cdacb9754942b0f5e529d')
    def test_overlay_entrypoint_and_sampler(self):
        from prepare_ehe_5e9 import overlay
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'ehe';overlay(p)
            self.assertEqual(read(p/'overlay_manifest.json')['base_source_sha256']['ehe_spect.cc'],
                             digest(ROOT/'Geant4Sim/Geant4Code_EHE/ehe_spect.cc'))
            self.assertIn('TransportTiming.json',(p/'ehe_spect.cc').read_text())
            self.assertEqual(digest(p/'src/PrimaryGeneratorAction.cc'),digest(ROOT/'Geant4Sim/Geant4Code/src/PrimaryGeneratorAction.cc'))
            self.assertIn('298.5',(p/'include/DetectorConstruction.hh').read_text())
            self.assertIn('GetPrimary218',(p/'src/RunAction.cc').read_text())
            self.assertIn('  src/ehe_G4MultiUnion_11_1.cc\n)',(p/'CMakeLists.txt').read_text())
    def test_policy(self):
        policy('formal',200,10);policy('validation',10,10)
        for x in (('formal',10000,50),('formal',200,50),('validation',200,10)):
            with self.assertRaises(ValueError):policy(*x)
    def test_actual_allocation(self):
        self.assertEqual(allocated_bytes('AllocTRES=cpu=6,mem=60000M,gres/gpu=1'),60000*1024**2)
        with self.assertRaises(ValueError):allocated_bytes('ReqTRES=mem=60000M')
    def test_gate(self):
        _,hold=physical_gate([130,130,130],[100,100,100],[20,1,1],[True,True,False]);self.assertEqual(hold.tolist(),[False,True,False])
    def test_mlem_checkpoint_both_modes(self):
        torch.manual_seed(41);g=ActiveGeometry(np.arange(7),np.ones(7),np.tile(np.arange(7)[:,None],(1,3)));r=ViewResponse(torch.rand(5,7)+.1,g,'none');s=r.sensitivity();y=torch.rand(5,3)*100
        x=torch.rand(7,1);z=torch.rand(5,1);v=r.matrix(1)
        self.assertLess(float(abs(((v@x)*z).sum()-(x*(v.T@z)).sum())),1e-5)
        for bg in (None,torch.rand(5,3)*2):
            old,history=single_mlem(r,y,s,20,10,additive_background=bg);saved=[]
            new,h=single_mlem_checkpointed(r,y,s,20,10,additive_background=bg,checkpoint_callback=lambda i,v:saved.append(i))
            self.assertTrue(torch.equal(old,new));self.assertTrue(torch.equal(history,h));self.assertEqual(saved,[10,20])
    def test_worker_tagged_closure(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);a=np.ones(2312,dtype=np.int64)
            for name in FILES:np.savetxt(p/name,a*(2 if '_from' not in name else 1),delimiter=',',fmt='%d',newline=',')
            # One CSV row must have exactly 2312 fields, no trailing comma.
            for name in FILES:(p/name).write_text(','.join(str(int(v)) for v in a*(2 if '_from' not in name else 1))+'\n')
            write(p/'TransportSummary.json',dict(primary_events=100,primary_counts=[29,71,0],detector_bins=2312));validate(p,100)
            (p/'CntStat_218_from440.csv').write_text(','.join(['0']*2312)+'\n')
            with self.assertRaises(ValueError):validate(p,100)
    def test_source_mass_and_rotation(self):
        from ehe_gpu_pipeline import source_grid
        truth={'x_mm':np.array([-1.5,1.5]),'y_mm':np.array([-1.5,1.5]),'activity_218_zyx':np.ones((40,2,2))}
        for v in range(20):self.assertAlmostEqual(float(source_grid(truth,218,v).sum()),1,places=7)
    def test_acceptance_detects_wrong_s_and_background(self):
        from verify_ehe import operator_closure
        from torch_active_operator import forward_project
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);result=root/'result';result.mkdir();responses=root/'responses'
            inverse=np.stack([np.roll(np.arange(8),v) for v in range(20)],axis=1)
            geometry=dict(active_indices=np.arange(7),ellipse_fraction=np.array([1]*7+[0]),inverse_rotation=inverse)
            g=ActiveGeometry(**{'active':geometry['active_indices'],'fraction':geometry['ellipse_fraction'],'inverse_rotation':inverse})
            rng=np.random.default_rng(71);raw=rng.uniform(.1,1,(8,2312)).astype('<f4');image=torch.linspace(1,2,7)[:,None]
            for name in RESPONSES:
                folder=responses/name;folder.mkdir(parents=True);raw.tofile(folder/'SysMat_polar')
                full=raw.sum(axis=1,dtype=np.float64);full.tofile(folder/'S_full.float64')
                np.mean(full[inverse[geometry['active_indices']]],axis=1).tofile(folder/'S_active.float64')
            image.numpy().tofile(result/f'Image_{CHANNELS[0]}_final.float32')
            background=forward_project(ViewResponse(torch.from_numpy(raw.T),g),image).numpy()
            np.save(result/'fixed_cross_background.npy',background)
            self.assertTrue(operator_closure(result,HERE,responses,geometry)['passed'])
            np.save(result/'fixed_cross_background.npy',background*1.1)
            with self.assertRaisesRegex(ValueError,'background'):operator_closure(result,HERE,responses,geometry)
            np.save(result/'fixed_cross_background.npy',background)
            folder=responses/'A218';s=np.fromfile(folder/'S_active.float64','<f8');(s*1.1).tofile(folder/'S_active.float64')
            with self.assertRaisesRegex(ValueError,'sensitivity'):operator_closure(result,HERE,responses,geometry)
    def test_macro_identity_allows_only_platform_newlines(self):
        from verify_ehe import source_macro_matches
        with tempfile.TemporaryDirectory() as t:
            original=Path(t)/'windows.mac';actual=Path(t)/'linux.mac'
            original.write_bytes(b'/source/angle 18\r\n/run/beamOn 25000000\r\n')
            actual.write_bytes(b'/source/angle 18\n/run/beamOn 25000000\n')
            self.assertTrue(source_macro_matches(original,actual))
            actual.write_bytes(b'/source/angle 36\n/run/beamOn 25000000\n')
            self.assertFalse(source_macro_matches(original,actual))

if __name__=='__main__':unittest.main()
