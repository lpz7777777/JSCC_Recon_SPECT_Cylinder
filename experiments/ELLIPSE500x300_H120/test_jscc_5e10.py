"""Scientific branch equivalence and independent dose/topology safeguards."""
import ast, copy, tempfile, unittest, shlex
from pathlib import Path
import numpy as np
import torch
from jscc_5e10_common import *
from jscc_5e10_compton_mlem import compton_mlem
from jscc_5e10_contract import verify_topology,write_checkpoint,verify_checkpoints
from torch_active_operator import ActiveGeometry,ViewResponse,single_mlem,compton_and_joint_mlem
from single_checkpoint_mlem import single_mlem_checkpointed

class JSCC5e10Tests(unittest.TestCase):
    def geometry(self):
        return ActiveGeometry([0,2],[1.,0.,1.],np.tile(np.arange(3)[:,None],(1,20)))

    def test_standalone_compton_matches_original_branch_and_all_saved_iterations(self):
        torch.manual_seed(34);g=self.geometry()
        response=ViewResponse(torch.tensor([[.2,.3,.4],[.5,.1,.3]]),g)
        projection=torch.arange(40,dtype=torch.float32).reshape(2,20)+1
        blocks=[[torch.rand(3,2)+.01,torch.rand(2,2)+.01] for _ in range(20)]
        sensitivity=torch.tensor([[2.],[3.]])
        reference,_=compton_and_joint_mlem(response,projection,blocks,response.sensitivity(),sensitivity,100,50)
        checkpoints=[]
        actual=compton_mlem(blocks,sensitivity,100,50,checkpoint_callback=lambda i,h:checkpoints.append((i,h[-1].clone())))
        for x,y in zip(actual,reference):torch.testing.assert_close(x,y,rtol=0,atol=0)
        self.assertEqual([i for i,h in checkpoints],[50,100]);self.assertTrue(torch.equal(checkpoints[-1][1],actual[1][-1]))

    def test_single_and_fixed_background_mlem_match_original_exactly(self):
        g=self.geometry();response=ViewResponse(torch.tensor([[.2,.3,.4],[.5,.1,.3]]),g)
        projection=torch.arange(40,dtype=torch.float32).reshape(2,20)+1
        for background in (None,torch.full((2,20),.3)):
            reference=single_mlem(response,projection,response.sensitivity(),100,50,background)
            actual=single_mlem_checkpointed(response,projection,response.sensitivity(),100,50,background)
            for x,y in zip(actual,reference):torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_real_dose_and_all_independent_worker_seeds_required(self):
        c=dict(passed=True,study=STUDY,total_primary_photons=TOTAL,worker_indices=list(range(1000)),
            seeds=list(range(SEED_BASE,SEED_BASE+1000)),views=list(range(1,21)),primary_counts=[15_000_000_000,35_000_000_000,0],
            binary_sha256=BINARY_SHA,crystal_sha256=CRYSTAL_SHA)
        self.assertIs(validate_collection(c),c)
        for k,v in [('total_primary_photons',5_000_000_000),('seeds',[SEED_BASE]*1000),('primary_counts',[5,5,0]),('views',list(range(1,20)))]:
            wrong=copy.deepcopy(c);wrong[k]=v
            with self.assertRaises(ValueError):validate_collection(wrong)

    def test_32_gpu_ids_and_conservative_aggregate_node_memory(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'allocation.txt';p.write_text('NumNodes=8 AllocTRES=cpu=320,mem=3200000M,node=8,gres/gpu=32')
            mem=400000*(1<<20)
            records=[dict(rank=i,local_rank=i%4,node='n'+str(i//4),gpu_uuid='GPU'+str(i),host_peak_rss_bytes=50<<30,
                host_allocated_bytes_node=mem,peak_reserved_bytes=16<<30,total_device_bytes=24<<30,
                measured_gpu_used_peak_bytes=17<<30,gpu_memory_samples=3) for i in range(32)]
            self.assertEqual(verify_topology(records,p),mem)
            wrong=copy.deepcopy(records);wrong[1]['gpu_uuid']=wrong[0]['gpu_uuid']
            with self.assertRaises(ValueError):verify_topology(wrong,p)
            wrong=copy.deepcopy(records)
            for r in wrong[:4]:r['host_peak_rss_bytes']=90<<30
            with self.assertRaises(ValueError):verify_topology(wrong,p)
            wrong=copy.deepcopy(records);wrong[0]['measured_gpu_used_peak_bytes']=21<<30
            with self.assertRaises(ValueError):verify_topology(wrong,p)
            p.write_text('NumNodes=8 AllocTRES=cpu=320,mem=3200000M,node=8,gres/gpu=8')
            with self.assertRaises(ValueError):verify_topology(records,p)

    def test_checkpoint_preserves_history_and_rejects_corruption(self):
        active=np.arange(78920);g=ActiveGeometry(active,np.r_[np.ones(78920),np.zeros(53120)],np.tile(np.arange(132040)[:,None],(1,20)))
        h={c:np.ones((1,78920),np.float32)*(i+1) for i,c in enumerate(CHANNELS)}
        with tempfile.TemporaryDirectory() as d:
            for phase,channels in PHASE_CHANNELS.items():
                write_checkpoint(d,phase,10,{channels[0]:torch.from_numpy(h[channels[0]][0])},g,'sha','validation')
            self.assertEqual(len(verify_checkpoints(d,h,active,'sha','validation')),3)
            (Path(d)/'checkpoints_440_single/checkpoint_000010/Image_440_SinglePhoton_active.float32').write_bytes(b'bad')
            with self.assertRaises(ValueError):verify_checkpoints(d,h,active,'sha','validation')

    def test_only_three_channels_and_independent_modes(self):
        self.assertEqual(len(CHANNELS),3);self.assertFalse(any('Plus' in c for c in CHANNELS))
        self.assertEqual(execution_policy('validation'),(10,10));self.assertEqual(execution_policy('formal'),(10000,50))
        with self.assertRaises(ValueError):execution_policy('old')

    def test_cuda_launcher_uses_four_local_ranks_and_no_explicit_mem(self):
        # Compile just the pure launch generator from its frozen source; no network import.
        source=Path(__file__).parent/'jscc_5e10_reconstruction_workflow.py'
        fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='launcher')
        env=dict(q=shlex.quote,GPU_BASE='/study',FACTORS='/factors',GPU_PYTHON='/python')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),env)
        script=env['launcher']('validation','/release',3600,10800,14220)
        self.assertIn('--nproc_per_node=4',script);self.assertIn('--node_rank="$SLURM_PROCID"',script)
        self.assertNotIn('--mem=',script);self.assertNotIn('cuda:0',script)

    def test_shared_numerical_sample_precedes_rank_partitioning(self):
        source=(Path(__file__).parent/'run_jscc_5e10.py').read_text()
        self.assertIn('numerical_checks(np.array(all_rows[:32],copy=True)',source)
        self.assertNotIn('numerical_checks(raw,',source)

    def test_formal_entry_has_no_joint_solver_call(self):
        tree=ast.parse((Path(__file__).parent/'run_jscc_5e10.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='compton_and_joint_mlem']
        self.assertEqual(len(calls),1)
        parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        n=calls[0];guards=[]
        while n in parents:
            n=parents[n]
            if isinstance(n,ast.If):guards.append(ast.unparse(n.test))
        self.assertIn("a.mode == 'validation'",guards)

if __name__=='__main__':unittest.main()
