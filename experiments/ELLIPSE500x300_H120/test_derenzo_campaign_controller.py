"""Dynamic topology evidence, immutable submissions and queued formal gates."""
import ast
import copy
import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import derenzo_campaign_workflow as workflow
from derenzo_campaign_common import HERE,write,read
from derenzo_campaign_gate import limits_from_validation
from jscc_5e10_common import host_allocated_bytes


class CampaignControllerTests(unittest.TestCase):
    def test_actual_resources_reject_duplicates_allocation_and_node_memory_errors(self):
        tree=ast.parse((HERE/'derenzo_campaign_reconstruction.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_resources')
        env=dict(Path=Path,re=re,host_allocated_bytes=host_allocated_bytes)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual resource checker>','exec'),env)
        plan=dict(nodes=1,gpus_per_node=4,world_size=4);memory=504000*(1<<20)
        records=[dict(rank=i,local_rank=i,node='n0',gpu_uuid='GPU'+str(i),host_peak_rss_bytes=50<<30,
                      host_allocated_bytes_node=memory,peak_reserved_bytes=8<<30,total_device_bytes=32<<30,
                      measured_gpu_used_peak_bytes=9<<30,gpu_memory_samples=3) for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'allocation.txt';path.write_text('NumNodes=1 AllocTRES=cpu=32,mem=504000M,node=1,gres/gpu=4')
            self.assertEqual(env['verify_resources'](records,plan,path),memory)
            bad=copy.deepcopy(records);bad[1]['gpu_uuid']=bad[0]['gpu_uuid']
            with self.assertRaises(ValueError):env['verify_resources'](bad,plan,path)
            bad=copy.deepcopy(records)
            for r in bad:r['host_peak_rss_bytes']=110<<30
            with self.assertRaises(MemoryError):env['verify_resources'](bad,plan,path)
            path.write_text('NumNodes=1 AllocTRES=cpu=32,mem=504000M,node=1,gres/gpu=1')
            with self.assertRaises(ValueError):env['verify_resources'](records,plan,path)

    def test_registered_job_or_unresolved_intent_cannot_submit_again(self):
        with tempfile.TemporaryDirectory() as d,patch.object(workflow,'connection') as connect:
            root=Path(d);(root/'EHE_218').mkdir()
            with patch.object(workflow,'REPORT',root):
                write(root/'EHE_218/formal_job.json',dict(job=123))
                self.assertEqual(workflow.submit_gpu('EHE_218','formal','validation'),dict(job=123))
                write(root/'EHE_218/validation_submission_intent.json',dict(resolved=False))
                with self.assertRaisesRegex(RuntimeError,'Unresolved'):workflow.submit_gpu('EHE_218','validation','validation')
            connect.assert_not_called()

    def test_own_energy_formal_gate_and_measured_time_scaling(self):
        records=[dict(phase_solve_seconds={'218_SinglePhoton':1.},prepare_seconds=0.,elapsed_seconds=20.)]
        ehe=limits_from_validation(dict(dataset='EHE_218',resources=records))
        self.assertEqual(ehe['minutes'],120)
        jscc=limits_from_validation(dict(dataset='JSCC_218',resources=records))
        self.assertGreaterEqual(jscc['minutes'],ehe['minutes'])
        bad=copy.deepcopy(records);bad[0]['phase_solve_seconds']['218_SinglePhoton']=float('nan')
        with self.assertRaises(ValueError):limits_from_validation(dict(dataset='EHE_218',resources=bad))


if __name__=='__main__':unittest.main()
