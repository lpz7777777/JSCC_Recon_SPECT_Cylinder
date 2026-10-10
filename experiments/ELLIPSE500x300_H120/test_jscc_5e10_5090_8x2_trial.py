"""Check complete-row/unchanged-science topology adaptation and resource rejection."""
import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jscc_5e10_5090_8x2_trial as trial
from jscc_5e10_common import host_allocated_bytes


class TrialIsolation(unittest.TestCase):
    def source(self):
        return Path(os.environ['JSCC_TRIAL_ORIGINAL_KERNEL']) if 'JSCC_TRIAL_ORIGINAL_KERNEL' in os.environ else trial.DATA / 'kernel_payload'

    def test_selection_science_and_all_raw_rows_preserved(self):
        root = self.source()
        changed = trial.topology_sources(root)
        old = ast.parse((root / 'jscc_5e10_selection.py').read_text())
        new = ast.parse(changed['jscc_5e10_selection.py'])
        # The one loop over final receipts changes its rank count; no event math changes.
        class Restore(ast.NodeTransformer):
            def visit_comprehension(self, node):
                if isinstance(node.target, ast.Name) and node.target.id == 'r' and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range' and isinstance(node.iter.args[0], ast.Name) and node.iter.args[0].id == 'world':
                    node.iter.args[0] = ast.Constant(32)
                return self.generic_visit(node)
        self.assertEqual(ast.dump(old), ast.dump(Restore().visit(new)))
        original_prefix = (root / 'jscc_5e10_contract.py').read_text().split('def verify_topology(')[0]
        self.assertEqual(original_prefix, changed['jscc_5e10_contract.py'].split('def verify_topology(')[0])
        self.assertIn('torch.set_num_threads(10)', changed['jscc_5e10_runtime.py'])

    def test_launch_keeps_input_algorithm_and_isolates_outputs(self):
        release = '/immutable/trial'
        script = trial.trial_launcher(release)
        self.assertIn('--nnodes=8 --nproc_per_node=2', script)
        self.assertIn('torch.cuda.device_count()==2', script)
        self.assertIn(release + '/jscc_5e10_selection.py', script)
        self.assertIn(trial.GPU_BASE + '/input', script)
        self.assertIn('/allocations/' + trial.STAGE + '_', script)
        self.assertNotIn('run_jscc_5e10.py', script)
        self.assertNotIn('--mem', script)
        with patch.object(trial, 'launcher', return_value='unknown launch'):
            with self.assertRaises(ValueError): trial.trial_launcher(release)

    def test_actual_16_gpu_topology_and_80percent_guard(self):
        text = trial.topology_sources(self.source())['jscc_5e10_contract.py']
        node = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'verify_topology'][0]
        namespace = dict(Path=Path, re=trial.re, host_allocated_bytes=host_allocated_bytes)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<trial-resource-contract>', 'exec'), namespace)
        verify = namespace['verify_topology']
        memory = 252000 * (1 << 20)
        resources = [dict(rank=i, local_rank=i % 2, node='node' + str(i // 2), gpu_uuid='uuid' + str(i),
            host_peak_rss_bytes=1 << 30, host_allocated_bytes_node=memory, total_device_bytes=32 << 30,
            peak_reserved_bytes=6 << 30, measured_gpu_used_peak_bytes=7 << 30, gpu_memory_samples=2) for i in range(16)]
        with tempfile.TemporaryDirectory() as folder:
            allocation = Path(folder) / 'allocation.txt'
            allocation.write_text('NumNodes=8 AllocTRES=cpu=128,mem=2016000M,node=8,gres/gpu=16')
            self.assertEqual(verify(resources, allocation), memory)
            resources[1]['gpu_uuid'] = resources[0]['gpu_uuid']
            with self.assertRaises(ValueError): verify(resources, allocation)
            resources[1]['gpu_uuid'] = 'uuid1'
            resources[0]['host_peak_rss_bytes'] = memory
            with self.assertRaises(ValueError): verify(resources, allocation)
            resources[0]['host_peak_rss_bytes'] = 1 << 30
            allocation.write_text('NumNodes=8 AllocTRES=cpu=256,mem=4032000M,node=8,gres/gpu=32')
            with self.assertRaises(ValueError): verify(resources, allocation)


if __name__ == '__main__':
    unittest.main()
