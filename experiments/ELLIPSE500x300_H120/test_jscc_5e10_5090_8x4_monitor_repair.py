"""Verify the recovery preserves primary topology and scientific sources."""
import ast
import os
from pathlib import Path
import unittest

import jscc_5e10_5090_8x4_monitor_repair as repair
from jscc_5e10_common import read, digest


class PrimaryMonitorRepair(unittest.TestCase):
    def original(self):
        return Path(os.environ.get('JSCC_REPAIR_ORIGINAL_KERNEL', str(repair.ORIGINAL)))

    def repaired(self):
        return Path(os.environ.get('JSCC_UUID_REPAIR_KERNEL', str(repair.PAYLOAD)))

    def test_original_32_gpu_setup_and_resource_collection_unchanged(self):
        old = ast.parse((self.original() / 'jscc_5e10_runtime.py').read_text())
        new = ast.parse((self.repaired() / 'jscc_5e10_runtime.py').read_text())
        for name in ('setup', 'gather_resources'):
            a = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b))
        old_monitor = next(n for n in old.body if isinstance(n, ast.ClassDef) and n.name == 'DeviceMonitor')
        new_monitor = next(n for n in new.body if isinstance(n, ast.ClassDef) and n.name == 'DeviceMonitor')
        for name in ('sample', 'loop', 'finish'):
            a = next(n for n in old_monitor.body if n.name == name)
            b = next(n for n in new_monitor.body if n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b))

    def test_all_original_scientific_members_and_input_bindings_unchanged(self):
        old = read(self.original() / 'kernel_config.json')
        new = read(self.repaired() / 'kernel_config.json')
        self.assertEqual({k: v for k, v in old.items() if k != 'files'},
                         {k: v for k, v in new.items() if k != 'files'})
        for name, sha in old['files'].items():
            if name != 'jscc_5e10_runtime.py' and '__pycache__' not in Path(name).parts and not name.endswith('.pyc'):
                self.assertEqual(digest(self.repaired() / name), sha, name)
        self.assertEqual((new['nodes'], new['gpus_per_node'], new['world_size']), (8, 4, 32))

    def test_only_original_selection_entry_and_isolated_outputs_launched(self):
        release = '/immutable/primary-monitor-repair'
        script = repair.repair_launcher(release)
        self.assertIn('--nnodes=8 --nproc_per_node=4', script)
        self.assertIn('torch.cuda.device_count()==4', script)
        self.assertIn(release + '/jscc_5e10_selection.py', script)
        self.assertIn(repair.GPU_BASE + '/input', script)
        self.assertIn(repair.GPU_BASE + '/allocations/' + repair.STAGE + '_', script)
        self.assertIn(repair.GPU_BASE + '/' + repair.STAGE + '_', script)
        self.assertNotIn('run_jscc_5e10.py', script)
        self.assertNotIn('--mem', script)
        self.assertNotIn('/selection_1685272', script)


    def test_actual_pending_scheduler_range_and_wrong_requests(self):
        snapshot = (repair.REPORT / (repair.STAGE + '_submit_scontrol.txt')).read_text()
        self.assertIn('NumNodes=8-8 ', snapshot)
        self.assertTrue(repair.submitted_topology_matches(snapshot))
        self.assertTrue(repair.submitted_topology_matches(snapshot.replace('NumNodes=8-8 ', 'NumNodes=8 ')))
        for before, after in [('NumNodes=8-8 ', 'NumNodes=8-9 '),
                              ('NumCPUs=256 ', 'NumCPUs=128 '),
                              ('gres/gpu=32,', 'gres/gpu=16,'),
                              ('gres/gpu=32 ', 'gres/gpu=16 '),
                              ('TresPerNode=gres/gpu:4 ', 'TresPerNode=gres/gpu:2 '),
                              ('Partition=gpu_5090 ', 'Partition=gpu_4090 ')]:
            if before in snapshot:
                self.assertFalse(repair.submitted_topology_matches(snapshot.replace(before, after)))


if __name__ == '__main__':
    unittest.main()
