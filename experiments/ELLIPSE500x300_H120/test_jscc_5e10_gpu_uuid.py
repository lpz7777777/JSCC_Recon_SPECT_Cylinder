"""Exercise the actual monitor repair against CUDA/NVIDIA UUID formatting."""
import ast
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import jscc_5e10_5090_8x2_trial as trial
from jscc_5e10_gpu_uuid import canonical_gpu_uuid

RAW = '12345678-9abc-def0-1234-56789abcdef0'
NVIDIA = 'GPU-' + RAW


class GPUIdentity(unittest.TestCase):
    def runtime(self):
        if 'JSCC_UUID_REPAIR_KERNEL' in os.environ:
            return (Path(os.environ['JSCC_UUID_REPAIR_KERNEL']) / 'jscc_5e10_runtime.py').read_text()
        return trial.topology_sources(trial.DATA / 'kernel_payload', monitor_repair=True)['jscc_5e10_runtime.py']

    def test_full_uuid_only(self):
        self.assertEqual(canonical_gpu_uuid(RAW), NVIDIA)
        self.assertEqual(canonical_gpu_uuid(NVIDIA), NVIDIA)
        self.assertEqual(canonical_gpu_uuid(RAW.upper()), NVIDIA)
        for value in ('GPU-1234', '', '0'*32, '123456789abcdef0123456789abcdef0', 'MIG-' + RAW):
            with self.assertRaises(ValueError): canonical_gpu_uuid(value)

    def monitor(self, query):
        node = next(n for n in ast.parse(self.runtime()).body if isinstance(n, ast.ClassDef) and n.name == 'DeviceMonitor')
        torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(uuid=RAW)), __version__='2.8.0+cu128')
        namespace = dict(torch=torch, subprocess=SimpleNamespace(check_output=query), os=os,
                         threading=SimpleNamespace(Event=threading.Event, Thread=Mock()), canonical_gpu_uuid=canonical_gpu_uuid)
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-monitor>', 'exec'), namespace)
        with patch.dict(os.environ, RANK='0'):
            return namespace['DeviceMonitor'](0)

    def test_actual_monitor_selects_uuid_not_ordinal(self):
        query = Mock(return_value='GPU-ffffffff-ffff-ffff-ffff-ffffffffffff, 999, 32000\n' + NVIDIA + ', 123, 32000\n')
        monitor = self.monitor(query)
        self.assertEqual((monitor.raw_uuid, monitor.uuid, monitor.samples, monitor.peak), (RAW, NVIDIA, 1, 123*(1<<20)))
        self.assertIn('--query-gpu=uuid,memory.used,memory.total', query.call_args[0][0])

    def test_actual_monitor_rejects_missing_or_ambiguous_device(self):
        for output in ('GPU-ffffffff-ffff-ffff-ffff-ffffffffffff, 999, 32000\n', (NVIDIA + ', 123, 32000\n')*2):
            with self.assertRaisesRegex(ValueError, 'UUID missing'):
                self.monitor(Mock(return_value=output))

    def test_selection_and_contract_are_byte_unchanged_by_monitor_repair(self):
        root = Path(os.environ['JSCC_TRIAL_ORIGINAL_KERNEL']) if 'JSCC_TRIAL_ORIGINAL_KERNEL' in os.environ else trial.DATA / 'kernel_payload'
        original = trial.topology_sources(root)
        repair = trial.topology_sources(root, monitor_repair=True)
        for name in ('jscc_5e10_selection.py', 'jscc_5e10_contract.py'):
            self.assertEqual(original[name], repair[name])
        self.assertIn('cuda_property_uuid_raw=_monitor.raw_uuid', repair['jscc_5e10_runtime.py'])
        self.assertIn('measured_gpu_used_peak_bytes=_monitor.peak', repair['jscc_5e10_runtime.py'])


if __name__ == '__main__':
    unittest.main()
