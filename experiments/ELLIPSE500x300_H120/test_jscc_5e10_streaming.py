"""Exact disk replay and damaged-cache/topology rejection checks."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from jscc_5e10_streaming_cache import CacheWriter, DiskBlocks
from jscc_5e10_streaming_contract import verify_topology
from jscc_5e10_compton_mlem import compton_mlem
from torch_active_operator import ActiveGeometry, ViewResponse, compton_and_joint_mlem


class StreamingTests(unittest.TestCase):
    def store(self, folder, name, arrays):
        writer = CacheWriter(Path(folder) / name, arrays[0].shape[1], reserve_bytes=0)
        for x in arrays:
            writer.append(x)
        return writer.finish()

    def test_full_saved_trajectory_matches_resident_and_original_compton(self):
        torch.manual_seed(503)
        geometry = ActiveGeometry([0, 2], [1., 0., 1.], np.tile(np.arange(3)[:, None], (1, 20)))
        response = ViewResponse(torch.tensor([[.2, .3, .4], [.5, .1, .3]]), geometry)
        projection = torch.arange(40, dtype=torch.float32).reshape(2, 20) + 1
        sensitivity = torch.tensor([[2.], [3.]])
        resident = [[torch.rand(3, 2) + .01, torch.rand(2, 2) + .01] for _ in range(20)]
        with tempfile.TemporaryDirectory() as folder:
            disk = [DiskBlocks(self.store(folder, str(i), values)) for i, values in enumerate(resident)]
            actual = compton_mlem(disk, sensitivity, 100, 50)
            expected = compton_mlem(resident, sensitivity, 100, 50)
            original, _ = compton_and_joint_mlem(response, projection, resident, response.sensitivity(), sensitivity, 100, 50)
            for x, y, z in zip(actual, expected, original):
                self.assertTrue(torch.equal(x, y))
                self.assertTrue(torch.equal(x, z))

    def test_exact_zero_and_small_float32_values_survive_repeated_replay(self):
        values = torch.tensor([[0., 1.e-40, .5], [3., 0., 2.e-35]], dtype=torch.float32)
        with tempfile.TemporaryDirectory() as folder:
            block = DiskBlocks(self.store(folder, 'v', [values]))
            for _ in range(3):
                self.assertTrue(torch.equal(torch.cat(list(block)), values))
            self.assertEqual(block.read_passes, 3)

    def test_encoded_corruption_and_truncated_records_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = self.store(folder, 'v', [torch.ones(2, 3)])
            p = Path(receipt['path'])
            body = p.read_bytes()
            p.write_bytes(body[:-1])
            with self.assertRaises(ValueError):
                list(DiskBlocks(receipt))
            p.write_bytes(body[:-1] + bytes([body[-1] ^ 1]))
            with self.assertRaises(ValueError):
                list(DiskBlocks(receipt))

    def test_missing_or_extra_event_counts_and_file_sha_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = self.store(folder, 'v', [torch.ones(2, 3)])
            wrong = dict(receipt, events=3)
            with self.assertRaises(ValueError):
                list(DiskBlocks(wrong))
            Path(receipt['path']).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                DiskBlocks(receipt).verify_file()

    def test_partial_cache_never_overwritten_and_disk_reserve_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'v'
            writer = CacheWriter(p, 3, reserve_bytes=1 << 62)
            with self.assertRaises(OSError):
                writer.append(torch.ones(1, 3))
            writer.file.close()
            with self.assertRaises(FileExistsError):
                CacheWriter(p, 3)

    def test_24_rank_identity_and_aggregate_memory_are_measured(self):
        with tempfile.TemporaryDirectory() as folder:
            allocation = Path(folder) / 'allocation.txt'
            allocation.write_text('NumNodes=8 AllocTRES=cpu=144,mem=1440000M,node=8,gres/gpu=24')
            memory = 180000 * (1 << 20)
            records = [dict(rank=i, local_rank=i % 3, node='n' + str(i // 3), gpu_uuid='GPU' + str(i),
                host_peak_rss_bytes=16 << 30, host_allocated_bytes_node=memory,
                peak_reserved_bytes=12 << 30, total_device_bytes=24 << 30,
                measured_gpu_used_peak_bytes=13 << 30, gpu_memory_samples=3) for i in range(24)]
            self.assertEqual(verify_topology(records, allocation), memory)
            wrong = copy.deepcopy(records)
            wrong[1]['gpu_uuid'] = wrong[0]['gpu_uuid']
            with self.assertRaises(ValueError):
                verify_topology(wrong, allocation)
            wrong = copy.deepcopy(records)
            for r in wrong[:3]:
                r['host_peak_rss_bytes'] = 50 << 30
            with self.assertRaises(ValueError):
                verify_topology(wrong, allocation)


if __name__ == '__main__':
    unittest.main()
