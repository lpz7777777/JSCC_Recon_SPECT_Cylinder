"""Damage-rejection tests for independent partition evidence, without GPU work."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from jscc_5e10_common import digest, read, write
from jscc_5e10_selection_parts import verify_parts


class PartsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ('parts', 'selections', 'selected_rows'):
            (self.root / name).mkdir()
        self.manifest = dict(part_receipt_sha256={}, events_per_view=[2, 2], original_accepted=6, removed=2, accepted_events=4)
        for view in (1, 2):
            indices, rows = [], []
            for rank in (0, 1):
                name = f'{view:02d}_rank{rank:02d}'
                selected = np.array([rank * 3 + 1], dtype='<i8')
                row = np.array([[view, rank, 100, 340]], dtype='<f4')
                np.save(self.root / 'parts' / (name + '.npy'), selected)
                np.save(self.root / 'parts' / (name + '_rows.npy'), row)
                receipt = dict(rank=rank, view=view, raw_rows=3, raw_global_offset=rank * 3,
                               original_accepted=rank + 1, kept=1, removed=rank,
                               q_chunk_max_absolute_difference=0,
                               selection_sha256=digest(self.root / 'parts' / (name + '.npy')),
                               selected_rows_sha256=digest(self.root / 'parts' / (name + '_rows.npy')))
                write(self.root / 'parts' / (name + '.json'), receipt)
                self.manifest['part_receipt_sha256'][name] = digest(self.root / 'parts' / (name + '.json'))
                indices.append(selected)
                rows.append(row)
            np.save(self.root / 'selections' / f'{view}.npy', np.concatenate(indices))
            np.save(self.root / 'selected_rows' / f'{view}.npy', np.concatenate(rows))
        self.collection = dict(raw_list_rows=12)

    def tearDown(self):
        self.temp.cleanup()

    def verify(self):
        return verify_parts(self.root, self.manifest, self.collection, world=2, views=2)

    def change_receipt(self, **changes):
        path = self.root / 'parts/01_rank01.json'
        write(path, dict(read(path), **changes))
        self.manifest['part_receipt_sha256']['01_rank01'] = digest(path)

    def test_complete_closure(self):
        self.assertEqual(self.verify()['receipts'], 4)

    def test_rank_gap_rejected_even_with_new_hash(self):
        self.change_receipt(raw_global_offset=4)
        with self.assertRaisesRegex(ValueError, 'offsets'):
            self.verify()

    def test_missing_receipt_rejected(self):
        del self.manifest['part_receipt_sha256']['01_rank01']
        with self.assertRaisesRegex(ValueError, 'Missing/extra'):
            self.verify()

    def test_selected_cache_change_rejected(self):
        path = self.root / 'selected_rows/1.npy'
        array = np.load(path)
        array[0, 2] += 1
        np.save(path, array)
        with self.assertRaisesRegex(ValueError, 'Final arrays'):
            self.verify()

    def test_partition_bytes_change_rejected(self):
        path = self.root / 'parts/01_rank01.npy'
        np.save(path, np.array([5], dtype='<i8'))
        with self.assertRaisesRegex(ValueError, 'array SHA'):
            self.verify()

    def test_counts_rejected_even_with_new_hash(self):
        self.change_receipt(removed=0)
        with self.assertRaisesRegex(ValueError, 'counts'):
            self.verify()

    def test_collection_total_rejected(self):
        self.collection['raw_list_rows'] -= 1
        with self.assertRaisesRegex(ValueError, 'totals'):
            self.verify()

    def test_executed_q_failure_rejected(self):
        self.change_receipt(q_chunk_max_absolute_difference=1e-6)
        with self.assertRaisesRegex(ValueError, 'stable-q'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
