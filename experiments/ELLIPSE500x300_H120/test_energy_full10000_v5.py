"""Full-entry numerical equivalence and durable 200-frame safeguards."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch_active_operator import ActiveGeometry, ViewResponse, single_mlem
from single_checkpoint_mlem import single_mlem_checkpointed
from energy_full10000_v5_contract import (execution_policy, load_contract,
    write_checkpoint, verify_checkpoints, CHANNELS, PHASE_CHANNELS)
from verify_energy_full10000_v5 import validate_authority


class Full10000EntryTests(unittest.TestCase):
    def geometry(self):
        return ActiveGeometry([0, 2], [1., 0., 1.], np.tile(np.arange(3)[:, None], (1, 2)))

    def test_original_single_equations_and_save_callbacks_match_with_and_without_cross_talk(self):
        g = self.geometry()
        response = ViewResponse(torch.tensor([[.2, .3, .4], [.5, .1, .3]]), g)
        projection = torch.tensor([[8., 6.], [11., 12.]])
        sensitivity = response.sensitivity()
        for background in (None, torch.tensor([[.1, 1.2], [3., .2]])):
            reference = single_mlem(response, projection, sensitivity, 100, 50, background)
            saved = []
            actual = single_mlem_checkpointed(response, projection, sensitivity, 100, 50,
                background, checkpoint_callback=lambda i, h: saved.append((i, h[-1].clone())))
            for x, y in zip(reference, actual):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertEqual([x[0] for x in saved], [50, 100])
            torch.testing.assert_close(saved[-1][1], actual[1][-1], rtol=0, atol=0)

    def test_explicit_10000_policy_and_old_entry_protection(self):
        self.assertEqual(execution_policy('formal'), (10000, 50))
        from energy_5e9_v5_contract import load_contract as old_load
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'contract.json'; path.write_text('{}')
            with self.assertRaises(ValueError): old_load(path, 'formal', 10000, 50)
            for mode, iterations, step in [('formal', 2000, 50), ('formal', 10000, 10), ('validation', 50, 50)]:
                with self.assertRaises(ValueError): load_contract(path, mode, iterations, step)

    def test_actual_authority_required(self):
        with self.assertRaises(ValueError): validate_authority(None, None, Path('contract.json'))

    def test_all_six_200_frame_checkpoints_and_corruption_rejection(self):
        g = self.geometry(); active = np.array([0, 2]); histories = {}
        for channel in CHANNELS:
            histories[channel] = np.arange(400, dtype=np.float32).reshape(200, 2) + 1
        with tempfile.TemporaryDirectory() as folder:
            for phase, channels in PHASE_CHANNELS.items():
                for frame in range(200):
                    frames = {c: torch.from_numpy(histories[c][frame]) for c in channels}
                    write_checkpoint(folder, phase, (frame+1)*50, frames, g, 'sha', 'formal')
            records = verify_checkpoints(folder, histories, active, 'sha', 'formal', full_count=3)
            self.assertEqual(len(records), 600)
            p = Path(folder) / 'checkpoints_compton_jscc/checkpoint_010000/Image_440_ComptonOnly_active.float32'
            p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError): verify_checkpoints(folder, histories, active, 'sha', 'formal', full_count=3)

    def test_invalid_snapshot_never_published_or_overwritten(self):
        g = self.geometry()
        with tempfile.TemporaryDirectory() as folder:
            frame = {'440_SinglePhoton': torch.tensor([float('nan'), 1.])}
            with self.assertRaises(ValueError): write_checkpoint(folder, '440_single', 50, frame, g, 'sha', 'formal')
            self.assertEqual(list(Path(folder).rglob('checkpoint_*')), [])
            frame = {'440_SinglePhoton': torch.ones(2)}
            write_checkpoint(folder, '440_single', 10000, frame, g, 'sha', 'formal')
            with self.assertRaises(ValueError): write_checkpoint(folder, '440_single', 10000, frame, g, 'sha', 'formal')

    def test_real_frozen_contract_at_deployment(self):
        path = Path(__file__).parent / 'contract.json'
        if not path.exists(): self.skipTest('Real new contract only available in frozen deployment')
        cfg = load_contract(path, 'validation', 10, 10)
        self.assertEqual(cfg['accepted_events'], 483743)
        self.assertEqual(cfg['model'], 'continuous_energy')


if __name__ == '__main__': unittest.main()
