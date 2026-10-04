import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from generate_compton_tiled_field_v3 import specs_for_plan,verify_resume,PE_HASH,SCATTER_HASH
from generate_compton_a_tile_pilot_v3 import extract_selected
from generate_compton_a_guard import write
from plan_compton_a_tiles_v3 import tile_spec


class ResumableTileTests(unittest.TestCase):
    def test_shards_are_disjoint_complete_and_keep_exact_specs(self):
        p=dict(xy_tile_indices=[[21,21],[21,42]],pilot_specs=[tile_spec(21,21,5)])
        allspecs=specs_for_plan(p,'full');shards=[allspecs[i::4] for i in range(4)]
        self.assertEqual(len(allspecs),20)
        self.assertEqual(len({s['name'] for group in shards for s in group}),20)
        self.assertEqual(specs_for_plan(p,'pilot'),p['pilot_specs'])
        with self.assertRaises(ValueError):specs_for_plan(p,'invalid')

    def test_resume_checks_every_stored_value_without_recomputing(self):
        selected=np.array([0,2,4]);spec=tile_spec(21,21,5)
        raw=np.arange(5*17**3,dtype='<f4').reshape(5,17,17,17)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);compact=extract_selected(raw,folder/'A_selected.float32',selected)
            r=dict(status='PHYSICAL_TILE_STORED_ACCURACY_HOLD',spec=spec,plan_sha256='plan',compact=compact,
                pe_binary_sha256=PE_HASH,scatter_binary_sha256=SCATTER_HASH)
            write(folder/'compact_receipt.json',r)
            self.assertEqual(verify_resume(folder,spec,'plan',selected),r)
            with self.assertRaises(ValueError):verify_resume(folder,spec,'changed',selected)
            with self.assertRaises(ValueError):verify_resume(folder,spec,'plan',selected[::-1])
            with (folder/'A_selected.float32').open('r+b') as f:f.write(b'bad!')
            with self.assertRaises(ValueError):verify_resume(folder,spec,'plan',selected)
            self.assertTrue((folder/'compact_receipt.json').exists())

    def test_incomplete_tile_is_preserved_and_requires_explicit_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);raw=folder/'failed_raw';raw.write_bytes(b'preserve')
            with self.assertRaises(ValueError):verify_resume(folder,tile_spec(21,21,5),'plan',np.arange(3))
            self.assertEqual(raw.read_bytes(),b'preserve')


if __name__=='__main__':unittest.main()
