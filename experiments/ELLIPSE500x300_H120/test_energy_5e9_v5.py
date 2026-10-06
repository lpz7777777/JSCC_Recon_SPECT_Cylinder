"""Identity and policy isolation tests for existing legacy 5e9 input."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from prepare_energy_5e9_v5 import validate_transport, metadata, selected_rows


class FiveBillionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.collection=dict(dataset='NEMA_Body_H60',level='5e9',views=list(range(1,21)),
            primary_counts=[1469053733,3530946267,0],worker_indices=list(range(200)),
            seeds=list(range(30100101,30100301)))

    def test_actual_primary_categories_are_not_worker_counts(self):
        proof=validate_transport(self.collection)
        self.assertEqual(proof['actual_primary_gamma'],5000000000)
        self.assertEqual(proof['event_policy'],'legacy')

    def test_1e9_ideal_and_invalid_identity_rejected(self):
        for key,bad in [('level','1e9'),('primary_counts',[0,1000000000,0]),
            ('seeds',list(range(200))),('worker_indices',list(range(199))),
            ('views',[True]+list(range(2,21))),('event_policy','ideal_first_scatter_v2')]:
            c=copy.deepcopy(self.collection);c[key]=bad
            with self.assertRaises(ValueError):validate_transport(c)

    def test_legacy_association_never_uses_ideal_only_or_ideal_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            with (folder/'events_v01.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['global_legacy_row','global_ideal_row','worker'])
                writer.writeheader();writer.writerows([
                    dict(global_legacy_row=-1,global_ideal_row=0,worker=0),
                    dict(global_legacy_row=0,global_ideal_row=-1,worker=1),
                    dict(global_legacy_row=1,global_ideal_row=5,worker=2)])
            rows=list(metadata(folder,1,{0}))
            self.assertEqual(len(rows),1);self.assertEqual(rows[0]['worker'],'1')
            self.assertEqual(len(list(metadata(folder,1))),2)

    def test_selection_uses_separate_legacy_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            from types import SimpleNamespace
            folder=Path(tmp);np.save(folder/'circle_train_legacy_v01_kept_rows.npy',np.array([1,5]))
            np.save(folder/'circle_train_ideal_v01_kept_rows.npy',np.array([8]))
            np.testing.assert_array_equal(selected_rows(SimpleNamespace(analysis=folder),'circle_train',1),[1,5])


if __name__=='__main__':unittest.main()
