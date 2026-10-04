"""Regression must retain historical q3; gates and iteration bounds are strict."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from compton_geometry_run_contract import validate_run


class RunContractTests(unittest.TestCase):
    def test_regression_keeps_q3_and_switches_only_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); evidence = root/'spatial.json'
            evidence.write_text(json.dumps(dict(status='PASSED')))
            digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            study = dict(study='compton_response_geometry_v3', variant='R1_stable_point',
                group='ideal_first_scatter_v2', max_min_standardized_arm=3,
                quality_domain='full_circle_132040', iterations=2000, save_step=50,
                geometry_sha256='geometry', kernel_sha256='kernel',
                validation_evidence_sha256={'spatial.json': digest(evidence)})
            args = dict(regression=True, pilot=False, dry_run=False, iterations=50,
                save_step=50, dataset='NEMA_Body_H60', level='1e9', channels='compton-jscc',
                sensitivity=root/'Sensi_d', geometry_sha='geometry', kernel_sha='kernel',
                digest=digest, config_directory=root)
            self.assertEqual(validate_run(study, **args), ('legacy', 3.0))
            args.update(regression=False, pilot=True, iterations=10, save_step=10)
            self.assertEqual(validate_run(study, **args), ('stable_float64', 3.0))
            args.update(pilot=False, iterations=10000, save_step=50)
            with self.assertRaises(ValueError): validate_run(study, **args)
            args.update(iterations=2000); evidence.write_text(json.dumps(dict(status='HOLD')))
            with self.assertRaises(ValueError): validate_run(study, **args)
            study['validation_evidence_sha256']['spatial.json'] = digest(evidence)
            with self.assertRaises(ValueError): validate_run(study, **args)
            evidence.write_text(json.dumps(dict(status='PASSED')))
            study['validation_evidence_sha256']['spatial.json'] = digest(evidence)
            study['variant'] = 'R2_stable_overlap'
            with self.assertRaises(ValueError): validate_run(study, **args)

    def test_image_regression_reads_40_frame_q3_baseline_and_rejects_filter_off(self):
        from verify_first_scatter import verify
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);result=root/'result';baseline=root/'baseline';result.mkdir();baseline.mkdir()
            digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
            gate=root/'validation_gate.json';gate.write_text(json.dumps(dict(status='PASSED')))
            active=np.arange(82040);geometry=root/'geometry.npz';np.savez(geometry,active_indices=active)
            config=root/'R1.json';kernel='frozen_kernel'
            cfg=dict(study='compton_response_geometry_v3',variant='R1_stable_point',group='ideal_first_scatter_v2',
                max_min_standardized_arm=3,quality_domain='full_circle_132040',iterations=2000,save_step=50,
                geometry_sha256=digest(geometry),kernel_sha256=kernel,
                validation_gate_sha256=digest(gate),validation_evidence_sha256={'validation_gate.json':digest(gate)},
                baseline_accepted_compton_events=91231,kept_compton_events=91225,
                baseline_per_view=[91231],kept_per_view=[91225],baseline_input_sha256={},
                baseline_factor_manifest_sha256={},baseline_sensi_d_sha256='oldS',sensi_d_sha256='newS',channels=['440_ComptonOnly'])
            config.write_text(json.dumps(cfg))
            image=np.ones(82040,dtype='<f4');full=np.r_[image,np.zeros(50000,dtype='<f4')]
            for name,data in (('active',image),('full',full),('history',image)):
                data.tofile(result/f'Image_440_ComptonOnly_{name}.float32')
            np.tile(image,40).tofile(baseline/'Image_440_ComptonOnly_history.float32')
            resources=[dict(rank=r,node=f'node{r}',accepted_events=91231 if r==0 else 0,
                host_allocated_bytes=1024,host_peak_rss_bytes=100,peak_reserved_bytes=100,total_device_bytes=1024) for r in range(4)]
            run=dict(iterations=50,save_step=50,world_size=4,pixels_active=82040,pixels_full=132040,resources=resources,
                dataset='NEMA_Body_H60',count_level='1e9',channels='compton-jscc',
                code_sha256={'compton_event_response.py':kernel},input_sha256={},factor_manifest_sha256={},
                geometry_sha256=digest(geometry),sensi_d_sha256='oldS',accepted_compton_events=91231,
                accepted_compton_events_per_view=[91231],response_mismatch=dict(filter_enabled=True,
                    config_sha256=digest(config),event_policy=cfg['group'],validation_gate_sha256=digest(gate),
                    geometry_mode='legacy',variant=cfg['variant']))
            path=result/'run_manifest.json';path.write_text(json.dumps(run))
            verify(result,config,geometry,baseline,'regression')
            self.assertEqual(json.loads((result/'verification.json').read_text())['outputs'][0]['baseline_frame50_relative_l2'],0)
            run['response_mismatch']['filter_enabled']=False;path.write_text(json.dumps(run))
            with self.assertRaises(ValueError):verify(result,config,geometry,baseline,'regression')


if __name__ == '__main__': unittest.main()
