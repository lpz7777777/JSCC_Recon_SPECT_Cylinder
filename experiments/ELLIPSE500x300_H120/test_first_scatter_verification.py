"""Exercise rejection gates with small histories on the exact production shapes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from verify_first_scatter import digest,verify,allocated_host_bytes

class VerificationGates(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.result=self.root/"pilot";self.result.mkdir()
        self.geometry=self.root/"geometry.npz"
        active=np.arange(82040,dtype=np.int32);np.savez(self.geometry,active_indices=active)
        gate=self.root/"validation_gate.json";gate.write_text('{"status":"PASSED"}')
        self.config=self.root/"legacy.json"
        self.cfg=dict(group="legacy",channels=["440_ComptonOnly","440_SinglePlusCompton"],
            validation_gate_sha256=digest(gate),geometry_sha256=digest(self.geometry),baseline_input_sha256={"input":"a"},
            baseline_factor_manifest_sha256={"Factor":"b"},kept_compton_events=500,kept_per_view=[25]*20,
            baseline_per_view=[0]*20,sensi_d_sha256="S1",baseline_sensi_d_sha256="S0")
        self.config.write_text(json.dumps(self.cfg))
        self.run=dict(iterations=10,save_step=10,world_size=4,pixels_active=82040,pixels_full=132040,
            resources=[dict(rank=i,node=f"node{i}",accepted_events=125,host_allocated_bytes=1000,
                host_peak_rss_bytes=500,peak_reserved_bytes=500,total_device_bytes=1000) for i in range(4)],
            response_mismatch=dict(filter_enabled=True,config_sha256=digest(self.config),event_policy="legacy",
                validation_gate_sha256=digest(gate)),input_sha256=self.cfg["baseline_input_sha256"],
            factor_manifest_sha256=self.cfg["baseline_factor_manifest_sha256"],geometry_sha256=digest(self.geometry),
            accepted_compton_events=500,accepted_compton_events_per_view=[25]*20,sensi_d_sha256="S1")
        self.save_manifest()
        for channel in self.cfg["channels"]:
            a=np.ones(82040,dtype="<f4");f=np.zeros(132040,dtype="<f4");f[active]=a
            for kind,value in (("active",a),("full",f),("history",a)):
                value.tofile(self.result/f"Image_{channel}_{kind}.float32")
    def save_manifest(self):(self.result/"run_manifest.json").write_text(json.dumps(self.run))
    def run_verify(self):verify(self.result,self.config,self.geometry,self.root,"pilot")
    def test_valid_full_shape_pilot_is_accepted(self):self.run_verify()
    def test_actual_slurm_memory_overrides_conflicting_per_cpu_memory(self):
        allocation=self.root/"allocation.txt"
        allocation.write_text("NumNodes=4 NumCPUs=24 MinMemoryCPU=15750M AllocTRES=cpu=24,mem=240000M,node=4,gres/gpu=4")
        actual=60000*1024**2
        self.assertEqual(allocated_host_bytes(allocation,4),actual)
        for r in self.run["resources"]:r["host_allocated_bytes"]=actual;r["host_peak_rss_bytes"]=int(.5*actual)
        self.save_manifest();verify(self.result,self.config,self.geometry,self.root,"pilot",allocation)
        self.run["resources"][0]["host_allocated_bytes"]=94500*1024**2;self.save_manifest()
        with self.assertRaisesRegex(ValueError,"actual Slurm allocation"):
            verify(self.result,self.config,self.geometry,self.root,"pilot",allocation)
        allocation.write_text("NumNodes=4 AllocTRES=(null)")
        with self.assertRaisesRegex(ValueError,"memory missing"):allocated_host_bytes(allocation,4)
    def test_independent_hold_never_accepts_images(self):
        (self.root/"validation_gate.json").write_text('{"status":"HOLD"}')
        self.cfg["validation_gate_sha256"]=digest(self.root/"validation_gate.json")
        self.config.write_text(json.dumps(self.cfg))
        with self.assertRaisesRegex(ValueError,"not passed"):self.run_verify()
    def test_wrong_sensitivity_and_view_closure_are_rejected(self):
        for key,value in (("sensi_d_sha256","wrong"),("accepted_compton_events_per_view",[26]+[25]*19)):
            original=copy.deepcopy(self.run);self.run[key]=value;self.save_manifest()
            with self.assertRaises(ValueError):self.run_verify()
            self.run=original
    def test_insufficient_gpu_or_host_margin_is_rejected(self):
        for key in ("peak_reserved_bytes","host_peak_rss_bytes"):
            self.run["resources"][0][key]=801;self.save_manifest()
            with self.assertRaisesRegex(ValueError,"resource margin"):self.run_verify()
            self.run["resources"][0][key]=500
    def test_nonfinite_history_and_ellipse_leak_are_rejected(self):
        channel=self.cfg["channels"][0]
        h=self.result/f"Image_{channel}_history.float32";a=np.fromfile(h,"<f4");a[0]=np.nan;a.tofile(h)
        with self.assertRaisesRegex(ValueError,"Invalid image"):self.run_verify()
        a[0]=1;a.tofile(h)
        full=self.result/f"Image_{channel}_full.float32";f=np.fromfile(full,"<f4");f[-1]=1;f.tofile(full)
        with self.assertRaisesRegex(ValueError,"ellipse closure"):self.run_verify()

if __name__=="__main__":unittest.main()
