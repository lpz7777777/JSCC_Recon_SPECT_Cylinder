"""Direct numeric tests for the paired study; runs on the existing Torch host."""
from dataclasses import fields
from pathlib import Path
import sys
import unittest
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parents[1])]
from first_scatter_offline import quadrature,energy_layers
import tempfile
import csv
from torch_active_operator import ActiveGeometry,ViewResponse,compton_and_joint_mlem
from compton_event_response import (ComptonEventSettings,PreparedComptonEvents,
    prepare_compton_events,build_detector_position_variance,min_standardized_compton_arm,
    select_normalized_response_rows)
from detector_csv import load_detector_coordinates

BASE=Path("/home/lipeize/JSCC_FOV120_20260924/experiments/ELLIPSE500x300_H120/generated")

class FirstScatterNumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_full_circle_q_is_independent_of_batch_and_rank_partition(self):
        geo=np.load(HERE/"geometry.npz" if (HERE/"geometry.npz").exists() else BASE/"Geometry/geometry.npz")
        self.assertEqual(geo["coordinates_mm"].shape,(132040,3))
        detector=torch.tensor(load_detector_coordinates(BASE/"FactorsCalibrated/440keV_RotateNum20/Detector.csv",10496))
        var=build_detector_position_variance(detector,0)
        settings=ComptonEventSettings(.440,.13*np.sqrt(511/440),2*.440**2/(.511+2*.440)-.001,.05,.35)
        events=torch.tensor([[1,.125,2625,.290],[122,.09,5340,.330],[100,.2,7990,.225],
                             [7629,.128853,1936,.316147]]*2)
        prepared,_=prepare_compton_events(events,settings,detector,var,var,input_energies_already_smeared=True)
        self.assertIsNotNone(prepared)
        coords=torch.tensor(geo["coordinates_mm"],dtype=torch.float32)
        whole=min_standardized_compton_arm(prepared,coords,settings)
        scores=torch.empty_like(whole)
        for rank in range(4):
            indices=torch.arange(prepared.count)[rank::4]
            if not len(indices):continue
            part=PreparedComptonEvents(**{f.name:None if getattr(prepared,f.name) is None else
                 getattr(prepared,f.name)[indices] for f in fields(prepared)})
            scores[indices]=min_standardized_compton_arm(part,coords,settings)
        self.assertTrue(torch.equal(scores,whole))
        normal=torch.tensor([[.5,.5],[.5,.5]])
        self.assertEqual(select_normalized_response_rows(normal,1,torch.tensor([3.,3.00001]),3)[0].tolist(),[True,False])

    def test_object_ellipse_operator_and_its_transpose(self):
        active=np.array([0,1,3]);fraction=np.array([.02,1.,0.,.7])
        geometry=ActiveGeometry(active,fraction,np.array([[0,1],[1,2],[2,3],[3,0]]))
        gen=torch.Generator().manual_seed(912)
        rows=torch.rand((6,4),generator=gen,dtype=torch.float64)
        x=torch.rand((3,1),generator=gen,dtype=torch.float64)
        y=torch.rand((6,1),generator=gen,dtype=torch.float64)
        for view in range(2):
            lhs=(geometry.forward(rows,x,view)*y).sum()
            rhs=(x*geometry.adjoint(rows,y,view)).sum()
            self.assertLess(float(abs(lhs-rhs)/abs(lhs)),1e-5)

    def test_persistent_callbacks_leave_joint_mlem_unchanged(self):
        geo=ActiveGeometry(np.arange(3),np.array([1.,.1,.7]),np.array([[0,1],[1,2],[2,0]]))
        response=ViewResponse(torch.tensor([[.2,.1,.5],[.4,.6,.2]]),geo)
        projection=torch.tensor([[13.,15.],[23.,24.]])
        blocks=[[torch.tensor([[.3,.1,.2],[.2,.4,.1]])],[torch.tensor([[.1,.2,.5]])]]
        sensitivity=response.sensitivity();sd=torch.tensor([[.2],[.1],[.3]])
        before=compton_and_joint_mlem(response,projection,blocks,sensitivity,sd,100,50)
        snapshots=[]
        def save(i,hd,hj):snapshots.append((i,hd[-1].clone(),hj[-1].clone()))
        after=compton_and_joint_mlem(response,projection,blocks,sensitivity,sd,100,50,checkpoint_callback=save)
        for a,b in zip(before,after):
            for aa,bb in zip(a,b):self.assertTrue(torch.equal(aa,bb))
        self.assertEqual([s[0] for s in snapshots],[50,100])

    def test_quadrature_volume_jacobian_and_no_double_volume(self):
        points,w=quadrature((0,3,0,2*np.pi),1.5,64)
        self.assertAlmostEqual(w.sum(),np.pi*9*3,places=10)
        self.assertTrue(np.allclose(np.average(points,axis=0,weights=w),(0,0,1.5),atol=1e-10))
        p,w=quadrature((147,153,np.pi/2-.04,np.pi/2+.04),-58.5,128,8,8)
        self.assertTrue(np.all((p[:,0]/250)**2+(p[:,1]/150)**2<=1+1e-13))
        self.assertTrue(np.all((p[:,2]>=-60)&(p[:,2]<=-57)))
        # B/V is a density response: a constant A integrated once is A*V.
        full_volume=(153**2-147**2)*.08/2*3
        b=7*full_volume
        self.assertAlmostEqual(float(np.dot(np.full(len(w),b/full_volume),w)),7*w.sum(),places=9)

    def test_world_positions_and_detector_centers_use_same_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/"point_0";folder.mkdir()
            r=dict(c1=1,c2=2,source_x=0,source_y=-345,source_z=0,
                p1_x=0,p1_y=-45,p1_z=0,p2_x=12,p2_y=-15,p2_z=0,
                transfer_mev=.1,true_e1=.1,measured_e1=.11,view=1,seed=1,event_id=1,
                legacy=1,ideal=1,reason="accepted")
            with (folder/"events_v01.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(r));writer.writeheader();writer.writerow(r)
            rows=energy_layers(root,np.array([[0,300,0],[12,330,0]],float),root)
            self.assertAlmostEqual(rows[0]["crystal_center_residual"],0,places=12)

if __name__=="__main__":unittest.main()
