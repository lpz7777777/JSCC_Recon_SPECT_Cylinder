"""Independent numerical and geometric checks for the spike ablation."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import torch

from regularized_update import SpatialGraph, solve_surrogate, kl_prox, AblationUpdate
from torch_active_operator import ActiveGeometry, ViewResponse, forward_project, single_mlem, compton_and_joint_mlem

HERE=Path(__file__).resolve().parent
EXPERIMENT=Path(os.environ.get("JSCC_PROJECT_ROOT",str(HERE.parents[1])))/"experiments/ELLIPSE500x300_H120"
STUDY=EXPERIMENT/"generated/SpikeAblation/NEMA_5e9_SPIKE_ABLATION_V1"


def graph():
    return SpatialGraph([0,1],[1,2],[.5,.5],[1.,1.],3)


def toy():
    geom=ActiveGeometry(np.arange(3),np.ones(3),np.arange(3)[:,None])
    response=ViewResponse(torch.tensor([[1.,.3,.1],[.2,.7,.2],[.1,.1,.9]]),geom)
    counts=forward_project(response,torch.tensor([[2.],[1.],[3.]]))
    events=[[torch.tensor([[.7,.2,.1],[.1,.3,.6],[.15,.7,.15]])]]
    return geom,response,counts,events,torch.tensor([[.4],[.5],[.6]])


class Checks(unittest.TestCase):
    def test_graph_adjoint_and_constant(self):
        g=graph(); x=torch.tensor([.4,2.,7.],dtype=torch.float64); p=torch.tensor([.5,-.7],dtype=torch.float64)
        self.assertAlmostEqual(float(g.difference(x)@p),float(x@g.transpose(p)),places=13)
        self.assertEqual(float(g.penalty(torch.ones(3,dtype=torch.float64),0)),0.)
        self.assertEqual(float(g.penalty(torch.ones(3,dtype=torch.float64),1)),0.)

    def test_kl_prox_stationarity(self):
        v=torch.tensor([.1,2.,4.],dtype=torch.float64); a=torch.tensor([.01,.5,.4],dtype=torch.float64)
        step=torch.tensor([30.,3.,1.],dtype=torch.float64); incoming=torch.tensor([-10.,1.,8.],dtype=torch.float64)
        x=kl_prox(incoming,v,a,step)
        residual=x-incoming+step*a*(1-v/x)
        self.assertLess(float(residual.abs().max()),1e-12)

    def test_huber_surrogate_against_scipy(self):
        g=graph(); current=torch.ones(3,dtype=torch.float64); v=torch.tensor([.2,8.,.5],dtype=torch.float64)
        a=torch.tensor([.2,.3,.5],dtype=torch.float64); lam=.1; delta=1.
        result,_,info=solve_surrogate(current,v,a,g,lam,delta,5000,1e-11)
        def objective(x):
            diff=np.abs(np.diff(x)); rho=np.where(diff<=delta,diff*diff/(2*delta),diff-delta/2)
            return np.dot(a.numpy(),x-v.numpy()+v.numpy()*np.log(v.numpy()/x))+lam*.5*rho.sum()
        reference=minimize(objective,np.ones(3),method="L-BFGS-B",bounds=[(1e-9,None)]*3,
                           options={"ftol":1e-14,"gtol":1e-10})
        np.testing.assert_allclose(result.numpy(),reference.x,rtol=3e-5,atol=2e-6)
        self.assertLess(info["gap"],1e-8)
        self.assertLessEqual(info["surrogate_delta"],0)

    def test_tv_surrogate_against_constrained_reference(self):
        g=graph(); current=torch.ones(3,dtype=torch.float64); v=torch.tensor([.2,8.,.5],dtype=torch.float64)
        a=torch.tensor([.2,.3,.5],dtype=torch.float64); lam=.15
        result,_,info=solve_surrogate(current,v,a,g,lam,0.,5000,1e-11)
        def objective(q):
            x=q[:3]
            return np.dot(a.numpy(),x-v.numpy()+v.numpy()*np.log(v.numpy()/x))+lam*.5*q[3:].sum()
        constraints=[{"type":"ineq","fun":lambda q:q[3:]-np.diff(q[:3])},
                     {"type":"ineq","fun":lambda q:q[3:]+np.diff(q[:3])}]
        reference=minimize(objective,np.r_[np.ones(3),np.ones(2)],method="SLSQP",
                           bounds=[(1e-9,None)]*3+[(0,None)]*2,constraints=constraints,
                           options={"ftol":1e-12,"maxiter":500})
        self.assertTrue(reference.success,reference.message)
        np.testing.assert_allclose(result.numpy(),reference.x[:3],rtol=4e-5,atol=4e-6)
        self.assertLess(info["gap"],1e-8)

    def updater(self, folder, method, strength=0., group=(0,1,2)):
        np.savez(Path(folder)/"model.npz",edge_i=[0,1],edge_j=[1,2],graph_weight=[.5,.5],
                 gradient_scale=[1.,1.],binding_group=np.array(group))
        return AblationUpdate({"inner_max":120,"inner_gap_tolerance":1e-8},Path(folder)/"model.npz",
                              {"method":method,"strength":strength,"huber_delta":1.},"cpu")

    def test_zero_strength_matches_original_joint(self):
        _,response,counts,events,sd=toy(); ss=response.sensitivity()
        expected=compton_and_joint_mlem(response,counts,events,ss,sd,20,5)
        with tempfile.TemporaryDirectory() as folder:
            update=self.updater(folder,"huber",0)
            update.configure("440_compton",sd,3)
            update.configure("440_jscc",ss+sd,float(counts.sum())+3)
            result=compton_and_joint_mlem(response,counts,events,ss,sd,20,5,update_rule=update)
        for left,right in zip(expected,result):
            torch.testing.assert_close(left[0],right[0],rtol=2e-6,atol=1e-6)

    def test_binding_is_explicit_reduced_operator_mlem(self):
        _,response,counts,_,_=toy(); H=response.matrix(0).numpy(); P=np.array([[1,0],[1,0],[0,1]],dtype=np.float32)
        small_geom=ActiveGeometry(np.arange(2),np.ones(2),np.arange(2)[:,None])
        small=ViewResponse(torch.from_numpy(H@P),small_geom)
        expected,_=single_mlem(small,counts,small.sensitivity(),40,10)
        with tempfile.TemporaryDirectory() as folder:
            update=self.updater(folder,"binding",group=(0,0,1))
            update.configure("single",response.sensitivity(),float(counts.sum()))
            fit,_=single_mlem(response,counts,response.sensitivity(),40,10,progress_label="single",update_rule=update)
        np.testing.assert_allclose(fit.numpy(),P@expected.numpy(),rtol=2e-6,atol=2e-6)

    def test_penalized_joint_objective_decreases(self):
        _,response,counts,events,sd=toy(); ss=response.sensitivity()
        for method in ("huber","tv"):
            with tempfile.TemporaryDirectory() as folder:
                update=self.updater(folder,method,.03)
                update.configure("440_compton",sd,3)
                update.configure("440_jscc",ss+sd,float(counts.sum())+3)
                result=compton_and_joint_mlem(response,counts,events,ss,sd,100,50,update_rule=update)
                for state in update.states.values():
                    self.assertLess(state["max_objective_increase"],2e-6)
                self.assertTrue(torch.isfinite(result[1][0]).all())

    def test_frozen_full_geometry_graph_and_ties(self):
        with np.load(STUDY/"spatial_model.npz") as model:
            i,j=model["edge_i"],model["edge_j"]; n=len(model["binding_group"])
            adjacency=coo_matrix((np.ones(len(i)*2),(np.r_[i,j],np.r_[j,i])),shape=(n,n))
            self.assertEqual(connected_components(adjacency,directed=False)[0],1)
            self.assertEqual(n,82040)
            self.assertEqual(len(np.unique(model["binding_group"])),81240)
            self.assertTrue(np.all(model["graph_weight"]>0))
            self.assertTrue(np.all(model["face_area_mm2"]>0))
            with np.load(EXPERIMENT/"generated/Geometry/geometry.npz") as g:
                xyz=g["coordinates_mm"][g["active_indices"]]
                anchor=model["binding_anchor"]
                np.testing.assert_array_equal(xyz[:,2],xyz[anchor,2])
                self.assertLessEqual(np.linalg.norm(xyz-xyz[anchor],axis=1).max(),12.00001)
                self.assertAlmostEqual(model["effective_volume_mm3"].sum(),np.pi*250*150*120,delta=np.pi*250*150*120*.001)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
