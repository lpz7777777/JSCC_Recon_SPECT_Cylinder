"""Independent density, continuity and exact-chunking checks."""
from pathlib import Path
import json
import unittest
import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.optimize import brentq,minimize_scalar
from compton_event_response import (PreparedComptonEvents,min_standardized_compton_arm,
    ComptonEventSettings)
from compton_energy_probability_v5 import (ContinuousTransferLaw,
    selected_logpdf_pit,energy_log_density,geometry_arrays,normalized_proxy_response,
    fixed_q_min,fixed_q_intervals)

HERE=Path(__file__).parent


class EnergyCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=HERE/'transfer_training_summary.json'
        if not path.exists():path=HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4/evidence/energy_energy_probability_summary.json'
        cls.law=ContinuousTransferLaw.load(path)

    def prepared(self):
        p1=torch.tensor([[0.,300.,0.],[12.,390.,6.]],dtype=torch.float32)
        p2=torch.tensor([[24.,330.,6.],[-6.,330.,0.]],dtype=torch.float32)
        variance=torch.tensor([[.75,.75,.75],[1/3,3.,1/3]],dtype=torch.float32)
        return PreparedComptonEvents(torch.tensor([1,2]),torch.tensor([2,1]),
            torch.tensor([.095,.180]),torch.tensor([.330,.230]),p1,p2,variance,variance)

    def test_all_layers_and_angles_have_defined_physical_law(self):
        for layer in range(4):
            for b in np.linspace(0,np.pi,181):
                means,sd,weights,_=self.law.components(b,layer,.01)
                self.assertTrue(len(means)>0 and np.all((means>0)&(means<.440)))
                self.assertTrue(np.isfinite(sd).all() and (sd>0).all())
                self.assertAlmostEqual(weights.sum(),1.)

    def test_density_is_continuous_when_transfer_nodes_cross_zero(self):
        from compton_energy_probability_v5 import free_transfer
        checked=0
        for layer in range(4):
            for k in (0,8,32,64):
                function=lambda b:free_transfer(b)+self.law.nodes(b,layer)[0][k]
                if function(0)*function(np.pi)>=0:continue
                root=brentq(function,0,np.pi);checked+=1
                logp=[]
                for b in (root-1e-9,root+1e-9):
                    means,sd,weights,_=self.law.components(b,layer,.01)
                    logp.append(logsumexp(-.5*((.06-means)/sd)**2-np.log(sd)+np.log(weights)))
                self.assertLess(abs(np.expm1(logp[1]-logp[0])),1e-5)
        self.assertGreaterEqual(checked,4)

    def test_anchor_interpolation_is_continuous(self):
        for layer,(angles,_) in self.law.anchors.items():
            for a in angles:
                left,_=self.law.nodes(np.radians(a)-1e-8,layer)
                right,_=self.law.nodes(np.radians(a)+1e-8,layer)
                np.testing.assert_allclose(left,right,atol=1e-8,rtol=0)

    def test_disjoint_selection_density_integrates_to_one(self):
        means,sd,weights,_=self.law.components(1.1,0,.015)
        intervals=[(.07,.095),(.115,.25)]
        density=lambda x:np.exp(selected_logpdf_pit(x,intervals,means,sd,weights)[0])
        area=sum(quad(density,lo,hi,epsabs=1e-9)[0] for lo,hi in intervals)
        self.assertAlmostEqual(area,1.,places=8)
        self.assertAlmostEqual(selected_logpdf_pit(.07,intervals,means,sd,weights)[1],0.)
        self.assertAlmostEqual(selected_logpdf_pit(.25,intervals,means,sd,weights)[1],1.)

    def test_torch_density_matches_scalar_probability(self):
        p=self.prepared();coords=torch.tensor([[0.,0.,0.],[120.,12.,57.],[-225.,0.,0.]])
        ll,beta=energy_log_density(p,coords,self.law,node_chunk=17)
        _,sp=geometry_arrays(p,coords)
        for e in range(2):
            for j in range(len(coords)):
                ref=self.law.log_density(float(p.e1[e]),float(beta[e,j]),(0,3)[e],float(sp[e,j]))
                self.assertAlmostEqual(float(ll[e,j]),float(ref),places=9)

    def test_analytic_bin_integral_matches_independent_quadrature(self):
        from compton_energy_probability_v5 import log_transfer_integral,ENERGY_VARIANCE_SLOPE
        for energy in (.05,.17,.275):
            for lo,hi in ((0.,.035),(.08,.14),(.24,.44)):
                for d in (0.,.0001,.01):
                    def log_density(t):
                        variance=ENERGY_VARIANCE_SLOPE*t+d
                        if variance==0:return -np.inf
                        return -.5*(energy-t)**2/variance-.5*np.log(2*np.pi*variance)
                    mode=minimize_scalar(lambda t:-log_density(t),bounds=(lo,hi),method='bounded').x
                    scale=max(log_density(lo),log_density(hi),log_density(mode))
                    value=quad(lambda t:np.exp(log_density(t)-scale),lo,hi,epsabs=1e-10,epsrel=1e-10)[0]
                    reference=np.log(value)+scale
                    actual=float(log_transfer_integral(energy,lo,hi,d))
                    self.assertAlmostEqual(np.exp(actual-reference),1.,places=7)

    def test_node_chunk_and_event_partition_do_not_change_response(self):
        p=self.prepared();coords=torch.tensor([[0.,0.,0.],[120.,12.,57.],[-225.,0.,0.]])
        x,_=energy_log_density(p,coords,self.law,1)
        y,_=energy_log_density(p,coords,self.law,31)
        torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)
        pieces=[]
        for i in range(2):
            values={k:(None if v is None else v[i:i+1]) for k,v in p.__dict__.items()}
            pieces.append(energy_log_density(PreparedComptonEvents(**values),coords,self.law,31)[0])
        torch.testing.assert_close(x,torch.cat(pieces),atol=1e-10,rtol=1e-10)

    def test_hybrid_tensor_matches_its_scalar_reference(self):
        p=self.prepared();coords=torch.tensor([[0.,0.,0.],[120.,12.,57.],[-225.,0.,0.]])
        ll,beta=energy_log_density(p,coords,self.law,node_chunk=17,backend='tail16_mid2')
        _,sp=geometry_arrays(p,coords)
        for i in range(2):
            for j in range(len(coords)):
                m,s,w,_=self.law.components(float(beta[i,j]),(0,3)[i],float(sp[i,j]),order='tail16_mid2')
                ref=logsumexp(-.5*((float(p.e1[i])-m)/s)**2-np.log(s)-.5*np.log(2*np.pi)+np.log(w))
                self.assertAlmostEqual(float(ll[i,j]),float(ref),places=9)

    def test_forward_proxy_has_no_selection_mass_divisor(self):
        p=self.prepared();coords=torch.tensor([[0.,0.,0.],[120.,12.,57.],[-225.,0.,0.]])
        B=torch.tensor([[2.,.5,1.],[.1,3.,4.]])
        R=normalized_proxy_response(p,coords,B,self.law,16)
        ll,beta=energy_log_density(p,coords,self.law,16)
        energy=p.e1.double();kn=.440/(.440-energy)+(.440-energy)/.440
        expected=torch.softmax(ll+torch.log(kn[:,None]-torch.sin(beta)**2)+torch.log(B[p.cpnum1-1].double()),dim=1)
        torch.testing.assert_close(R.double(),expected,atol=1e-7,rtol=1e-7)
        torch.testing.assert_close(R.sum(1),torch.ones(2))

    def test_q_domain_uses_original_kernel_and_float32_energy(self):
        p=self.prepared();coords=torch.tensor([[0.,0.,0.],[120.,12.,57.],[-225.,0.,0.]])
        beta,sp=geometry_arrays(p,coords)
        settings=ComptonEventSettings(.440,.13*np.sqrt(.511/.440),
            2*.440**2/(.511+2*.440)-.001,.05,.35,geometry_mode='stable_float64')
        direct=min_standardized_compton_arm(p,coords,settings).numpy()
        q=np.array([fixed_q_min([float(p.e1[i])],beta[i],sp[i])[0] for i in range(2)])
        np.testing.assert_allclose(q,direct,rtol=0,atol=1e-12)
        for i in range(2):
            intervals=fixed_q_intervals(.05,settings.energy_threshold_max_mev,beta[i],sp[i],64)
            for energy in np.linspace(.051,settings.energy_threshold_max_mev-.001,67):
                measured_ok=fixed_q_min([energy],beta[i],sp[i])[0]<=3
                domain_ok=any(lo<=energy<=hi for lo,hi in intervals)
                self.assertEqual(measured_ok,domain_ok)


if __name__=='__main__':unittest.main()
