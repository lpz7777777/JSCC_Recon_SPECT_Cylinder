"""Independent numerical checks for the diagnostic stable geometry prototype."""
import unittest
import torch
from diagnose_compton_geometry_stability import stable_geometry

class StableGeometryTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(4)

    def test_variance_matches_finite_difference_of_geometric_angle(self):
        source=torch.tensor([40.,-80.,30.],dtype=torch.float64)
        r1=torch.tensor([4.,300.,-15.],dtype=torch.float64)
        r2=torch.tensor([-20.,360.,20.],dtype=torch.float64)
        v1=torch.tensor([.75,.75,.75],dtype=torch.float64)
        v2=torch.tensor([1/3,3.,1/3],dtype=torch.float64)
        def angle(a,b):
            u=(a-source)/(a-source).norm();v=(b-a)/(b-a).norm()
            return torch.atan2(torch.linalg.cross(u,v).norm(),u@v)
        derivatives=[]
        for index in range(6):
            delta=torch.zeros(3,dtype=torch.float64);delta[index%3]=1e-3
            plus=angle(r1+delta,r2) if index<3 else angle(r1,r2+delta)
            minus=angle(r1-delta,r2) if index<3 else angle(r1,r2-delta)
            derivatives.append((plus-minus)/.002)
        expected=(torch.stack(derivatives).square()*torch.cat([v1,v2])).sum().sqrt()
        _,sigma,_=stable_geometry((r1-source)[None,None],(r2-r1)[None,None],v1[None],v2[None])
        self.assertLess(abs(float(sigma)-float(expected)),1e-10)

    def test_near_parallel_and_antiparallel_obey_finite_bound(self):
        angles=torch.tensor([0.,1e-12,1e-9,1e-6,1e-4,.1],dtype=torch.float64)
        for sign in (-1,1):
            a=torch.tensor([[[0.,300.,0.]]],dtype=torch.float64).expand(1,len(angles),3)
            b=torch.stack([angles.sin()*30,sign*angles.cos()*30,torch.zeros_like(angles)],1)[None]
            beta,sigma,upper=stable_geometry(a,b,torch.full((1,3),.75),torch.tensor([[1/3,3.,1/3]]))
            self.assertTrue(bool(torch.isfinite(beta).all() and torch.isfinite(sigma).all()))
            self.assertTrue(bool((sigma<=upper*(1+1e-10)).all()))
            self.assertLess(float(sigma.max()),.1)

    def test_translation_invariance(self):
        source=torch.tensor([[[4.,20.,3.]]],dtype=torch.float64)
        first=torch.tensor([[[10.,300.,8.]]],dtype=torch.float64)
        second=torch.tensor([[[-5.,360.,-9.]]],dtype=torch.float64)
        shift=torch.tensor([[[1e4,-2e4,3e4]]],dtype=torch.float64)
        v=torch.ones(1,3,dtype=torch.float64)
        a=stable_geometry(first-source,second-first,v,v)
        b=stable_geometry((first+shift)-(source+shift),(second+shift)-(first+shift),v,v)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,rtol=1e-12,atol=1e-12)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable on this host')
    def test_cpu_cuda_agreement_near_collinearity(self):
        angles=torch.logspace(-12,-2,30,dtype=torch.float64)
        a=torch.tensor([[[0.,300.,0.]]],dtype=torch.float64).expand(1,len(angles),3)
        b=torch.stack([angles.sin()*30,-angles.cos()*30,torch.zeros_like(angles)],1)[None]
        v1=torch.full((1,3),.75);v2=torch.tensor([[1/3,3.,1/3]])
        cpu=stable_geometry(a,b,v1,v2)
        gpu=stable_geometry(a.cuda(),b.cuda(),v1.cuda(),v2.cuda())
        for x,y in zip(cpu,gpu):torch.testing.assert_close(x,y.cpu(),rtol=1e-10,atol=1e-10)

if __name__=='__main__':unittest.main()
