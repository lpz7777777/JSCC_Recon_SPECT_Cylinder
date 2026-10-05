"""Continuous energy-density candidate; default production kernel is untouched.

The transfer law remains conditional on the recorded ideal multihit training
sample. This module does not certify an unconditional C1/C2/E2 joint model.
Conditional validation divides by the selection mass. Forward proxy weights
use the unconditioned energy density, never that position-dependent divisor.
"""
from pathlib import Path
import json
import math
import numpy as np
import torch
from scipy.special import log_ndtr, logsumexp
from compton_event_response import (stable_compton_geometry,ComptonEventSettings,
    compton_theta_from_e1,_energy_angle_sigma)

E0=.440
MASS=.511
RESOLUTION=.13
QUADRATURE_ORDER=8
ENERGY_VARIANCE_SLOPE=(RESOLUTION/2.355)**2*MASS
EDGES=np.array([0,45,60,75,90,120,150,180.000001])


def free_transfer(beta):
    return E0-E0/(1+E0/MASS*(1-np.cos(beta)))


def transfer_derivative(beta):
    return E0**2/MASS*np.sin(beta)/(1+E0/MASS*(1-np.cos(beta)))**2


def log_transfer_integral(energy,lo,hi,position_variance):
    """Integral over T of N(E; T, c*T+d), evaluated in stable log space.

    A primitive is Phi(u)-exp(2*E/c+2*d/c**2)*Phi(v), with
    u=(T-E)/sqrt(c*T+d), v=-(T+E+2*d/c)/sqrt(c*T+d).
    Differentiate it to recover the Gaussian density. Use its complementary
    primitive above E to avoid subtracting CDF values rounded to one.
    """
    lo,hi,d=np.broadcast_arrays(lo,hi,position_variance);c=ENERGY_VARIANCE_SLOPE
    exponent=2*energy/c+2*d/c**2
    values=[]
    for t in (lo,hi):
        sd=np.sqrt(np.maximum(c*t+d,1e-300))
        u=(t-energy)/sd;v=-(t+energy+2*d/c)/sd
        a=log_ndtr(u);b=exponent+log_ndtr(v)
        with np.errstate(divide='ignore',invalid='ignore'):
            cdf=a+np.log(-np.expm1(np.minimum(b-a,0)))
        tail=np.logaddexp(log_ndtr(-u),b)
        values.append((cdf,tail))
    high=np.where(hi<=energy,values[1][0],values[0][1])
    low=np.where(hi<=energy,values[0][0],values[1][1])
    with np.errstate(divide='ignore',invalid='ignore'):
        result=high+np.log(-np.expm1(np.minimum(low-high,0)))
    return np.where(hi>lo,result,-np.inf)


class ContinuousTransferLaw:
    """Fixed quantile interpolation; sparse angles use endpoint extrapolation.

    Extrapolation defines a candidate at every angle; it is flagged as such,
    not treated as evidence that the material law is measured there.
    """
    def __init__(self, summary):
        self.summary=summary
        self.anchors={}
        for layer in range(4):
            rows=sorted((r for r in summary['training_laws'] if r['layer']==layer),
                        key=lambda r:r['angle_bin'])
            if len(rows)<2:raise ValueError('Layer lacks two independent training angle bins')
            angles=np.array([(EDGES[r['angle_bin']]+EDGES[r['angle_bin']+1])/2 for r in rows])
            nodes=np.array([r['quantile_nodes_MeV'] for r in rows])
            if nodes.shape!=(len(rows),128) or not np.isfinite(nodes).all():
                raise ValueError('Frozen 128-node transfer laws required')
            if np.any(np.diff(nodes,axis=1)<=0):
                raise ValueError('Strict quantiles required for continuous residual bins')
            self.anchors[layer]=(angles,nodes)

    @classmethod
    def load(cls,path):return cls(json.loads(Path(path).read_text()))

    def nodes(self,beta,layer):
        angles,nodes=self.anchors[int(layer)]
        degrees=np.degrees(beta)
        result=np.array([np.interp(degrees,angles,nodes[:,k]) for k in range(128)])
        return result,bool(degrees<angles[0] or degrees>angles[-1])

    def residual_edges(self,beta,layer):
        nodes,extra=self.nodes(beta,layer)
        edges=np.r_[nodes[0]-(nodes[1]-nodes[0])/2,
                    (nodes[:-1]+nodes[1:])/2,
                    nodes[-1]+(nodes[-1]-nodes[-2])/2]
        return edges,extra

    def components(self,beta,layer,sigma_position=0.,material=True,order=QUADRATURE_ORDER):
        """Continuous bin mass at physical boundaries; no all-or-nothing nodes.

        Each midpoint-delimited residual bin has prior mass 1/128. Its retained
        width supplies the weight after intersecting physical transfer support.
        Fixed Gauss-Legendre quadrature integrates each retained uniform bin.
        """
        t=float(free_transfer(beta))
        if material:
            edges,extra=self.residual_edges(beta,layer)
            lo=np.maximum(t+edges[:-1],0);hi=np.minimum(t+edges[1:],E0)
            width=np.maximum(hi-lo,0);mass=width/np.diff(edges)
            parts=[(0,1,16),(1,127,2),(127,128,16)] if order=='tail16_mid2' else [(0,128,order)]
            means=[];weights=[]
            for begin,end,count in parts:
                x,w=np.polynomial.legendre.leggauss(count)
                means.append((lo[begin:end,None]+width[begin:end,None]*(x+1)/2).ravel())
                weights.append((mass[begin:end,None]*w[None]/2).ravel())
            means=np.concatenate(means);weights=np.concatenate(weights)
            keep=weights>0;means=means[keep];weights=weights[keep]
            if not len(means):raise ValueError('Physical transfer support is empty')
            weights/=weights.sum()
        else:
            # Dirac-limit numerical regularization only at beta exactly 0.
            means=np.array([np.clip(t,1e-12,E0-1e-12)]);weights=np.ones(1);extra=False
        sigma=np.sqrt((RESOLUTION/2.355)**2*MASS*means+
                      (float(transfer_derivative(beta))*sigma_position)**2)
        return means,sigma,weights,extra

    def log_density(self,energy,beta,layer,sigma_position=0.,material=True):
        t=float(free_transfer(beta));d=(float(transfer_derivative(beta))*sigma_position)**2
        if not material:
            variance=ENERGY_VARIANCE_SLOPE*max(t,1e-12)+d
            return float(-.5*(energy-t)**2/variance-.5*np.log(2*np.pi*variance))
        edges,_=self.residual_edges(beta,layer)
        lo=np.maximum(t+edges[:-1],0);hi=np.minimum(t+edges[1:],E0)
        retained=np.maximum(hi-lo,0)/np.diff(edges)
        integrals=log_transfer_integral(energy,lo,hi,d)-np.log(np.diff(edges))
        return float(logsumexp(integrals)-np.log(retained.sum()))

    def selected_probability(self,energy,intervals,beta,layer,sigma_position=0.,material=True,order=QUADRATURE_ORDER):
        means,sd,weights,extra=self.components(beta,layer,sigma_position,material,order)
        _,pit,z=selected_logpdf_pit(energy,intervals,means,sd,weights)
        return self.log_density(energy,beta,layer,sigma_position,material)-z,pit,z,extra


def log_interval_mass(lo,hi,means,sigma,weights=None):
    if hi<=lo:return -np.inf
    lower=(lo-means)/sigma;upper=(hi-means)/sigma
    high=np.where(lower>=0,log_ndtr(-lower),log_ndtr(upper))
    low=np.where(lower>=0,log_ndtr(-upper),log_ndtr(lower))
    with np.errstate(divide='ignore',invalid='ignore'):
        terms=high+np.log(-np.expm1(low-high))
    weights=np.full(len(means),1/len(means)) if weights is None else weights
    return float(logsumexp(terms+np.log(weights)))


def selected_logpdf_pit(energy,intervals,means,sigma,weights=None):
    intervals=np.asarray(intervals,dtype=float).reshape(-1,2)
    if np.any(intervals[:,1]<=intervals[:,0]) or np.any(intervals[1:,0]<intervals[:-1,1]):
        raise ValueError('Disjoint sorted positive selection intervals required')
    if not any(lo<=energy<=hi for lo,hi in intervals):raise ValueError('Energy outside selection domain')
    weights=np.full(len(means),1/len(means)) if weights is None else weights
    z=logsumexp([log_interval_mass(lo,hi,means,sigma,weights) for lo,hi in intervals])
    ll=logsumexp(-.5*((energy-means)/sigma)**2-np.log(sigma)-.5*np.log(2*np.pi)+np.log(weights))
    below=logsumexp([log_interval_mass(lo,min(hi,energy),means,sigma,weights) for lo,hi in intervals])
    if not np.isfinite(z):raise ValueError('Nonfinite selection mass')
    return float(ll-z),float(np.exp(below-z)),float(z)


def geometry_arrays(prepared,coordinates):
    beta,sp,_=stable_compton_geometry(prepared.pos1.double()[:,None]-coordinates.double()[None],
        (prepared.pos2.double()-prepared.pos1.double())[:,None],
        prepared.sigma_pos1_sq,prepared.sigma_pos2_sq,True)
    return beta,sp


def analytic_bin_tensor(energy,left,right,bin_width,position_variance):
    """Pure float64 tensor function; optional fusion changes execution only."""
    width=(right-left).clamp_min(0);c=ENERGY_VARIANCE_SLOPE;d=position_variance
    exponent=2*energy/c+2*d/c**2;primitive=[]
    for transfer in (left,right):
        sd=(c*transfer+d).clamp_min(1e-300).sqrt()
        u=(transfer-energy)/sd;v=-(transfer+energy+2*d/c)/sd
        a=torch.special.log_ndtr(u);bb=exponent+torch.special.log_ndtr(v)
        cdf=a+(-torch.expm1((bb-a).clamp_max(0))).log()
        tail=torch.logaddexp(torch.special.log_ndtr(-u),bb)
        primitive.append((cdf,tail))
    high=torch.where(right<=energy,primitive[1][0],primitive[0][1])
    low=torch.where(right<=energy,primitive[0][0],primitive[1][1])
    integral=high+(-torch.expm1((low-high).clamp_max(0))).log()
    logpdf=torch.where(width>0,integral-bin_width.log(),-torch.inf)
    return torch.logsumexp(logpdf,dim=-1),(width/bin_width).sum(-1)


_fused_bin_tensor=None


def bin_tensor_backend(compiled):
    global _fused_bin_tensor
    if not compiled:return analytic_bin_tensor
    if _fused_bin_tensor is None:
        _fused_bin_tensor=torch.compile(analytic_bin_tensor,dynamic=True,fullgraph=True)
    return _fused_bin_tensor


def fixed_q_min(energies,beta,sigma_position,chunk=32):
    """Frozen stable-float64 q on the complete circle, with List float32 energies."""
    result=[]
    settings=ComptonEventSettings(E0,.13*np.sqrt(.511/E0),
        2*E0**2/(MASS+2*E0)-.001,.05,.35,geometry_mode='stable_float64')
    for part in np.array_split(np.asarray(energies),max(1,math.ceil(len(energies)/chunk))):
        e=torch.tensor(part,device=beta.device,dtype=torch.float32).double()
        theta=compton_theta_from_e1(e,E0)
        b=beta[None].expand(len(e),-1)
        se=_energy_angle_sigma(e,settings,b,theta)
        q=((b-theta[:,None]).abs()/(se.square()+sigma_position[None].square()).sqrt()).amin(1)
        result.extend(q.cpu().numpy().tolist())
    return np.asarray(result)


def fixed_q_intervals(lo,hi,beta,sigma_position,nodes=512):
    """Numerical selection intervals; must compare adjacent scan refinements.

    Never assumes a single interval and never selects an imaging event. Very
    narrow components can be missed by a finite scan; refinement evidence is
    reported rather than asserting an analytic exact acceptance domain.
    """
    energy=np.linspace(lo,hi,nodes+1);accepted=fixed_q_min(energy,beta,sigma_position)<=3
    cuts=[]
    for i in np.flatnonzero(accepted[1:]!=accepted[:-1]):
        left,right=float(energy[i]),float(energy[i+1]);left_ok=bool(accepted[i])
        for _ in range(24):
            middle=(left+right)/2;middle_ok=bool(fixed_q_min([middle],beta,sigma_position)[0]<=3)
            if middle_ok==left_ok:left=middle
            else:right=middle
        cuts.append((left+right)/2)
    boundaries=[lo,*cuts,hi];intervals=[]
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        if fixed_q_min([(left+right)/2],beta,sigma_position)[0]<=3:intervals.append((left,right))
    return intervals


def energy_log_density(prepared,coordinates,law,node_chunk=16,material=True,compiled_bins=False,backend='analytic',gaussian_float32=False):
    """Block-wise analytic bin integral, including existing position variance.

    No event selection and no energy-gate/q normalizer occur here. The position
    uncertainty remains the original uniform-crystal, first-order propagation.
    No source truth is an argument.
    """
    beta,sp=geometry_arrays(prepared,coordinates)
    free=E0-E0/(1+E0/MASS*(1-beta.cos()))
    slope=E0**2/MASS*beta.sin()/(1+E0/MASS*(1-beta.cos())).square()
    layers=torch.round((prepared.pos1[:,1].abs()-300)/30).long()
    output=torch.empty_like(beta)
    for layer in layers.unique().tolist():
        chosen=layers==layer;b=beta[chosen];t=free[chosen];s=(slope*sp)[chosen]
        energy=prepared.e1.double()[chosen,None,None]
        if material:
            angle,nodes=law.anchors[int(layer)]
            a=torch.tensor(np.radians(angle),device=b.device,dtype=b.dtype)
            edges=np.concatenate((nodes[:,:1]-(nodes[:,1:2]-nodes[:,:1])/2,
                (nodes[:,:-1]+nodes[:,1:])/2,
                nodes[:,-1:]+(nodes[:,-1:]-nodes[:,-2:-1])/2),axis=1)
            n=torch.tensor(edges,device=b.device,dtype=b.dtype)
            bounded=b.clamp(float(a[0]),float(a[-1]))
            upper=torch.searchsorted(a,bounded.contiguous()).clamp(1,len(a)-1)
            lower=upper-1;alpha=(bounded-a[lower])/(a[upper]-a[lower])
            if backend=='tail16_mid2':
                chunks=[(0,1,16)]+[(i,min(127,i+node_chunk),2) for i in range(1,127,node_chunk)]+[(127,128,16)]
            elif backend=='analytic':chunks=[(i,min(128,i+node_chunk),None) for i in range(0,128,node_chunk)]
            else:raise ValueError('Unknown numerical integration backend')
        else:chunks=[(0,1,None)]
        total=torch.full_like(b,-torch.inf);mass_total=torch.zeros_like(b)
        for start,end,quadrature_order in chunks:
            if material:
                delta=n[lower,start:end+1]*(1-alpha[...,None])+n[upper,start:end+1]*alpha[...,None]
                left=(t[...,None]+delta[...,:-1]).clamp_min(0)
                right=(t[...,None]+delta[...,1:]).clamp_max(E0)
                if quadrature_order is not None:
                    x,w=np.polynomial.legendre.leggauss(quadrature_order)
                    x=torch.tensor((x+1)/2,device=b.device,dtype=b.dtype)
                    w=torch.tensor(w/2,device=b.device,dtype=b.dtype)
                    width=(right-left).clamp_min(0)
                    means=left[...,None]+width[...,None]*x
                    weights=(width/(delta[...,1:]-delta[...,:-1]))[...,None]*w
                    safe=means.flatten(-2).clamp(1e-12,E0-1e-12);weights=weights.flatten(-2)
                    sigma=(ENERGY_VARIANCE_SLOPE*safe+s[...,None].square()).sqrt()
                    dtype=torch.float32 if gaussian_float32 else torch.float64
                    mean_eval=safe.to(dtype);sigma_eval=sigma.to(dtype)
                    logpdf=-.5*((energy.to(dtype)-mean_eval)/sigma_eval).square()-sigma_eval.log()-.5*math.log(2*math.pi)+weights.to(dtype).log()
                    total=torch.logaddexp(total,torch.logsumexp(logpdf,dim=-1).double());mass_total+=weights.sum(-1)
                    continue
                value,retained=bin_tensor_backend(compiled_bins)(energy,left,right,
                    delta[...,1:]-delta[...,:-1],s[...,None].square())
                total=torch.logaddexp(total,value);mass_total+=retained
                continue
            else:
                safe=t[...,None].clamp(1e-12,E0-1e-12);weights=torch.ones_like(safe)
                sigma=(ENERGY_VARIANCE_SLOPE*safe+s[...,None].square()).sqrt()
                logpdf=-.5*((energy-safe)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)
            total=torch.logaddexp(total,torch.logsumexp(logpdf,dim=-1))
            mass_total+=weights.sum(-1)
        if bool((mass_total==0).any()):raise ValueError('Empty physical transfer law at a grid point')
        output[chosen]=total-mass_total.log()
    if not bool(torch.isfinite(output).all()):raise ValueError('Nonfinite energy density')
    return output,beta


def normalized_proxy_response(prepared,coordinates,B,law,node_chunk=16,compiled_bins=False,backend='analytic',gaussian_float32=False):
    """Same B and measured-energy KN proxy; only angular matching factor changes.

    Row scaling is numerical/event-only. Source-dependent selection masses must
    not be divided out. S must be freshly generated from these exact rows.
    This is a calibrated response proxy, not certified physical joint p(y|x).
    """
    logp,beta=energy_log_density(prepared,coordinates,law,node_chunk=node_chunk,compiled_bins=compiled_bins,backend=backend,gaussian_float32=gaussian_float32)
    e=prepared.e1.double();kn=E0/(E0-e)+(E0-e)/E0
    logweights=logp+(kn[:,None]-beta.sin().square()).log()+B[prepared.cpnum1-1].double().log()
    norm=torch.logsumexp(logweights,dim=1,keepdim=True)
    if not bool(torch.isfinite(norm).all()):raise ValueError('Frozen event has no candidate response')
    response=(logweights-norm).exp().to(B.dtype)
    if not bool(torch.isfinite(response).all()):raise ValueError('Invalid candidate response')
    return response
