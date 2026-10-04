"""Device interpolation and bounded integration for the same physical R2 rows.

No response approximation or additional event filter. Double precision
interpolation/accumulation uses the unchanged float32 shared K. Only sampled
A rows needed by the current event block are copied to the device.
"""
import time
import numpy as np
import torch
from compton_overlap_integrator import OverlapIntegrator


class DeviceARows:
    def __init__(self,provider,crystals,device):
        self.device=device
        self.unique=np.unique(crystals)
        self.events=torch.as_tensor(np.searchsorted(self.unique,crystals),device=device)
        self.base=torch.as_tensor(np.array(provider.base[self.unique],copy=True),dtype=torch.float64,device=device)
        self.patches=[torch.as_tensor(np.array(p['values'][provider.selected[self.unique]],copy=True),
                                     dtype=torch.float64,device=device) for p in provider.patches]
        self.scales=torch.as_tensor(provider.scales[self.unique],dtype=torch.float64,device=device)
        self.bytes=sum(v.numel()*v.element_size() for v in (self.base,*self.patches,self.scales,self.events))

    def evaluate(self,compiled):
        out=torch.empty((len(self.unique),compiled['count']),dtype=torch.float64,device=self.device)
        u=torch.arange(len(self.unique),device=self.device)
        for source,(indices,cache) in compiled['groups'].items():
            indices=torch.as_tensor(indices,device=self.device)
            if source==0:
                vertices,bary,lower,t=cache
                vertices=torch.as_tensor(vertices,device=self.device)
                bary=torch.as_tensor(bary,dtype=torch.float64,device=self.device)
                lower=torch.as_tensor(lower,device=self.device)
                t=torch.as_tensor(t,dtype=torch.float64,device=self.device)
                left=(self.base[u[:,None,None],lower[None,:,None],vertices[None,:,:]]*bary[None]).sum(2)
                right=(self.base[u[:,None,None],(lower+1)[None,:,None],vertices[None,:,:]]*bary[None]).sum(2)
                values=left*(1-t)[None]+right*t[None]
            else:
                (ix,iy,iz),(tx,ty,tz)=cache
                ix,iy,iz=(torch.as_tensor(a,device=self.device) for a in (ix,iy,iz))
                tx,ty,tz=(torch.as_tensor(a,dtype=torch.float64,device=self.device) for a in (tx,ty,tz))
                field=self.patches[source-1]
                values=torch.zeros((len(u),len(ix)),dtype=torch.float64,device=self.device)
                for z in (0,1):
                    for y in (0,1):
                        for x in (0,1):
                            weight=(tx if x else 1-tx)*(ty if y else 1-ty)*(tz if z else 1-tz)
                            values+=field[u[:,None],(iz+z)[None],(iy+y)[None],(ix+x)[None]]*weight[None]
                values*=self.scales[:,None]
            if not bool(torch.isfinite(values).all()) or bool((values < -1e-20).any()):
                raise ValueError('Invalid device A interpolation; no event may be removed')
            out[:,indices]=values.clamp_min(0)
        return out[self.events]


def segment_sum(values,slots,count):
    """Contiguous deterministic segments, without floating atomic updates.

    Cumulative double precision sums are checked against the independent CPU
    quadrature path on real rows before any production use.
    """
    changes=np.r_[0,np.flatnonzero(np.diff(slots))+1,len(slots)]
    if np.any(np.diff(slots)<0):raise ValueError('Cell domains must remain contiguous')
    prefix=torch.cumsum(values,dim=1)
    begin=torch.as_tensor(changes[:-1],device=values.device)
    end=torch.as_tensor(changes[1:]-1,device=values.device)
    subtract=prefix[:,(begin-1).clamp_min(0)]
    subtract=torch.where((begin==0)[None],torch.zeros_like(subtract),subtract)
    answer=torch.zeros((values.shape[0],count),dtype=torch.float64,device=values.device)
    answer[:,torch.as_tensor(slots[changes[:-1]],device=values.device)]=prefix[:,end]-subtract
    return answer


class DeviceOverlapIntegrator(OverlapIntegrator):
    def integrate(self,prepared,view,order=(16,8,8),*,progress=None):
        if not 0<=view<20 or len(order)!=3 or min(order)<1 or prepared.count<1:
            raise ValueError('Invalid accepted block/view/quadrature')
        start=time.monotonic();device=prepared.e1.device
        cp=prepared.cpnum1.detach().cpu().numpy().astype(int)-1
        sampled=DeviceARows(self.provider,cp,device)
        integrals=torch.zeros((prepared.count,len(self.partial),2),dtype=torch.float64,device=device)
        total_nodes=0;maximum_chunk=0;field_counts={}
        for offset in range(0,len(self.partial),self.cells_per_block):
            p,w,slots,compiled=self._block(view,order,offset)
            count=min(self.cells_per_block,len(self.partial)-offset)
            response=sampled.evaluate(compiled)
            result=torch.zeros((prepared.count,2*count),dtype=torch.float64,device=device)
            weights=torch.as_tensor(w,dtype=torch.float64,device=device)
            for lo in range(0,len(w),self.node_chunk):
                hi=min(lo+self.node_chunk,len(w));maximum_chunk=max(maximum_chunk,hi-lo)
                xyz=torch.as_tensor(p[lo:hi],dtype=torch.float64,device=device)
                k=self.kernel(prepared,xyz,self.settings).double()
                if k.shape!=(prepared.count,hi-lo) or not bool(torch.isfinite(k).all()) or bool((k<0).any()):
                    raise ValueError('Invalid shared event kernel; HOLD')
                result+=segment_sum(k*response[:,lo:hi]*weights[None,lo:hi],slots[lo:hi],2*count)
            integrals[:,offset:offset+count]=result.reshape(prepared.count,count,2)
            total_nodes+=len(w)
            for name,value in self.provider.field_counts(compiled).items():
                field_counts[name]=field_counts.get(name,0)+int(value)
            if progress is not None:
                progress(dict(completed_cells=offset+count,total_cells=len(self.partial),
                              elapsed_seconds=time.monotonic()-start,nodes=total_nodes))
        if not bool(torch.isfinite(integrals).all()) or bool((integrals<0).any()):
            raise ValueError('Invalid device-integrated rows; HOLD')
        self.statistics=dict(events=prepared.count,partial_cells=len(self.partial),view_zero_based=int(view),order=list(order),
            quadrature_nodes=total_nodes,maximum_point_chunk=maximum_chunk,cache_bytes=self.cache_size,cache_limit_bytes=self.limit,
            provider_node_counts=field_counts,elapsed_seconds=time.monotonic()-start,diagnostic_only=self.diagnostic_only,
            execution='device_A_and_double_segment_sum',device_sampled_A_bytes=sampled.bytes,
            event_filter_applied=False,density_volume_applied_once=True)
        return integrals[:,:,0].contiguous(),integrals[:,:,1].contiguous()
