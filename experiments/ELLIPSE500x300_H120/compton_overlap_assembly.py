"""R2 mixed-reference assembly, shared by imaging and matched sensitivity.

Inputs are unnormalised K*B rows and separately integrated K*A*dV. This layer
does not select events or compute quadrature. It must not receive previously
normalised rows, nor apply f/volume again to intersection integrals.
"""
import numpy as np
import torch


class OverlapAssembly:
    def __init__(self, active, partial, inverse_rotation, full_count):
        self.active=np.asarray(active,dtype=np.int64)
        self.partial=np.asarray(partial,dtype=np.int64)
        self.inverse=np.asarray(inverse_rotation,dtype=np.int64)
        self.full_count=full_count
        if self.inverse.ndim!=2 or self.inverse.shape[0]!=full_count:
            raise ValueError('Rotation table does not match full circle')
        for indices in (self.active,self.partial):
            if len(np.unique(indices))!=len(indices) or np.any(np.diff(indices)<=0):
                raise ValueError('Cell identities must be unique and sorted')
            if indices.min()<0 or indices.max()>=full_count:raise ValueError('Invalid full-grid index')
        position=np.searchsorted(self.active,self.partial)
        if np.any(position>=len(self.active)) or not np.array_equal(self.active[position],self.partial):
            raise ValueError('Partial cells must occur in active columns')
        self.partial_position=position
        for view in range(self.inverse.shape[1]):
            if not np.array_equal(np.sort(self.inverse[:,view]),np.arange(full_count)):
                raise ValueError('Rotation is not an integer permutation')

    def assemble(self, raw_full_kb, object_integrals, reference_integrals, view,
                 *, input_layout='unnormalized_full_circle'):
        if input_layout!='unnormalized_full_circle':raise ValueError('R2 requires raw full-circle K*B')
        if not 0<=view<self.inverse.shape[1]:raise ValueError('Unknown view')
        if raw_full_kb.ndim!=2 or raw_full_kb.shape[1]!=self.full_count:
            raise ValueError('Wrong complete-circle response shape')
        shape=(raw_full_kb.shape[0],len(self.partial))
        if object_integrals.shape!=shape or reference_integrals.shape!=shape:
            raise ValueError('Object and full-cell integrals must share partial cell identities')
        for values in (raw_full_kb,object_integrals,reference_integrals):
            if values.device!=raw_full_kb.device or not bool(torch.isfinite(values).all()) or bool((values<0).any()):
                raise ValueError('Invalid response values or device')
        device=raw_full_kb.device
        rotated=torch.tensor(self.inverse[self.partial,view],device=device,dtype=torch.long)
        keep=torch.ones(self.full_count,device=device,dtype=torch.bool);keep[rotated]=False
        # Sum the remaining bins directly instead of subtracting two large
        # near-equal sums. Include the full modified cells outside the ellipse.
        reference=(raw_full_kb[:,keep].double().sum(1)+reference_integrals.double().sum(1))
        if bool((reference<=0).any()):raise ValueError('Zero complete reference; event identities cannot be silently dropped')
        active=torch.tensor(self.inverse[self.active,view],device=device,dtype=torch.long)
        output=raw_full_kb.index_select(1,active).double().clone()
        positions=torch.tensor(self.partial_position,device=device,dtype=torch.long)
        output[:,positions]=object_integrals.double()
        output/=reference[:,None]
        if bool((output.sum(1)<=0).any()):
            raise ValueError('Accepted event has zero active response; HOLD, do not remove it')
        if not bool(torch.isfinite(output).all()):raise ValueError('Nonfinite integrated response')
        return output.float(),reference


def sensitivity_contribution(integrated_rows, emitted_primaries, circle_volume, views):
    """Contribution of one event/view block to object-active S2.

    Caller must combine all 20 correlated rotations of each MC worker before
    computing uncertainties, and sum all blocks exactly once.
    """
    if emitted_primaries<=0 or circle_volume<=0 or views<=0:raise ValueError('Invalid emission denominator')
    if integrated_rows.ndim!=2 or not bool(torch.isfinite(integrated_rows).all()) or bool((integrated_rows<0).any()):
        raise ValueError('Invalid integrated rows')
    return integrated_rows.double().sum(0)*(circle_volume/emitted_primaries/views)
