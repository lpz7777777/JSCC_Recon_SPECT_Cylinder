"""Isolated whole-polar-cell support and numerical operator checks.

Produces no Factors and never overwrites the existing fraction geometry.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
from torch_active_operator import ActiveGeometry

HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def generate():
    source=HERE/'generated/Geometry/geometry.npz'
    output=HERE/'generated/process_list_global_audit_v4/WholeCellGeometry'
    output.mkdir(parents=True,exist_ok=False)
    report=HERE/'reports/NEMA_Body_H60/process_list_global_audit_v4'
    with np.load(source) as old:arrays={k:old[k] for k in old.files}
    xyz=arrays['coordinates_mm'];volume=arrays['cell_volume_mm3']
    binary=((xyz[:,0]/250)**2+(xyz[:,1]/150)**2<=1+1e-12).astype(np.float64)
    active=np.flatnonzero(binary).astype(np.int32)
    arrays['ellipse_fraction']=binary;arrays['active_indices']=active
    np.savez_compressed(output/'geometry.npz',**arrays)
    g=ActiveGeometry(active,binary,arrays['inverse_rotation'])
    # Independent deterministic measurements: unequal physical cell volumes.
    rng=np.random.default_rng(51005001)
    B=torch.tensor(rng.random((3,len(xyz)))*volume[None],dtype=torch.float64)
    image=torch.tensor(rng.random((len(active),1)),dtype=torch.float64)
    measurements=torch.tensor(rng.random((3,1)),dtype=torch.float64)
    errors=[]
    for view in range(20):
        prediction=g.forward(B,image,view)
        weights=g.adjoint(B,measurements,view)
        lhs=float((prediction*measurements).sum());rhs=float((image*weights).sum())
        errors.append(abs(lhs-rhs)/max(abs(lhs),abs(rhs)))
        np.testing.assert_allclose(g.compact(B,view).numpy(),B.numpy()[:,arrays['inverse_rotation'][active,view]],rtol=0,atol=0)
    S=torch.arange(len(xyz),dtype=torch.float64)
    np.testing.assert_array_equal(g.compton_sensitivity(S).numpy().ravel(),S.numpy()[active])
    # Event constants cancel within R_ej / (R_e rho); physical S is fixed.
    R=g.compact(B,7);scale=torch.tensor([[.1],[3.],[100.]],dtype=torch.float64)
    weight=R.T@(1/(R@image));scaled=(R*scale).T@(1/((R*scale)@image))
    scale_error=float(torch.linalg.norm(weight-scaled)/torch.linalg.norm(weight))
    chunks=sum((part.T@(1/(part@image)) for part in R.split(1)),torch.zeros_like(image))
    chunk_error=float(torch.linalg.norm(weight-chunks)/torch.linalg.norm(weight))
    nominal=np.pi*250*150*120;actual=float(volume[active].sum())
    receipt=dict(study='process_list_global_audit_v4',geometry_basis='union_of_complete_polar_cells',
        full_points=len(xyz),active_points=len(active),layers=40,views=20,
        source_geometry_sha256=digest(source),geometry_sha256=digest(output/'geometry.npz'),
        nominal_ellipse_volume_mm3=nominal,actual_whole_cell_volume_mm3=actual,
        relative_volume_difference=actual/nominal-1,minimum_active_volume_mm3=float(volume[active].min()),
        partial_overlap_applied=False,B_basis='A times full cell volume; volume multiplied exactly once',
        existing_production_geometry_changed=False,existing_production_kernel_changed=False,
        max_forward_adjoint_relative_error=max(errors),event_constant_scaling_relative_error=scale_error,
        exact_event_chunking_relative_error=chunk_error,
        limitations='Synthetic operator tests only; not the 50-iteration production regression or independent physical validation')
    assert max(errors)<1e-5 and scale_error<1e-5 and chunk_error<1e-5
    (output/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (report/'whole_cell_implementation_checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':generate()
