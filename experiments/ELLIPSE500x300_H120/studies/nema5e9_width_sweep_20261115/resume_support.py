"""Read-only parent binding and copy-once checkpoint inheritance for r5."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def checked_bytes(root,name,manifest):
    raw=(root/name).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest[name]:raise ValueError('Parent SHA changed: '+name)
    return raw

def verify_parent(plan):
    parent=Path(plan['parent_result'])
    for name,digest in plan['parent_files'].items():
        if sha(parent/name)!=digest:raise ValueError('Parent result changed: '+name)
    release=Path(plan['parent_release'])
    for name,digest in plan['parent_release_files'].items():
        if sha(release/name)!=digest:raise ValueError('Parent source changed: '+name)
    if (parent/'width80').exists() or (parent/'width60').exists():
        raise ValueError('Previously unseen candidate output; cannot rerun')

def inherit_branch(plan,out,rank,device):
    """Load final state on every rank, copy saved history only on rank zero."""
    root=Path(plan['parent_result']);files=plan['parent_files'];frames=[];checksums={}
    stop=plan['completed_iterations'];n=plan['active_points'];step=plan['save_step']
    if stop!=9950 or step!=50 or n!=78920:raise ValueError('Unexpected resume contract')
    for iteration in range(step,stop+1,step):
        name=f'checkpoints/iteration_{iteration:05d}.float32'
        if rank==0 or iteration==stop:
            raw=checked_bytes(root,'width100/'+name,files)
            frame=np.frombuffer(raw,dtype='<f4').copy().reshape(n,1)
            if not np.isfinite(frame).all() or (frame<0).any():raise ValueError('Invalid parent density')
            if rank==0:
                dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
                with dest.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
                frames.append(torch.from_numpy(frame));checksums[name]=files['width100/'+name]
            if iteration==stop:initial=torch.tensor(frame,device=device)
    if rank==0:
        for name in files:
            leaf=Path(name).name
            if name.startswith('width100/') and (leaf.startswith('benchmark_') or leaf=='original_regression_00050.json'):
                raw=checked_bytes(root,name,files)
                with (out/leaf).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    return initial,frames,checksums

def inherited_sensitivity(plan,computed,rank,output):
    raw=checked_bytes(Path(plan['parent_result']),'width100/S_SC_active.float32',plan['parent_files'])
    saved=torch.tensor(np.frombuffer(raw,dtype='<f4').copy().reshape(-1,1),device=computed.device)
    rel=float(torch.linalg.vector_norm((saved-computed).double())/torch.linalg.vector_norm(saved.double()))
    if not np.isfinite(rel) or rel>1e-5:raise ValueError('SC sensitivity no longer matches source operator')
    if rank==0:
        with (output/'SC_sensitivity_resume_check.json').open('x') as f:
            json.dump(dict(relative_L2=rel,threshold=1e-5,bitwise=torch.equal(saved,computed),
                           used='Exact inherited S for all three widths',parent_job=plan['parent_job']),f,indent=2)
    return saved
