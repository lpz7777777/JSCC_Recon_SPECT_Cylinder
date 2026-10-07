"""Read-only JSCC window counts and own sensitivity; no execution or data edits.

Each complete matrix is read once: the bytes hashed are those used for row sums.
Only new evidence in the independent EHE experiment is written.
"""
import argparse,hashlib
from pathlib import Path
import numpy as np
from ehe_common import digest,read,write,array_write,hashes

FOLDERS={'A218':'218keV_RotateNum20','A440':'440keV_RotateNum20','C440to218':'440keV_to218win_RotateNum20'}

def sensitivity_statistics(s,volume):
    s=np.asarray(s,np.float64);volume=np.asarray(volume,np.float64)
    if s.shape!=volume.shape or np.any(~np.isfinite(s)) or np.any(s<=0) or np.any(volume<=0):raise ValueError('Invalid whole-cell sensitivity statistics')
    def stats(v):return dict(min=float(v.min()),max=float(v.max()),mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),p99=float(np.quantile(v,.99)))
    return dict(density_response_mm3=stats(s),cell_volume_normalized_detection_probability=stats(s/volume))

def build_reference(release,factors,inputs,output):
    cfg=read(release/'contract.json');g=np.load(release/'whole_geometry.npz')
    if digest(release/'whole_geometry.npz')!=cfg['whole_geometry_sha256']:raise ValueError('Accepted JSCC geometry changed')
    active=g['active_indices'];inverse=g['inverse_rotation'];volume=g['cell_volume_mm3'][active]
    if len(active)!=78920 or inverse.shape!=(132040,20):raise ValueError('Accepted JSCC whole basis differs')
    output.mkdir(parents=True,exist_ok=False);summary={};window_counts={}
    for name,folder in FOLDERS.items():
        path=factors/folder/'SysMat_polar';relative=folder+'/SysMat_polar'
        if path.stat().st_size!=132040*10496*4:raise ValueError('Accepted JSCC matrix shape differs')
        raw=np.memmap(path,'<f4',mode='r',shape=(132040,10496));sums=np.empty(132040,np.float64);h=hashlib.sha256()
        for start in range(0,132040,512):
            block=raw[start:start+512];h.update(memoryview(block))
            if np.any(~np.isfinite(block)) or np.any(block<0):raise ValueError('Accepted JSCC matrix invalid')
            sums[start:start+len(block)]=block.sum(axis=1,dtype=np.float64)
        if h.hexdigest()!=cfg['factor_payload_sha256'][relative]:raise ValueError('Accepted complete JSCC matrix SHA differs')
        s=np.mean(sums[inverse[active]],axis=1);array_write(output/(name+'_S_active.float64'),s)
        summary[name]=dict(matrix_sha256=h.hexdigest(),**sensitivity_statistics(s,volume));del raw
        print('READ_ONLY_JSCC_MEASUREMENT',name,flush=True)
    for e in (218,440):
        relative=f'CntStat/{e}keV_RotateNum20_Geant4JSCC/CntStat_NEMA_Body_H60_5e9.csv';path=inputs/relative
        if digest(path)!=cfg['input_sha256'][relative]:raise ValueError('Accepted JSCC observations changed')
        counts=np.loadtxt(path,delimiter=',',dtype=np.float64)
        if counts.shape!=(20,10496) or np.any(~np.isfinite(counts)) or np.any(counts<0) or not np.array_equal(counts,np.rint(counts)):raise ValueError('Actual JSCC count table invalid')
        window_counts[str(e)]=dict(total=int(counts.sum()),by_view=counts.sum(axis=1).astype(np.int64).tolist(),sha256=digest(path))
    write(output/'measurement_summary.json',dict(passed=True,reference_job=1669255,views=20,active_cells=78920,
        sensitivity=summary,actual_window_counts=window_counts,primary_tagged_jscc_window_counts='not recorded in legacy transport; do not infer actual fraction from an image',
        source_sha256=digest(__file__),contract_sha256=digest(release/'contract.json'),geometry_sha256=digest(release/'whole_geometry.npz'),files=hashes(output)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--factors',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build_reference(a.release,a.factors,a.inputs,a.output)
