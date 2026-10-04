"""Original-model bounded near-side sampling, followed by whole-cell diagnosis."""
import argparse
import json
from pathlib import Path
import subprocess
from generate_compton_a_guard import (run_one, digest, ENGINE_REL, SOURCE_RUN, PE_HASH, SCATTER_HASH)
import numpy as np
from compton_cartesian_patch import patch_specs, refined_patch_specs
from generate_compton_a_guard import axes, NDET


def validate_common_points(new, old, name):
    """An independent grid size may not alter the original matrix model."""
    new_spec=json.loads((new/name/'complete.json').read_text())['spec']
    old_spec=json.loads((old/name/'complete.json').read_text())['spec']
    pair=[]
    for fine,coarse in zip(axes(new_spec),axes(old_spec)):
        indices=np.flatnonzero(np.min(np.abs(fine[:,None]-coarse[None,:]),axis=0)<1e-10)
        fine_indices=np.array([np.argmin(abs(fine-coarse[i])) for i in indices])
        pair.append((fine_indices,indices))
    metrics={}
    for kind in ('pe.sysmat','pe_windowed.sysmat','Scatter_SysMat','SysMat_withScatter'):
        def path(base):
            candidates=list((base/name).glob(kind+'*'))
            if len(candidates)!=1:raise ValueError('Patch matrix identity is not unique')
            return candidates[0]
        a=np.memmap(path(new),mode='r',dtype='<f4',shape=(NDET,*reversed(new_spec['shape'])))
        b=np.memmap(path(old),mode='r',dtype='<f4',shape=(NDET,*reversed(old_spec['shape'])))
        (nx,ox),(ny,oy),(nz,oz)=pair
        actual=np.asarray(a[:,nz[:,None,None],ny[None,:,None],nx[None,None,:]],dtype=float)
        expected=np.asarray(b[:,oz[:,None,None],oy[None,:,None],ox[None,None,:]],dtype=float)
        delta=actual-expected;floor=float(expected.max())*1e-8
        passed=bool(np.all(abs(delta)<=1e-5*abs(expected)+floor))
        metrics[kind]=dict(passed=passed,relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(expected),1e-30)),
            points=len(nx)*len(ny)*len(nz),maximum_absolute_error=float(abs(delta).max()))
    receipt=dict(status='PASSED' if all(r['passed'] for r in metrics.values()) else 'HOLD',metrics=metrics,
        previous_complete_sha256=digest(old/name/'complete.json'))
    (new/name/'common_grid_gate.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if receipt['status']!='PASSED':raise ValueError('Dense patch common-point regression failed')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'guard', 'field', 'geometry', 'config', 'measure', 'inputs', 'factors', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--cuda', type=int, default=0); p.add_argument('--previous-patches',type=Path)
    p.add_argument('--intermediate-patches',type=Path);a = p.parse_args()
    anchor = json.loads((a.guard/'guard_ready.json').read_text())
    if anchor['parts'][0]['anchor']['status'] != 'PASSED': raise ValueError('Common matrix anchor failed')
    engine = a.root/ENGINE_REL; source = engine/SOURCE_RUN
    pe = engine/'PEGen_RayTracing_CircularHole/PEGen_V4_Production'
    scatter = engine/'ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local'
    if digest(pe) != PE_HASH or digest(scatter) != SCATTER_HASH: raise ValueError('Original binaries differ')
    old = json.loads((source/'ELLIPSE_inputs.json').read_text())
    if any(digest(source/n) != v for n, v in old['parameters'].items()): raise ValueError('Original parameters differ')
    a.output.mkdir(parents=True, exist_ok=False)
    specs=refined_patch_specs(.375 if a.intermediate_patches else .75) if a.previous_patches else patch_specs()
    plan = dict(parts=specs, total_points=sum(int(np.prod(s['shape'])) for s in specs), original_parameters=old['parameters'],
        original_anchor_ready_sha256=digest(a.guard/'guard_ready.json'),
        purpose='Bounded near-side 3 mm versus 1.5 mm A and whole-cell integral diagnostics; no imaging/q grid changes')
    (a.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    receipts=[];common=[]
    for spec in specs:
        receipts.append(run_one(spec,a.output,source,pe,scatter,a.cuda))
        if a.previous_patches:common.append(validate_common_points(a.output,a.intermediate_patches or a.previous_patches,spec['name']))
    ready = dict(status='COMPLETE_DIAGNOSTIC_ONLY', parts=receipts, pe_binary_sha256=PE_HASH,
        scatter_binary_sha256=SCATTER_HASH, original_matrix_read_only=True, calibration_refitted=False,
        new_transport_photons=0, reconstruction_permitted=False,common_grid_gates=common)
    (a.output/'patch_ready.json').write_text(json.dumps(ready, indent=2)+'\n')
    command = [__import__('sys').executable, str(Path(__file__).with_name('validate_compton_integral_guard.py'))]
    for name in ('field', 'geometry', 'config', 'measure', 'inputs', 'factors'):
        command += ['--'+name, str(getattr(a, name))]
    command += ['--patches',str(a.previous_patches or a.output),'--patch-only','--output',str(a.output/'integrals'),'--device','cuda:0']
    if a.previous_patches:command+=['--refined-patches',str(a.intermediate_patches or a.output)]
    if a.intermediate_patches:command+=['--ultrafine-patches',str(a.output)]
    subprocess.run(command, check=True, timeout=1800)


if __name__ == '__main__': main()
