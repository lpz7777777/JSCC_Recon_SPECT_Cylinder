"""Original-model bounded near-side sampling, followed by whole-cell diagnosis."""
import argparse
import json
from pathlib import Path
import subprocess
from generate_compton_a_guard import (run_one, digest, ENGINE_REL, SOURCE_RUN, PE_HASH, SCATTER_HASH)
from compton_cartesian_patch import patch_specs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'guard', 'field', 'geometry', 'config', 'measure', 'inputs', 'factors', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--cuda', type=int, default=0); a = p.parse_args()
    anchor = json.loads((a.guard/'guard_ready.json').read_text())
    if anchor['parts'][0]['anchor']['status'] != 'PASSED': raise ValueError('Common matrix anchor failed')
    engine = a.root/ENGINE_REL; source = engine/SOURCE_RUN
    pe = engine/'PEGen_RayTracing_CircularHole/PEGen_V4_Production'
    scatter = engine/'ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local'
    if digest(pe) != PE_HASH or digest(scatter) != SCATTER_HASH: raise ValueError('Original binaries differ')
    old = json.loads((source/'ELLIPSE_inputs.json').read_text())
    if any(digest(source/n) != v for n, v in old['parameters'].items()): raise ValueError('Original parameters differ')
    a.output.mkdir(parents=True, exist_ok=False)
    plan = dict(parts=patch_specs(), total_points=2295, original_parameters=old['parameters'],
        original_anchor_ready_sha256=digest(a.guard/'guard_ready.json'),
        purpose='Bounded near-side 3 mm versus 1.5 mm A and whole-cell integral diagnostics; no imaging/q grid changes')
    (a.output/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    receipts = [run_one(s, a.output, source, pe, scatter, a.cuda) for s in patch_specs()]
    ready = dict(status='COMPLETE_DIAGNOSTIC_ONLY', parts=receipts, pe_binary_sha256=PE_HASH,
        scatter_binary_sha256=SCATTER_HASH, original_matrix_read_only=True, calibration_refitted=False,
        new_transport_photons=0, reconstruction_permitted=False)
    (a.output/'patch_ready.json').write_text(json.dumps(ready, indent=2)+'\n')
    command = [__import__('sys').executable, str(Path(__file__).with_name('validate_compton_integral_guard.py'))]
    for name in ('field', 'geometry', 'config', 'measure', 'inputs', 'factors'):
        command += ['--'+name, str(getattr(a, name))]
    command += ['--patches', str(a.output), '--patch-only', '--output', str(a.output/'integrals'), '--device', 'cuda:0']
    subprocess.run(command, check=True, timeout=1800)


if __name__ == '__main__': main()
