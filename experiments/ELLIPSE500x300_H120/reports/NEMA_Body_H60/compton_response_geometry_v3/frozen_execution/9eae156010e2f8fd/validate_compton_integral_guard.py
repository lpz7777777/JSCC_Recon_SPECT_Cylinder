"""Whole object/full-reference integrals on held-out events and sampled A.

The optional dense near-side patch separates A interpolation error from
quadrature error. No event selection, q grid, imaging operator or S is changed.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from geometry import grid
from generate_compton_a_guard import digest, axes
from build_compton_a_guard_field import combined
from compton_boundary_quadrature import cell_quadrature, rotate_to_detector, GuardedPolarResponseField
from compton_cartesian_patch import CartesianPatch
from compton_event_response import (ComptonEventSettings, build_detector_position_variance,
    prepare_compton_events, build_compton_cone_weights, min_standardized_compton_arm)
from detector_csv import load_detector_coordinates


def control_cells(coords, fraction):
    partial = np.flatnonzero((fraction > 1e-12) & (fraction < 1-1e-12))
    ordered = partial[np.argsort(fraction[partial])]
    controls = set(ordered[np.linspace(0, len(ordered)-1, 8, dtype=int)].tolist())
    angle = np.arctan2(coords[partial, 1], coords[partial, 0])
    for target in np.arange(8)*np.pi/4:
        controls.add(int(partial[np.argmin(np.abs(np.angle(np.exp(1j*(angle-target)))))]))
    return sorted(controls)


def write_cases(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('field', 'geometry', 'config', 'measure', 'inputs', 'factors', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--patches', type=Path); p.add_argument('--patch-only', action='store_true')
    p.add_argument('--device', default='cuda:0'); a = p.parse_args()
    if a.patch_only and not a.patches: raise ValueError('Dense patches required')
    cfg = json.loads(a.config.read_text()); geo = np.load(a.geometry)
    fm = json.loads((a.field/'field_manifest.json').read_text())
    mm = json.loads((a.measure/'measure_manifest.json').read_text())
    if (digest(a.geometry) != fm['baseline_geometry_sha256'] or
        digest(a.geometry) != mm['baseline_geometry_sha256'] or mm['status'] != 'PASSED' or
        digest(a.measure/'measure.npz') != mm['measure_sha256'] or
        digest(a.field/'A_field.float64') != fm['A_field_sha256'] or
        digest(a.field/'field_geometry.npz') != fm['field_geometry_sha256']):
        raise ValueError('Frozen field/geometry/precise measure identity differs')
    a.output.mkdir(parents=True, exist_ok=False)
    coords, cells, _ = grid(cfg); n = cfg['points_per_layer']
    np.testing.assert_allclose(coords, geo['coordinates_mm'], rtol=0, atol=1e-10)
    controls = control_cells(coords, geo['ellipse_fraction'][:n])
    if a.patch_only: controls = [3161, 3231]
    measures = np.load(a.measure/'measure.npz')['effective_volume_mm3']
    fg = np.load(a.field/'field_geometry.npz')
    field = np.memmap(a.field/'A_field.float64', mode='r', dtype='<f8', shape=tuple(fm['shape']))
    interp = GuardedPolarResponseField(fg['xy_mm'], int(fg['original_points']), fg['z_mm'])
    patches = {}
    if a.patches:
        receipt = json.loads((a.patches/'patch_ready.json').read_text())
        if receipt['status'] != 'COMPLETE_DIAGNOSTIC_ONLY': raise ValueError('Patch generation incomplete')
        for layer, name in ((0, 'near_minus'), (20, 'near_middle'), (39, 'near_plus')):
            raw = combined(a.patches, name)
            spec = json.loads((a.patches/name/'complete.json').read_text())['spec']
            xyz = axes(spec)
            patches[layer] = (raw, CartesianPatch(xyz), CartesianPatch(tuple(v[::2] for v in xyz)))
    torch.set_num_threads(8); torch.set_grad_enabled(False); device = torch.device(a.device)
    if device.type == 'cuda': torch.cuda.set_device(device)
    factor = a.factors/'440keV_RotateNum20'
    detector = torch.tensor(load_detector_coordinates(factor/'Detector.csv', 10496), device=device)
    var = build_detector_position_variance(detector, 0)
    settings = ComptonEventSettings(.440, .13*np.sqrt(511/440),
        2*.440**2/(.511+2*.440)-.001, .05, .35, geometry_mode='stable_float64')
    B = np.memmap(factor/'SysMat_polar', mode='r', dtype='<f4', shape=(132040, 10496))
    allcoords = torch.tensor(geo['coordinates_mm'], dtype=torch.float32, device=device)
    rows = []; events = []; caches = {}; start = time.monotonic()
    orders = ((8, 4, 4), (16, 8, 8), (32, 12, 12))
    volume_errors = []; support_checks = 0
    for folder in sorted(a.inputs.glob('point_*')):
        for view in (1, 6, 11, 16):
            if a.patch_only and view not in (6, 16): continue
            path = folder/f'ideal_v{view:02d}.csv'
            raw = np.loadtxt(path, delimiter=',', usecols=(0, 1, 2, 3), dtype=np.float32, ndmin=2)
            prepared = None
            for row_id in range(min(len(raw), 32)):
                candidate, _ = prepare_compton_events(torch.tensor(raw[row_id:row_id+1], device=device),
                    settings, detector, var, var, input_energies_already_smeared=True)
                if candidate is not None and float(min_standardized_compton_arm(candidate, allcoords, settings)) <= 3:
                    prepared = candidate; break
            if prepared is None: continue
            c1 = int(prepared.cpnum1[0])-1
            norm = float((build_compton_cone_weights(prepared, allcoords, settings)[0].double()*
                         torch.tensor(np.array(B[:, c1]), device=device)).sum())
            if not np.isfinite(norm) or norm <= 0: raise ValueError('Diagnostic baseline reference invalid')
            events.append(dict(dataset=folder.name, view=view, input_row=row_id, input_sha256=digest(path)))
            for layer in (0, 20, 39):
                for cell in controls:
                    # This predetermined full-cell patch faces the detector at these two views.
                    patch_case = bool(a.patches and ((cell == 3161 and view == 16) or (cell == 3231 and view == 6)))
                    if a.patch_only and not patch_case: continue
                    for domain in ('object_intersection', 'full_reference_cell'):
                        integral = []; patch3 = None; patch15 = None
                        for oi, order in enumerate(orders):
                            key = (view, layer, cell, domain, oi)
                            if key not in caches:
                                nodes, w = cell_quadrature(cells[cell], coords[layer*n+cell, 2], *order,
                                    ellipse=domain == 'object_intersection')
                                nodes = rotate_to_detector(nodes, view-1)
                                caches[key] = (nodes, w, interp.cache(nodes))
                                if oi == 2:
                                    reference = measures[layer*n+cell] if domain == 'object_intersection' else geo['cell_volume_mm3'][layer*n+cell]
                                    volume_errors.append(abs(float(w.sum()/reference-1))); support_checks += 1
                            nodes, w, cache = caches[key]
                            k = build_compton_cone_weights(prepared,
                                torch.tensor(nodes, dtype=torch.float64, device=device), settings)[0].double().cpu().numpy()
                            spatial = interp.evaluate(field[c1], cache)
                            integral.append(float(np.dot(k*spatial, w)))
                            if patch_case and oi == 2:
                                actual, fine, coarse = patches[layer]
                                selected = int(fg['selected_raw_detectors'][c1]); scale = fm['calibration_scales'][c1]
                                values = np.asarray(actual[selected], dtype=float)*scale
                                patch15 = float(np.dot(k*fine.evaluate(values, fine.cache(nodes)), w))
                                patch3 = float(np.dot(k*coarse.evaluate(values[::2, ::2, ::2], coarse.cache(nodes)), w))
                        floor = 1e-10*norm
                        passed = abs(integral[2]-integral[1]) <= .01*abs(integral[2])+floor
                        refinement = None if patch15 is None else abs(patch15-patch3) <= .01*abs(patch15)+floor
                        rows.append(dict(dataset=folder.name, view=view, input_row=row_id, c1=c1+1,
                            c2=int(prepared.cpnum2[0]), layer=layer, cell=cell, domain=domain,
                            coarse=integral[0], medium=integral[1], fine=integral[2], reference_norm=norm,
                            quadrature_relative_change=float(abs(integral[2]-integral[1])/max(abs(integral[2]), floor)),
                            quadrature_passed=passed, patch_3mm=patch3, patch_1p5mm=patch15,
                            patch_refinement_passed=refinement,
                            patch_refinement_relative_change=None if patch15 is None else float(abs(patch15-patch3)/max(abs(patch15), floor)),
                            guard_to_patch_relative_change=None if patch15 is None else float((integral[2]-patch15)/max(abs(patch15), floor)),
                            guard_to_patch_absolute_over_reference=None if patch15 is None else float((integral[2]-patch15)/norm)))
            write_cases(a.output/'integral_cases.csv', rows)
            print(json.dumps(dict(dataset=folder.name, view=view, cases=len(rows), elapsed=time.monotonic()-start)), flush=True)
    qok = bool(rows) and all(r['quadrature_passed'] for r in rows)
    pok = all(r['patch_refinement_passed'] for r in rows if r['patch_refinement_passed'] is not None)
    gate = dict(status='QUADRATURE_PASSED_A_ACCURACY_PENDING' if qok and pok else 'HOLD',
        geometry_sha256=digest(a.geometry), field_manifest_sha256=digest(a.field/'field_manifest.json'),
        measure_manifest_sha256=digest(a.measure/'measure_manifest.json'), kernel_sha256=digest(Path(__import__('compton_event_response').__file__)),
        predefined_cells=controls, layers=[0, 20, 39], events=events, cases=len(rows),
        quadrature_passed=sum(r['quadrature_passed'] for r in rows),
        maximum_quadrature_relative_change=max(r['quadrature_relative_change'] for r in rows),
        maximum_volume_relative_error=max(volume_errors), sampled_support_checks=support_checks,
        patch_cases=sum(r['patch_1p5mm'] is not None for r in rows),
        patch_refinement_passed=sum(r['patch_refinement_passed'] is True for r in rows),
        interpretation='Same frozen stable-q eligible held-out events; object and full-cell K*A integrals. Diagnostic baseline Z is common to all comparisons, not the new mixed Z or a matched S2. Quadrature convergence alone does not certify A accuracy or permit imaging.',
        output_sha256=digest(a.output/'integral_cases.csv'), new_transport_photons=0,
        reconstruction_permitted=False, elapsed_seconds=time.monotonic()-start)
    if a.patches: gate['patch_ready_sha256'] = digest(a.patches/'patch_ready.json')
    (a.output/'integral_gate.json').write_text(json.dumps(gate, indent=2)+'\n')
    print('INTEGRAL_DIAGNOSTIC', gate['status'], flush=True)


if __name__ == '__main__': main()
