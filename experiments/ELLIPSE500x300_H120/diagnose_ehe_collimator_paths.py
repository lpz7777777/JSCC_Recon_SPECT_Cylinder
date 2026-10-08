"""Conditional Pb chord witnesses from frozen geometry; no response or event generation."""
import math
import re
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from ehe_common import DATA, REPORT, TRUTH, digest, read, write


def box_interval(start, end, lower, upper):
    direction = end - start
    lo, hi = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-14:
            assert lower[axis] <= start[axis] <= upper[axis]
            continue
        a, b = sorted([(lower[axis]-start[axis])/direction[axis],
                       (upper[axis]-start[axis])/direction[axis]])
        lo, hi = max(lo, a), min(hi, b)
    assert hi > lo
    return lo, hi


def chords(start, end, lower, upper, holes, origin):
    direction = end-start
    distance = float(np.linalg.norm(direction))
    lo, hi = box_interval(start, end, lower, upper)
    radial_a = direction[0]**2+direction[2]**2
    assert radial_a > 0
    intervals = []
    for index, h in enumerate(holes):
        px, pz = start[0]-h[0], start[2]-h[3]
        b = 2*(px*direction[0]+pz*direction[2])
        c = px*px+pz*pz-h[4]*h[4]
        discriminant = b*b-4*radial_a*c
        if discriminant <= 0:
            continue
        a = (-b-math.sqrt(discriminant))/(2*radial_a)
        z = (-b+math.sqrt(discriminant))/(2*radial_a)
        y1, y2 = sorted([(origin+h[1]-start[1])/direction[1],
                         (origin+h[2]-start[1])/direction[1]])
        a, z = max(a, y1, lo), min(z, y2, hi)
        if z > a:
            intervals.append((a, z, index))
    merged = []
    for a, b, index in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    box_length = (hi-lo)*distance
    void = sum(b-a for a, b in merged)*distance

    # Separate point-membership quadrature checks every finite cylinder directly.
    # This checks radial roots/union lengths, not response physics or GPU flags.
    count = 40000
    hole_points = 0
    for begin in range(0, count, 1000):
        t = lo+(np.arange(begin, min(begin+1000, count))+.5)/count*(hi-lo)
        pts = start[None, :]+t[:, None]*direction[None, :]
        radial = ((pts[:, None, 0]-holes[None, :, 0])**2
                  +(pts[:, None, 2]-holes[None, :, 3])**2)
        inside = ((radial <= holes[None, :, 4]**2)
                  &(pts[:, None, 1] >= origin+holes[None, :, 1])
                  &(pts[:, None, 1] <= origin+holes[None, :, 2]))
        hole_points += int(np.any(inside, axis=1).sum())
    sampled_void = box_length*hole_points/count
    # Each disjoint interval contributes at most two sample cells of discrepancy.
    membership_bound = 2*max(1, len(intervals))*box_length/count
    assert abs(sampled_void-void) <= membership_bound+1e-7
    assert -1e-7 <= void <= box_length+1e-7
    return dict(box_length_mm=box_length,finite_hole_void_length_mm=void,
                pb_length_mm=box_length-void,
                intersected_hole_indices=[i for a,b,i in intervals],
                sampled_void_length_mm=sampled_void,
                midpoint_membership_discrepancy_mm=abs(sampled_void-void),
                midpoint_discretization_bound_mm=membership_bound,
                midpoint_samples=count)


def acceptance(mean, resolution, source_energy, lower, upper):
    sigma = resolution*math.sqrt(source_energy/mean)*mean/2.35482
    a, b = (lower-mean)/(math.sqrt(2)*sigma), (upper-mean)/(math.sqrt(2)*sigma)
    if a >= 0: return .5*(math.erfc(a)-math.erfc(b))
    if b <= 0: return .5*(math.erfc(-b)-math.erfc(-a))
    return .5*(math.erf(b)-math.erf(a))


def kinematics(start, scatter, target, energy, resolution, window):
    incoming, outgoing = scatter-start, target-scatter
    cosine = float(np.dot(incoming, outgoing)/np.linalg.norm(incoming)/np.linalg.norm(outgoing))
    angle = math.acos(max(-1,min(1,cosine)))
    after = energy/(1+energy/511*(1-math.cos(angle)))
    sigma = resolution*math.sqrt(energy/after)*after/2.35482
    return dict(angle_degrees=math.degrees(angle),scattered_energy_keV=after,
                conditional_full_deposit_window_probability=acceptance(after,resolution,energy,*window),
                two_sigma_prefilter_rejects=bool(after+2*sigma <= window[0] or after-2*sigma >= window[1]))


def audit():
    began = time.monotonic()
    freeze = read(REPORT/'response_repair_freeze.json')
    payload = DATA/freeze['payload_dir']
    bindings = {}
    names = ['engine/ScatterGen_RayTracing_CircularHole/scatter.cu',
             'engine/physics_data/nist_xcom_materials_1_1000keV.h']
    names += ['params/C440to218/'+n for n in ['Params_Collimator.dat','Params_Detector.dat',
                                           'Params_Image.dat','Params_Physics.dat']]
    for name in names:
        assert digest(payload/name) == freeze['sha256'][name]
        bindings[name] = digest(payload/name)
    gate_sha = digest(REPORT/'physical_gate.json')
    gate = read(REPORT/'physical_gate.json')
    assert not gate['passed'] and digest(TRUTH) == gate['source_sha256']
    p = payload/'params/C440to218'
    collimator = np.fromfile(p/'Params_Collimator.dat','<f4').astype(float)
    detector = np.fromfile(p/'Params_Detector.dat','<f4')[1:].reshape(2312,12).astype(float)
    image = np.fromfile(p/'Params_Image.dat','<f4').astype(float)
    physics = np.fromfile(p/'Params_Physics.dat','<f4').astype(float)
    assert collimator[0] == 1 and collimator[10] == 1250 and physics[7] == 440
    holes = collimator[100:].reshape(1250,9)
    origin, thickness = image[11], collimator[12]
    lower = np.array([-collimator[11]/2,origin+collimator[14]-thickness/2,-collimator[13]/2])
    upper = np.array([collimator[11]/2,origin+collimator[14]+thickness/2,collimator[13]/2])
    hidx = int(np.argmin(holes[:,0]**2+holes[:,3]**2))
    h = holes[hidx]
    truth = np.load(TRUTH)
    xyz_indices = [int(np.argmin(abs(truth[n]-value)))
                   for n,value in zip(['x_mm','y_mm','z_mm'],[h[0],-100.5,-1.5])]
    start = np.array([truth[n][i] for n,i in zip(['x_mm','y_mm','z_mm'],xyz_indices)],dtype=float)
    density = float(truth['activity_440_zyx'][tuple(xyz_indices[::-1])])
    assert density > 0
    eligible = np.flatnonzero(detector[:,0] == detector[:,0].max())
    didx = int(eligible[np.argmin(abs(detector[eligible,2]-h[3]))])
    target = detector[didx,:3].copy();target[1] += origin
    table_text = (payload/names[1]).read_text(encoding='utf-8')
    tables = []
    for label in ['kXcomMuPhotoelectric','kXcomMuCompton']:
        body = re.search(label+r'\[[^\]]+\]\s*=\s*\{([^}]+)\}',table_text).group(1)
        values = np.array([float(v[:-1]) for v in re.findall(r'[+\-]?[\d.]+e[+\-]\d+f',body)],dtype=np.float32).astype(float)
        assert len(values) == 4000
        tables.append(values.reshape(4,1000)[2])
    def mu(e):
        return float(sum(np.interp(e,np.arange(1,1001),v) for v in tables))
    assert abs(mu(440)-collimator[16]-collimator[17]) < 1e-7
    results = []
    for distance_from_rear in [.25,.5,1.,2.]:
        point = np.array([h[0]+h[4]+.01,upper[1]-distance_from_rear,h[3]])
        assert np.all(point > lower) and np.all(point < upper)
        assert np.all((point[0]-holes[:,0])**2+(point[2]-holes[:,3])**2 > holes[:,4]**2)
        incoming = chords(start,point,lower,upper,holes,origin)
        outgoing = chords(point,target,lower,upper,holes,origin)
        kine = kinematics(start,point,target,physics[7],detector[didx,9],physics[5:7])
        exact_tau = mu(440)*incoming['pb_length_mm']+mu(kine['scattered_energy_keV'])*outgoing['pb_length_mm']
        slab_tau = mu(440)*incoming['box_length_mm']+mu(kine['scattered_energy_keV'])*outgoing['box_length_mm']
        results.append(dict(distance_from_rear_mm=distance_from_rear,scatter_point_mm=point.tolist(),
                            incoming=incoming,outgoing=outgoing,kinematics=kine,
                            frozen_pb_mu_primary_per_mm=mu(440),frozen_pb_mu_scattered_per_mm=mu(kine['scattered_energy_keV']),
                            finite_holes_pb_survival_factor=math.exp(-exact_tau),
                            solid_box_pb_survival_factor_same_legs=math.exp(-slab_tau),
                            removed_void_optical_depth=slab_tau-exact_tau))
    center = np.array([results[0]['scatter_point_mm'][0],origin+collimator[14],h[3]])
    center_kine = kinematics(start,center,target,physics[7],detector[didx,9],physics[5:7])
    proof = dict(passed=True,scope='Conditional geometry/material attenuation witnesses only, not a generated response or observed event',
                 scientific_status='HOLD',physical_gate_passed=False,science_job=1677211,
                 producer_release_key=freeze['release_key'],frozen_input_sha256=bindings,
                 source_truth_sha256=digest(TRUTH),source_point_mm=start.tolist(),source_density_440_gamma_per_mm3=density,
                 hole_index=hidx,hole_record=h.tolist(),detector_bin_index=didx,detector_center_mm=target.tolist(),
                 witness_construction='Select central registered hole and a positive-density 3mm source voxel; conditional Pb points 0.01mm outside that hole and 0.25/0.5/1/2mm before the rear plane; choose outermost-x NaI bin nearest the hole z.',
                 witnesses=results,same_xy_collimator_center_kinematics=center_kine,
                 frozen_production_model='collimatorScatterSysMatCuda uses representative x/z and layer-center y for geometry/window; attenuatedSlabDepthIntegral applies total Pb attenuation through full thickness. Circular holes enter buildCollimatorScatterSamples through lead-area weights; this depth factor does not subtract finite-hole chord intervals.',
                 original_physical_gate_sha256=gate_sha,code_sha256=digest(__file__),elapsed_seconds=time.monotonic()-began,
                 no_response_or_simulation_executed=True,no_production_input_modified=True,no_observed_count_fit=True,
                 limitations='These conditional two-leg Pb survival factors are not probabilities of scattering or detection. They exclude geometric solid angle, scattering interaction density, NaI/intervening-crystal attenuation and absorption. No angular/volume/source integration or pathway frequency was calculated; points need not coincide with producer quadrature nodes. This is not a quantified production error, a measured missing fraction, a 24.02% root-cause attribution or a CUDA capture. No corrected kernel or response was implemented.')
    write(REPORT/'physical_collimator_paths_read_only.json',proof)
    assert digest(REPORT/'physical_gate.json') == gate_sha
    print('CONDITIONAL_PB_PATH_WITNESSES',[(v['incoming']['pb_length_mm'],v['outgoing']['pb_length_mm'],v['kinematics']['scattered_energy_keV'],v['removed_void_optical_depth']) for v in results])
    print('HOLD_UNCHANGED',True,'elapsed_seconds',proof['elapsed_seconds'])


if __name__ == '__main__': audit()
