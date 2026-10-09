"""Read-only Pb area quadrature audit; no response integrand or matrix generation."""
import math
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from ehe_common import DATA, REPORT, RESPONSES, digest, read, write


def circle_rectangle(radius, xlo, xhi, zlo, zhi):
    """Analytic integral of clipped vertical disk chords, split at every branch."""
    a, b = max(xlo, -radius), min(xhi, radius)
    if b <= a or zhi <= -radius or zlo >= radius:
        return 0.0
    cuts = [a, b]
    for z in (zlo, zhi):
        if abs(z) < radius:
            root = math.sqrt(radius*radius-z*z)
            cuts.extend(x for x in (-root, root) if a < x < b)
    def primitive(x):
        return .5*(x*math.sqrt(max(0.0,radius*radius-x*x))
                   +radius*radius*math.asin(max(-1.0,min(1.0,x/radius))))
    total = 0.0
    cuts = sorted(set(cuts))
    for left, right in zip(cuts, cuts[1:]):
        s = math.sqrt(max(0.0,radius*radius-((left+right)/2)**2))
        if min(zhi,s) <= max(zlo,-s):
            continue
        upper_const, upper_s = (zhi,0) if zhi < s else (0,1)
        lower_const, lower_s = (zlo,0) if zlo > -s else (0,-1)
        total += ((upper_const-lower_const)*(right-left)
                  +(upper_s-lower_s)*(primitive(right)-primitive(left)))
    return max(0.0,total)


def sample_weights(holes, width, height, nx, nz, subdivisions, dtype):
    """CPU reference to frozen host builder, exhaustive conservative cell candidates."""
    f = dtype
    dx, dz = f(f(width)/nx), f(f(height)/nz)
    weights = []
    leads = []
    for iz in range(nz):
        for ix in range(nx):
            cx = f(f(-f(width)/f(2))+f(f(ix+.5)*dx))
            cz = f(f(-f(height)/f(2))+f(f(iz+.5)*dz))
            u = (np.arange(subdivisions,dtype=dtype)+f(.5))/f(subdivisions)-f(.5)
            x,z = np.meshgrid(cx+u*dx,cz+u*dz,indexing='xy')
            x,z = x.ravel(),z.ravel()
            # All possible holes; no fixed number of nearest neighbours.
            selected = ((abs(holes[:,0]-float(cx)) <= float(dx)/2+holes[:,4]+1e-4)
                        &(abs(holes[:,3]-float(cz)) <= float(dz)/2+holes[:,4]+1e-4))
            h = holes[selected].astype(dtype)
            inside = np.zeros(len(x),dtype=bool)
            for hole in h:
                ddx,ddz = x-hole[0],z-hole[3]
                inside |= ddx*ddx+ddz*ddz <= hole[4]*hole[4]
            count = int((~inside).sum())
            leads.append(count)
            weights.append(float(f(f(dx*dz)*f(f(count)/f(subdivisions*subdivisions)))))
    return np.array(weights), np.array(leads)


def audit():
    began = time.monotonic()
    freeze = read(REPORT/'response_repair_freeze.json')
    payload = DATA/freeze['payload_dir']
    source_name = 'engine/ScatterGen_RayTracing_CircularHole/scatter.cu'
    assert digest(payload/source_name) == freeze['sha256'][source_name]
    gate_sha = digest(REPORT/'physical_gate.json')
    assert not read(REPORT/'physical_gate.json')['passed']
    log_binding = read(DATA/'collimator_area_read_only/original_log_binding.json')
    bindings = {}
    geometry = None
    for response in RESPONSES:
        name = f'params/{response}/Params_Collimator.dat'
        sha = digest(payload/name)
        assert sha == freeze['sha256'][name]
        col = np.fromfile(payload/name,'<f4').astype(float)
        assert col[0] == 1 and col[10] == 1250
        current = np.concatenate((col[10:15],col[100:]))
        if geometry is None:
            geometry = current
        else:
            assert np.array_equal(geometry,current)
        for slab in range(4):
            receipt = REPORT/'response_preservation_1672966'/response/f'slab_{slab}'/'receipt.json'
            assert read(receipt)['files']['Params_Collimator.dat'] == sha
            bindings[f'{response}/slab_{slab}'] = dict(params_sha256=sha,receipt_sha256=digest(receipt))
    for record in log_binding['records']:
        path = DATA/'collimator_area_read_only'/f"{record['response']}_slab_{record['slab']}_scatter.log"
        assert digest(path) == record['original_scatter_log_sha256']
        assert record['pre_stop_receipt_sha256'] == bindings[f"{record['response']}/slab_{record['slab']}"]['receipt_sha256']
        assert record['collimator_area_lines'] == ['Collimator layer 0 XCOM material=Pb volume samples=1250 represented Pb/high-Z area=48324.4 mm^2']
    width,height = col[11],col[13]
    holes = col[100:].reshape(1250,9)
    assert np.all(abs(holes[:,0])+holes[:,4] < width/2)
    assert np.all(abs(holes[:,3])+holes[:,4] < height/2)
    d2 = ((holes[:,None,0]-holes[None,:,0])**2+(holes[:,None,3]-holes[None,:,3])**2)
    touching = (holes[:,None,4]+holes[None,:,4])**2
    np.fill_diagonal(d2,np.inf)
    assert np.all(d2 > touching)
    minimum_hole_gap = float(np.sqrt(d2).min()-2*holes[:,4].max())
    nx = int(np.floor(np.float32(np.sqrt(np.float32(1250*width/height)))+np.float32(.5)))
    nz = (1250+nx-1)//nx
    assert nx*nz == 1250
    exact = np.full((nz,nx),width/nx*height/nz)
    errors = []
    for hx,_,_,hz,r,*_ in holes:
        hole_area = 0.0
        for iz in range(max(0,int((hz-r+height/2)/(height/nz))),min(nz,int((hz+r+height/2)/(height/nz))+1)):
            for ix in range(max(0,int((hx-r+width/2)/(width/nx))),min(nx,int((hx+r+width/2)/(width/nx))+1)):
                value = circle_rectangle(r,-width/2+ix*width/nx-hx,-width/2+(ix+1)*width/nx-hx,
                                         -height/2+iz*height/nz-hz,-height/2+(iz+1)*height/nz-hz)
                exact[iz,ix] -= value
                hole_area += value
        errors.append(abs(hole_area-math.pi*r*r))
    assert max(errors) < 1e-8 and exact.min() > 0
    exact_total = float(width*height-np.sum(math.pi*holes[:,4]**2))
    assert abs(exact.sum()-exact_total) < 1e-7
    exact = exact.ravel()
    comparisons = []
    default_weights = None
    for n,dtype in [(8,np.float32),(8,np.float64),(16,np.float64),(32,np.float64)]:
        weights,counts = sample_weights(holes,width,height,nx,nz,n,dtype)
        difference = weights-exact
        relative = difference/exact
        comparisons.append(dict(subdivisions=n,cpu_dtype=np.dtype(dtype).name,
            sample_count=nx*nz*n*n,represented_area_mm2=float(weights.sum()),
            signed_total_area_error_fraction=float(difference.sum()/exact_total),
            absolute_cell_error_sum_mm2=float(abs(difference).sum()),
            absolute_cell_error_fraction=float(abs(difference).sum()/exact_total),
            minimum_cell_error_fraction=float(relative.min()),maximum_cell_error_fraction=float(relative.max()),
            maximum_absolute_cell_error_fraction=float(abs(relative).max()),zero_lead_sample_cells=int((counts==0).sum()),
            maximum_absolute_reweighting_fraction_relative_to_sampled=float(abs(exact/weights-1).max()),
            maximum_error_cell_xz_indices=[int(np.argmax(abs(relative))%nx),int(np.argmax(abs(relative))//nx)]))
        if n == 8 and dtype == np.float32:
            default_weights = weights
            assert format(float(weights.sum()),'.6g') == '48324.4'
    # Independent quadrature checks of the analytic clipped chord calculation.
    # Split at the same geometric breakpoints; Gauss integration is numerical,
    # while all 1250 disk partition sums above check conservation separately.
    gx,gw = np.polynomial.legendre.leggauss(256)
    gauss_differences = []
    for rect in [(-1.25,1.25,-1.25,1.25),(-.83,.71,-.4,.95),(0,1.25,0,1.25),(-.1,.2,1.15,1.4)]:
        a,b,c,d = rect
        a,b = max(a,-1.25),min(b,1.25)
        cuts = [a,b]
        for z in (c,d):
            if abs(z)<1.25:
                root = math.sqrt(1.25**2-z*z)
                cuts += [v for v in (-root,root) if a<v<b]
        numeric = 0.0
        cuts = sorted(set(cuts))
        for left,right in zip(cuts,cuts[1:]):
            x = (right-left)/2*gx+(right+left)/2
            s = np.sqrt(np.maximum(0,1.25**2-x*x))
            numeric += float((right-left)/2*np.dot(gw,np.maximum(0,np.minimum(d,s)-np.maximum(c,-s))))
        gauss_differences.append(abs(numeric-circle_rectangle(1.25,*rect)))
    assert max(gauss_differences) < 2e-7
    proof = dict(passed=True,scope='Geometric Pb cell-area weights only, no response integrand or event generation',
        scientific_status='HOLD',physical_gate_passed=False,original_job=1672966,science_job=1677211,
        producer_release_key=freeze['release_key'],scientific_source_sha256=digest(payload/source_name),
        original_physical_gate_sha256=gate_sha,code_sha256=digest(__file__),
        frozen_params_and_pre_stop_receipt_bindings=bindings,original_runtime_logs=log_binding,
        geometry=dict(width_mm=width,height_mm=height,holes=1250,nx=nx,nz=nz,
            holes_nonoverlapping_and_inside=True,minimum_hole_gap_mm=minimum_hole_gap,
            exact_pb_area_mm2=exact_total,maximum_disk_partition_conservation_error_mm2=max(errors),
            analytic_area_gauss_check_maximum_difference_mm2=max(gauss_differences)),
        comparisons=comparisons,
        default_reference_weights_sha256=__import__('hashlib').sha256(default_weights.astype('<f8').tobytes()).hexdigest(),
        original_log_total_area_error_fraction_interval=[(48324.35-exact_total)/exact_total,(48324.45-exact_total)/exact_total],
        configuration_identity='Frozen defaults: hole-count cells and 8x8 subgrid. CPU sequential float32 reference reproduces all12 logged sample counts and area at printed precision. Inherited environment overrides and per-cell runtime weights were not captured; this is not bitwise producer-state identification.',
        area_only_bound='For fixed nonnegative per-cell factors k, abs(sum(sampled*k)-sum(exact*k))/sum(exact*k) is at most max(abs(sampled/exact-1)); the corresponding bound with sampled sum as denominator is max(abs(exact/sampled-1)). This applies to the reported CPU reference weights only, and excludes representative-point, depth/path, angular, source and other response errors. It is not a bound on the actual complete response.',
        no_response_or_simulation_executed=True,no_production_input_modified=True,no_observed_count_fit=True,
        limitations='This geometric reference does not integrate the frozen scattering kernel, modify its sampling, or quantify the continuum finite-position/depth/angle error. Total area cancellation alone does not bound detection-weighted error. Higher-subdivision area comparisons are diagnostics, not new production inputs. Root cause remains UNDETERMINED; original22 HOLD and138720 inadequate bin diagnostics remain unchanged.',
        elapsed_seconds=time.monotonic()-began)
    write(REPORT/'physical_collimator_area_read_only.json',proof)
    assert digest(REPORT/'physical_gate.json') == gate_sha
    print('PB_AREA_READ_ONLY',proof['geometry'])
    print('AREA_COMPARISONS',comparisons)
    print('HOLD_UNCHANGED',True,'elapsed_seconds',proof['elapsed_seconds'])


if __name__ == '__main__': audit()
