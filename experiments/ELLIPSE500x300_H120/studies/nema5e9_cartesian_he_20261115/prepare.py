"""Build an isolated freeze from delivered HE and actual production v5 evidence."""
import ast
import hashlib
import io
import json
import shutil
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
import numpy as np
from common import TAG, sha, read, write
from cartesian import axes, area_fraction, field_coordinates, SHAPE

CODE = Path(__file__).resolve().parent
EXP = CODE.parents[1]
ROOT = EXP.parents[1]
DATA = EXP/'generated'/TAG
REPORT = EXP/'reports/NEMA_Body_H60'/TAG
SERVER = '65114_lipeize'
SERVER_ROOT = '/home/lipeize/JSCC_FOV120_20260924'
ENGINE = 'Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main'
CHANNELS = {
    'A440': ('JSCC_440keV_pe_v4_ELLIPSE500x300_H120', '440keV_RotateNum20'),
    'A218': ('JSCC_218keV_pe_v4_ELLIPSE500x300_H120', '218keV_RotateNum20'),
    'C440to218': ('JSCC_440keV_to_218keVwin_pe_v4_ELLIPSE500x300_H120', '440keV_to218win_RotateNum20')}
BINARIES = {
    'pe': ('PEGen_RayTracing_CircularHole/PEGen_V4_Production', '58f1fbc62b7c24f572a0a57818cded6d2278a7c988097ed9832d0bbe57af9c94'),
    'scatter': ('ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local', '717ce146cd56a00717cf0a8d440e464578fa80490a075ab867b3cce832596c2c')}


def fetch_dependencies(contract, target):
    """Only ~6 MB of original code/parameters, never repeat scientific outputs."""
    if (DATA/'dependencies.json').exists():
        old = read(DATA/'dependencies.json')
        for n, h in old['files'].items():
            assert sha(target/n) == h, n
        return old
    mapping = {}
    for channel, (run, factor) in CHANNELS.items():
        for name in ['Params_Image.dat', 'Params_Physics.dat', 'Params_Detector.dat', 'Params_Collimator.dat', 'ELLIPSE_inputs.json']:
            mapping[f'parameters/{channel}/{name}'] = f'{SERVER_ROOT}/{ENGINE}/runs/{run}/{name}'
        mapping[f'parameters/{channel}/factor_manifest.json'] = f'{SERVER_ROOT}/experiments/ELLIPSE500x300_H120/generated/FactorsCalibrated/{factor}/factor_manifest.json'
    for key, (path, _) in BINARIES.items():
        mapping['bin/'+key] = f'{SERVER_ROOT}/{ENGINE}/{path}'
    program = "import sys,tarfile,io,json,pathlib\nmapping="+repr(mapping)+"\nwith tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as t:\n for name,path in mapping.items():\n  p=pathlib.Path(path);raw=p.read_bytes();h=tarfile.TarInfo(name);h.size=len(raw);t.addfile(h,io.BytesIO(raw))\n"
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', SERVER,
        SERVER_ROOT+'/.venv/bin/python -c '+shlex.quote(program)], capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace')[:2000])
    records = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        for member in archive:
            if member.name not in mapping or not member.isfile():
                raise ValueError('Unexpected dependency archive member')
            data = archive.extractfile(member).read()
            path = target/member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert path.read_bytes() == data
            else:
                path.write_bytes(data)
            records[member.name] = sha(path)
    assert set(records) == set(mapping)
    for key, (_, expected) in BINARIES.items():
        assert records['bin/'+key] == expected
    for channel, (_, factor) in CHANNELS.items():
        d = target/'parameters'/channel
        assert sha(d/'factor_manifest.json') == contract['factor_payload_sha256'][factor+'/factor_manifest.json']
        provenance = read(d/'ELLIPSE_inputs.json')
        for name, expected in provenance['parameters'].items():
            assert sha(d/name) == expected, (channel, name)
        m = read(d/'factor_manifest.json')
        assert sha(d/'Params_Image.dat') == m['input_sha256']['params_image']
        assert sha(d/'Params_Detector.dat') == m['input_sha256']['params_detector']
    result = dict(source_host=SERVER, source_paths=mapping, files=records,
                  new_transport=0, repeated_matrix_transfer=False)
    write(DATA/'dependencies.json', result)
    return result


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    target = DATA/'payload'
    target.mkdir(exist_ok=True)
    if (REPORT/'freeze.json').exists():
        manifest = read(REPORT/'freeze.json')
        for n, h in manifest['files'].items():
            assert sha(target/n) == h
        print('EXISTING_CARTESIAN_FREEZE', manifest['key'])
        return
    parent = EXP/'generated/compton_energy_probability_v5_5e9_full10000/formal_payload'
    contract = read(parent/'contract.json')
    summary = read(EXP/'reports/NEMA_Body_H60/compton_energy_probability_v5_5e9_full10000/formal_summary.json')
    assert sha(parent/'contract.json') == summary['contract_sha256']
    hf = EXP/'generated/opengate_he_nema_20261114/5e9/delivery'
    hr = EXP/'reports/NEMA_Body_H60/opengate_he_nema_20261114/5e9/strict_delivery.json'
    receipt = read(hr)
    assert receipt['passed'] and receipt['all_files_SHA_verified']
    for name in ['reconstruction/440/final.npz', 'reconstruction/440/sensitivity.npz',
                 'local_execution/opengate_ehe_projector_20261113.py', 'local_execution/release.json']:
        assert sha(hf/name) == receipt['files'][name]['sha256'], name
    final = np.load(hf/'reconstruction/440/final.npz')
    sensitivity = np.load(hf/'reconstruction/440/sensitivity.npz')
    x, y, z = axes()
    for k, a in zip(['x_mm', 'y_mm', 'z_mm'], [x, y, z]):
        assert np.array_equal(a, final[k]), k
    fraction = sensitivity['fraction_xyz'].copy()
    assert fraction.shape == SHAPE and final['density_xyz'].shape == SHAPE
    assert np.isfinite(fraction).all() and fraction.min() >= 0 and fraction.max() <= 1
    independent = area_fraction(x, y, 250, 150)[:, :, None]*np.maximum(0.,
        np.minimum(z+2, 60)-np.maximum(z-2, -60))[None, None, :]/4
    assert np.allclose(independent, fraction, rtol=1e-12, atol=1e-12)
    active = np.flatnonzero(fraction.ravel() > 0)
    assert len(active) == 227040
    xyz = np.column_stack([a.ravel()[active] for a in np.meshgrid(x, y, z, indexing='ij')])
    grid = np.arange(130)*4.-258
    disk = np.broadcast_to(area_fraction(grid, grid, 255, 255).T, (30, 130, 130)).copy()
    volume = np.pi*255**2*120
    assert np.isclose(disk.sum()*64, volume, rtol=1e-12)
    np.savez_compressed(target/'cartesian_geometry.npz', x_mm=x, y_mm=y, z_mm=z,
        fraction_xyz=fraction, active_indices=active, active_coordinates_mm=xyz,
        field_coordinates_mm=field_coordinates(), reference_circle_fraction_zyx=disk)
    copies = ['compton_energy_probability_v5.py', 'compton_event_response.py',
        'compton_sparse_ops.py', 'detector_csv.py', 'torch_active_operator.py',
        'single_checkpoint_mlem.py', 'transfer_training_summary.json', 'verify_first_scatter.py']
    for name in copies:
        assert sha(parent/name) == contract['files'][name]
        shutil.copyfile(parent/name, target/name)
    wdir = EXP/'generated/nema5e9_width_sweep_20261115/payload_r3'
    wf = read(EXP/'reports/NEMA_Body_H60/nema5e9_width_sweep_20261115/freeze_r3.json')
    for name in ['calibration_rows.npy', 'calibration_extract.json', 'adapters.py', 'Detector.csv', 'jscc_5e10_joint_mlem.py']:
        assert sha(wdir/name) == wf['files'][name]
        shutil.copyfile(wdir/name, target/name)
    assert np.load(target/'calibration_rows.npy').shape == (170247, 4)
    for p in (parent/'selections').glob('*.npy'):
        assert sha(p) == contract['files']['selections/'+p.name]
        (target/'selections').mkdir(exist_ok=True)
        shutil.copyfile(p, target/'selections'/p.name)
    dependencies = fetch_dependencies(contract, target)
    plan = dict(study=TAG, channels=['440_SC', '440_JSCC', '218_SC'],
        total_dual_energy_gamma=5000000000, material='original vacuum; no new transport',
        iterations=10000, save_step=50, views=20, spacing_mm=[4, 4, 4], shape_xyz=list(SHAPE),
        active_cells=len(active), physical_FOV_mm=[500, 300, 120], native_axis_order='xyz',
        HE_delivery_sha256=sha(hr), HE_release=read(hf/'local_execution/release.json')['release'],
        HE_native_sensitivity_file_sha256=sha(hf/'reconstruction/440/sensitivity.npz'),
        HE_fraction_exact_copy=True, HE_fraction_independently_verified=True,
        response_field_shape_zyx=[30, 130, 130], response_grid_mm=4,
        response='New PE v4 + detector-local scatter, original frozen four-layer calibration scales, no refit. Cartesian bilinear rotation; no polar field/image interpolation.',
        weight='Full cell volume 64 mm3 in B; native HE ellipse overlap exactly once in every view.',
        compton='Original 100% continuous-energy v5 tail16_mid2, Gaussian FP32 evaluation; unchanged legacy selections, data, positions, widths and KN. New Cartesian reference normalizer.',
        sensitivity='Independent 170247 circle_train events, 1e9 primary gamma, same new response and reference-disk measure, view mean and native overlap; never observed NEMA C.T@1.',
        calibration_volume_mm3=volume, reference_circle_radius_mm=255,
        crosstalk='Same branch final 440 SC through fresh C440to218; not JSCC and not previous polar image.',
        input_sha256=contract['input_sha256'], events_per_view=contract['events_per_view'], events=483743,
        parent_contract_sha256=sha(parent/'contract.json'), dependencies_sha256=sha(DATA/'dependencies.json'),
        origin_width_freeze=wf['key'], numerical_tolerance=1e-5, event_block=32,
        source_tile=32768, no_random_resmearing=True, initial_density=1., regularization=None,
        reconstruction_nodes=8, gpus_per_node=1, cpus_per_task=8,
        response_cache_bytes=483743*len(active)*4,
        response_field_bytes_per_channel=10496*30*130*130*4,
        matrix_disk_cap_bytes=90<<30, result_disk_cap_bytes=3<<30,
        max_gpu_and_rss_fraction=.8, phase_limit_hours=22, allocation_hours=72,
        status='LOCAL_PREPARATION_NOT_SUBMITTED', scientific_model_change='Grid/basis and Cartesian rotation; not an equivalent engineering-only change',
        separate_validation_jobs=False, restart_timers=False)
    write(target/'study.json', plan)
    for name in ['common.py', 'cartesian.py', 'build_matrix.py', 'run_reconstruction.py', 'test_cartesian.py']:
        ast.parse((CODE/name).read_text(encoding='utf8'))
        shutil.copyfile(CODE/name, target/name)
    test = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(target/'test_cartesian.py')],
                          capture_output=True, text=True, encoding='utf8', timeout=180)
    (REPORT/'local_tests.txt').write_text(test.stdout+test.stderr, encoding='utf8')
    if test.returncode:
        raise RuntimeError('Local regression failed: '+test.stdout+test.stderr)
    files = {str(p.relative_to(target)).replace('\\', '/'): sha(p) for p in sorted(target.rglob('*')) if p.is_file()}
    key = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()[:16]
    manifest = dict(key=key, files=files, study=plan, local_tests_passed=True,
                    local_tests_sha256=sha(REPORT/'local_tests.txt'))
    write(target/'release_manifest.json', manifest)
    write(REPORT/'freeze.json', manifest)
    print('CARTESIAN_FROZEN', key, len(files), 'files', 'active', len(active))


if __name__ == '__main__':
    main()
