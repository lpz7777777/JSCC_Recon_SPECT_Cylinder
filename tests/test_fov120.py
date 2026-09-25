"""Cross-stage invariants for the axial FOV extension, using small synthetic data."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fov_config import load_config, factor_geometry, validate_factor_geometry


def test_sensitivity_provenance_python39_and_tamper(tmp_path, monkeypatch):
    import hashlib
    from fov_config import validate_sensitivity_provenance
    monkeypatch.delattr(hashlib, 'file_digest', raising=False)
    names = ('Sensi_d', 'factor_manifest.json', 'coor_polar_full.csv',
             'Detector.csv', 'polar_cell_volume_mm3.float64', 'SysMat_polar')
    hashes = {}
    for name in names:
        data = b'x' * (8 * 1024 * 1024 + 3) if name == 'SysMat_polar' else name.encode()
        (tmp_path / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    record = dict(operator='K*B', resolution_fwhm=.13, reference_keV=511,
                  sum_threshold_MeV=.350, input_already_smeared=True, hashes=hashes)
    (tmp_path / 'Sensi_d_provenance.json').write_text(json.dumps(record))
    validate_sensitivity_provenance(tmp_path, verify_matrix=True)
    with (tmp_path / 'SysMat_polar').open('r+b') as stream:
        stream.seek(-1, 2)
        stream.write(b'y')
    with pytest.raises(ValueError, match='Stale Sensi_d provenance: SysMat_polar'):
        validate_sensitivity_provenance(tmp_path, verify_matrix=True)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


xcat=module("xcat_fov_test","Geant4Sim/generate_xcat_ac225_psma_abdomen.py")
validator=module("xcat_validation_fov_test","Geant4Sim/validate_xcat_ac225_psma_abdomen.py")
workflow=module("fov_workflow_test","experiments/FOV120/workflow.py")
calibration=module("fov_calibration_test","experiments/FOV120/calibrate.py")
imaging=module("fov_imaging_test","experiments/FOV120/imaging.py")


def test_config_physical_and_computational_support_are_distinct():
    cfg=load_config()
    assert cfg["z_layers"]==40 and cfg["pixel_count"]==51240
    assert cfg["z_centers_mm"][[0,-1]].tolist()==[-58.5,58.5]
    assert cfg["physical_radius_mm"]==150 and cfg["support_radius_mm"]==153
    assert cfg["pixel_count"]*cfg["detector_count"]*4==2151260160


@pytest.mark.parametrize("nz",[20,40])
def test_xcat_box_roundtrip_and_boundary(nz):
    # Distinct first/last axial layers reveal hidden 20-layer or +/-30 offsets.
    native=np.zeros((nz*2,200,200))
    native[:2,98:102,98:102]=2
    native[-2:,98:102,98:102]=5
    native[4:6,98:100,0:2]=3  # native cells along the cylindrical boundary
    target=xcat.block_mean(native)
    boxes=xcat.boxify(218,target,native,np.ones_like(native,dtype=bool))
    sources=[{"energy":b.energy,"center":(b.x,b.y,b.z),"halfx":b.hx,"halfy":b.hy,
              "halfz":b.hz,"intensity":b.intensity} for b in boxes]
    restored=validator.backproject(sources,{218:.114},target.shape)[218]
    np.testing.assert_allclose(restored,target,atol=1e-12)
    assert max(abs(b.z)+b.hz for b in boxes)==nz*1.5


def test_geometry_discovers_layers(tmp_path):
    rows=[[x,0,z] for z in (-4.5,-1.5,1.5,4.5) for x in (0,6)]
    np.savetxt(tmp_path/"coor_polar_full.csv",rows,delimiter=",")
    _,per_layer,nz=factor_geometry(tmp_path)
    assert (per_layer,nz)==(2,4)


def test_layer_calibration_absolute_not_total_preserving():
    volumes=np.array([2.,3.,7.])
    matrix=np.array([[1,2,3,4],[2,3,4,5],[3,4,5,6]],dtype=float)
    det=np.array([[0,y,0] for y in (200,230,260,290)])
    scale,metrics=calibration.layer_scales(matrix,volumes,det,np.array([100,200,400,800]),1000)
    expected=matrix.sum(axis=0)/volumes.sum()
    np.testing.assert_allclose(expected*scale,[.1,.2,.4,.8])
    assert len(metrics)==4


def test_duplicate_seed_rejected(tmp_path):
    p=tmp_path/"jobs.json"
    p.write_text(json.dumps({"jobs":[{"seed":1,"index":0},{"seed":1,"index":1}]}))
    with pytest.raises(ValueError,match="Duplicate"):
        workflow.load_jobs(p)


def test_failed_worker_cannot_be_collected(tmp_path):
    job={"index":0,"seed":1,"dataset":"calibration_218","level":"pilot","view":1,"role":"calibration"}
    p=tmp_path/"jobs.json";p.write_text(json.dumps({"jobs":[job]}))
    worker=tmp_path/"workers/00000";worker.mkdir(parents=True)
    (worker/"worker.json").write_text(json.dumps(dict(job,status="failed")))
    with pytest.raises(ValueError,match="Failed"):
        workflow.collect(p,"calibration_218","pilot",tmp_path/"output")


def test_existing_xcat_cannot_be_overwritten(tmp_path):
    (tmp_path/"manifest.json").write_text("{}")
    with pytest.raises(FileExistsError):
        xcat.build(Path("missing.xif"),tmp_path)


def test_native_source_outside_grid_rejected():
    source={"energy":218,"center":(0,0,-61),"halfx":.75,"halfy":.75,"halfz":.75,"intensity":1}
    with pytest.raises(AssertionError,match="outside"):
        validator.backproject([source],{218:.114},(40,100,100))


def test_closed_loop_projection_view_order(tmp_path):
    matrix=np.array([[1,2],[3,5],[7,11]],dtype='<f4') # pixels x detectors
    matrix.tofile(tmp_path/'SysMat_polar')
    # project only uses coordinates to verify the number of image cells.
    np.savetxt(tmp_path/'coor_polar_full.csv',[[0,0,-3],[0,0,0],[0,0,3]],delimiter=',')
    rot=np.array([[1,3],[2,1],[3,2]])
    np.savetxt(tmp_path/'RotMat_full.csv',rot,delimiter=',',fmt='%d')
    rho=np.array([2.,3.,5.])
    projected=imaging.project(tmp_path,rho)
    expected=np.stack([matrix.T@rho,matrix.T@rho[[2,0,1]]])/2
    np.testing.assert_allclose(projected,expected)


def test_volume_weighted_metrics_not_equal_polar_samples():
    image=np.array([2.,4.]);truth=np.array([1.,2.]);volume=np.array([1.,3.])
    result=imaging.weighted_metrics(image,truth,volume,np.ones(2))
    assert result['mean']==3.5 and result['recovery']==2
    assert result['integral']==14


def test_center_matrix_comparison_detects_wrong_axial_offset(tmp_path):
    comparison=module('fov_center_matrix_test','experiments/FOV120/compare_center_matrices.py')
    new=np.broadcast_to(np.arange(40,dtype='<f4')[None,:,None,None],(1,40,51,51)).copy()
    old=new[:,10:30].copy()
    old.tofile(tmp_path/'old');new.tofile(tmp_path/'new')
    assert comparison.compare(tmp_path/'old',tmp_path/'new',1)['relative_l2']==0
    new[:,10]+=1;new.tofile(tmp_path/'new')
    assert comparison.compare(tmp_path/'old',tmp_path/'new',1)['relative_l2']>0


def test_axial_psf_reports_censored_edge():
    point=module('fov_point_test','experiments/FOV120/reconstruct_point.py')
    z=np.arange(-6,7,3)
    assert point.axial_fwhm(z,np.array([0,1,2,1,0]))['fwhm_mm']==6
    assert point.axial_fwhm(z,np.array([0,0,0,1,2]))['censored_at_boundary']


def test_response_geometry_rejects_same_size_different_coordinates(tmp_path):
    import shutil
    first=tmp_path/'first';first.mkdir()
    np.savetxt(first/'coor_polar_full.csv',[[0,0,-1.5],[6,0,-1.5],[0,0,1.5],[6,0,1.5]],delimiter=',')
    np.savetxt(first/'Detector.csv',[[i+1,0,200+30*i,0] for i in range(4)],delimiter=',')
    np.ones(4,dtype='<f8').tofile(first/'polar_cell_volume_mm3.float64')
    rot=np.array([[1,2],[2,1],[3,4],[4,3]])
    for name in ('RotMat_full.csv','RotMatInv_full.csv'):np.savetxt(first/name,rot,delimiter=',',fmt='%d')
    np.ones(16,dtype='<f4').tofile(first/'SysMat_polar')
    second=tmp_path/'second';shutil.copytree(first,second)
    validate_factor_geometry({'a':first,'b':second},scan_matrix=True)
    coords=np.loadtxt(second/'coor_polar_full.csv',delimiter=',');coords[:,2]+=3
    np.savetxt(second/'coor_polar_full.csv',coords,delimiter=',')
    with pytest.raises(ValueError,match='geometry mismatch'):
        validate_factor_geometry({'a':first,'b':second})


def test_full_height_evaluation_reads_six_images_and_histories(tmp_path):
    factor=tmp_path/'factor';factor.mkdir()
    xy=np.array([[0,0],[150,0],[0,150],[-150,0],[0,-150]])
    coords=np.array([[x,y,z] for z in np.arange(-58.5,60,3) for x,y in xy])
    np.savetxt(factor/'coor_polar_full.csv',coords,delimiter=',')
    np.ones(len(coords),dtype='<f8').tofile(factor/'polar_cell_volume_mm3.float64')
    truth=tmp_path/'truth.npz'
    low=np.ones(len(coords));high=2*low
    np.savez(truth,rho218=low,rho440=high)
    result=tmp_path/'synthetic_test_only';result.mkdir()
    (result/'run_manifest.json').write_text(json.dumps({'iterations':100,'save_step':50}))
    for index,key in enumerate(imaging.OUTPUTS):
        values=high if index<3 else low if index==3 else low+high
        values.astype('<f4').tofile(result/f'Image_{key}')
        if index<4:np.stack([values,values]).astype('<f4').tofile(result/f'Image_{key}_Iter_100_2')
    imaging.evaluate(result,factor,truth)
    metrics=json.loads((result/'FullFOV/metrics.json').read_text())
    assert metrics['crop_mm']==0 and len(metrics['results'])==6
    assert metrics['results']['440_SinglePhoton']['edge']['nrmse']==0
    assert len(list((result/'FullFOV').glob('*.png')))==12
