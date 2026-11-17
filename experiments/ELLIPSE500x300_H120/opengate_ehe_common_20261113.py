"""Independent OpenGATE EHE/NEMA study; no import-time remote operations."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = "opengate_ehe_nema5e9_20261113"
DATA = HERE / "generated" / STUDY
REPORT = HERE / "reports/NEMA_Body_H60" / STUDY
CPU_ROOT = "/WORK/maty_work/lpz/" + STUDY
GPU_ROOT = "/data/home/scxi717/run/lpz/" + STUDY
OLD_GEOM = HERE / "reports/NEMA_Body_H60/ehe_spect_5e9_200/geometry_evidence/raw"
TRUTH = HERE / "generated/NEMA_Body_H60/truth_3mm.npz"
TRUTH_META = HERE / "reports/NEMA_Body_H60/manifest.json"
BASELINE_META = ROOT / "Analysis_Plots/research/ehe_dense_completed_1761244_20261102/data/metadata.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("w", encoding="utf8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False, ensure_ascii=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save_array(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def holes_mm():
    rows = []
    for row in range(1, 26):
        offset = 2.95 if row % 2 == 0 else 0.
        for column in range(1, 51):
            rows.append([(column - .5 - 25) * 5.9 + offset,
                         (row - .5 - 12.5) * math.sqrt(3) * 5.9 / 2])
    a = np.asarray(rows, dtype=np.float64)
    return a - a.mean(axis=0)


def crystals_mm():
    return np.asarray([[4 * (ix - 34 - .5), 354., 4 * (iz - 17 - .5)]
                       for ix in range(1, 69) for iz in range(1, 35)], dtype=np.float64)


def window_keV(energy):
    half = .13 * math.sqrt(511. * energy) / 2
    return [energy - half, energy + half]


def prepare():
    """Freeze existing truth/geometry identities and a new exact-count acquisition."""
    if (DATA / "inputs/config.json").exists():
        raise FileExistsError("Preserve the existing independent study; do not regenerate seeds")
    truth_meta = read(TRUTH_META)
    if sha(TRUTH) != truth_meta["truth_sha256"]:
        raise ValueError("Original NEMA truth SHA mismatch")
    holes = np.genfromtxt(OLD_GEOM / "EHE_CollimatorHoles.csv", delimiter=",", names=True)
    detectors = np.genfromtxt(OLD_GEOM / "EHE_DetectorGeometry.csv", delimiter=",", names=True)
    np.testing.assert_allclose(holes_mm(), np.c_[holes["x_mm"], holes["z_mm"]], atol=6e-10, rtol=0)
    np.testing.assert_allclose(crystals_mm(), np.c_[detectors["x_mm"], detectors["y_mm"]+345,
                                                  detectors["z_mm"]], atol=1e-12, rtol=0)
    np.testing.assert_allclose(holes["radius_mm"], 1.25, atol=0, rtol=0)
    np.testing.assert_allclose(holes["y1_mm"]+345, 298.5, atol=0, rtol=0)
    np.testing.assert_allclose(holes["y2_mm"]+345, 349., atol=0, rtol=0)
    for name, target in (("size_x_mm",4), ("size_y_mm",10), ("size_z_mm",4)):
        np.testing.assert_allclose(detectors[name], target, atol=0, rtol=0)
    with np.load(TRUTH) as t:
        integrals = {str(e): float(t[f"activity_{e}_zyx"].sum(dtype=np.float64)*27) for e in (218,440)}
        axes = {a: t[a+"_mm"].astype(float) for a in "xyz"}
    yields = {"218": .114, "440": .259}
    denom = sum(integrals[e]*yields[e] for e in integrals)
    p218 = integrals["218"]*yields["218"]/denom
    rng = np.random.default_rng(613030001)
    jobs = []
    emitted = np.zeros((20,2), dtype=np.int64)
    for v in range(20):
        n218 = int(rng.binomial(250_000_000, p218))
        emitted[v] = [n218, 250_000_000-n218]
        for ei, energy in enumerate((218,440)):
            for worker in range(5):
                n = int(emitted[v,ei]//5 + (worker < emitted[v,ei]%5))
                i = len(jobs)
                jobs.append(dict(index=i, kind="imaging", view=v, angle_deg=18*v,
                                 energy_keV=energy, primaries=n, seed=613130001+i,
                                 smear_seed=613430001+i, importance_weight=1.))
    if sum(j["primaries"] for j in jobs) != 5_000_000_000:
        raise AssertionError("Exact imaging budget does not close")
    responses = []
    # These are response-characterisation histories, separate from 5e9 imaging.
    # A constant spherical angular patch contains the complete camera/source cell.
    # cell. Outside it photons cannot reach any matter in this vacuum experiment.
    # Its solid-angle fraction is retained; no angular selection in imaging.
    for energy in (218,440):
        for gap in (45,95,145,195,245,295,345,395,445,495,555):
            for sx,sz in ((-2,-2),(-2,2),(2,-2),(2,2)):
                i = len(responses)
                half = math.atan2(math.hypot(169,86.5), gap-2)
                responses.append(dict(index=i, kind="response", energy_keV=energy,
                                      gap_mm=gap, source_center_mm=[sx,298.5-gap,sz],
                                      source_size_mm=[4,4,4], angular_half_width_deg=math.degrees(half),
                                      importance_weight=half*math.sin(half)/math.pi,
                                      primaries=50_000_000, seed=613230001+i,
                                      smear_seed=613530001+i))
    cfg = dict(study=STUDY, created_utc=now(), material_condition="original_vacuum_baseline",
               imaging_primary_gamma_total=5_000_000_000, views=20,
               activity_integrals_mm3=integrals, gamma_yields=yields,
               imaging_primary_energy_fraction_expected={"218":p218,"440":1-p218},
               emitted_primaries_by_view_energy=emitted.tolist(),
               primary_count_definition="total 218+440 gamma, not Ac225 decays",
               imaging_direction="full 4pi isotropic; unit weight; no forcing",
               source_model="original 3mm fractional activity image; uniform within each voxel",
               source_axes_mm={a: axes[a].tolist() for a in axes},
               source_origin_sha256=sha(TRUTH), source_manifest_sha256=sha(TRUTH_META),
               old_geometry_sha256={n:sha(OLD_GEOM/n) for n in
                                    ("EHE_CollimatorHoles.csv","EHE_DetectorGeometry.csv","EHE_GeometrySummary.txt")},
               baseline_metadata_sha256=sha(BASELINE_META),
               collimator=dict(material="G4_Pb",size_mm=[330,50.5,165],center_mm=[0,323.75,0],
                               hole_count=1250,hole_diameter_mm=2.5,hole_pitch_mm=5.9,
                               hole_material="G4_Galactic",lattice="original triangular circular holes"),
               detector=dict(material="G4_SODIUM_IODIDE", shape_xz=[68,34],
                             pixel_size_mm=[4,10,4],center_y_mm=354,side_shield=False,
                             readout="per event per crystal summed energy, no intercrystal sum"),
               energy_resolution=dict(reference_FWHM_fraction=.13,reference_keV=511.,
                                      sigma_divisor=2.35482),
               windows_keV={str(e):window_keV(e) for e in (218,440)},
               physics_list="G4EmStandardPhysics_option4", production_cut_mm=.7,
               opengate_version="10.1.2",pytomography_version="3.4.0",
               reconstruction=dict(iterations=200,save_step=1,subsets=1,initial_density=1.,
                                   image_regularization=None,image_smoothing=None,
                                   grid_spacing_mm=4.,grid_shape_xyz=[126,126,34],
                                   physical_support="ellipse semi-axes 250/150mm, z +/-60mm",
                                   response="full-transport absolute 2D CDR; depth interpolation and 4 phase average",
                                   approximation="translation-invariant CDR; finite-collimator edge and phase dependence are not certified",
                                   crosstalk="fixed MC-derived 440->218 projection from this branch final440"),
               resource_fraction_limit=.8, max_response_or_imaging_concurrency=200,
               old_outputs_modified=False, independent_validation_jobs=False)
    dest=DATA/"inputs"
    dest.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copy2(TRUTH,dest/"truth_3mm.npz")
    save(dest/"config.json",cfg)
    save(dest/"imaging_jobs.json",jobs)
    save(dest/"response_jobs.json",responses)
    save(REPORT/"geometry_source_checks.json",dict(passed=True,created_utc=now(),
         geometry_tolerance_mm=6e-10,holes_checked=1250,crystals_checked=2312,
         source_sha256=sha(TRUTH),imaging_primaries=sum(j["primaries"] for j in jobs),
         response_primaries=sum(j["primaries"] for j in responses),
         condition="vacuum confirmed by user; new independent seeds; existing geometry/source reused"))
    return cfg
