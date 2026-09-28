"""Prepare the 500 x 300 x 120 mm XCAT source without scaling anatomy."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Geant4Sim import generate_xcat_ac225_psma_abdomen as old

HERE = Path(__file__).resolve().parent


def build(xif: Path, out: Path, total: int, workers: int):
    cfg = json.loads((HERE / "config.json").read_text())
    z0, z1 = cfg["xcat_crop_z"]
    if total % (20 * workers) or not (z1 - z0 == 80 and workers > 0):
        raise ValueError("Invalid XCAT job split")
    if out.exists():
        raise FileExistsError(out)
    dims, _, offset = old.parse_xif(xif)
    raw = np.memmap(xif, dtype="<f4", mode="r", offset=offset,
                    shape=(dims[2], dims[1], dims[0]))
    # Keep the full 256-column XCAT anatomy, centred in a 336-column canvas.
    codes = np.zeros((80, 200, 336), dtype=np.int16)
    codes[:, :, 40:296] = np.asarray(raw[z0:z1, 28:228, 0:256], dtype=np.int16)
    zz = np.arange(z0, z1)[:, None, None]
    yy = np.arange(28, 228)[None, :, None]
    xx = np.arange(-40, 296)[None, None, :]
    x = (xx - 127.5) * 1.5
    y = (yy - 127.5) * 1.5
    support = (((abs(x) + .75) / 250)**2 + ((abs(y) + .75) / 150)**2 <= 1)
    support = np.broadcast_to(support, codes.shape).copy()
    body = (codes != 0) & support
    masks = {"body": body, "kidney": (codes == 75) & support,
             "liver": (codes == 101) & support,
             "bowel": (codes == 102) & support,
             "spine_bone": (codes == 6) & support}
    dx, dy, dz = (xx - 132) * 1.5, (yy - 153) * 1.5, (zz - 795) * 1.5
    roi = (dx / 15)**2 + (dy / 12)**2 + (dz / 18)**2 <= 1
    lesion = roi & np.isin(codes, (2, 6)) & support
    masks["lesion"] = lesion
    profile = np.exp(-(dx*dx + dy*dy + dz*dz) /
                     (2 * (old.LESION_FWHM_MM / 2.354820045)**2))
    common = np.zeros(codes.shape, dtype=np.float32)
    common[body] = old.ACTIVITY["soft"]
    common[masks["bowel"]] = old.ACTIVITY["bowel"]
    common[masks["liver"]] = old.ACTIVITY["liver"]
    fr, bi = common.copy(), common.copy()
    fr[masks["kidney"]] = old.ACTIVITY["kidney_fr"]
    bi[masks["kidney"]] = old.ACTIVITY["kidney_bi"]
    for volume in (fr, bi):
        volume[lesion] = (old.ACTIVITY["soft"] +
                          (old.ACTIVITY["lesion"] - old.ACTIVITY["soft"]) * profile[lesion])
    fr3 = fr.reshape(40, 2, 100, 2, 168, 2).mean((1, 3, 5))
    bi3 = bi.reshape(40, 2, 100, 2, 168, 2).mean((1, 3, 5))
    out.mkdir(parents=True)
    np.savez_compressed(out / "truth_3mm.npz", fr_zyx=fr3, bi_zyx=bi3,
        fr_xyz=fr3.transpose(2, 1, 0), bi_xyz=bi3.transpose(2, 1, 0),
        x_mm=(np.arange(168) + .5 - 84) * 3,
        y_mm=(np.arange(100) + .5 - 50) * 3,
        z_mm=(np.arange(40) + .5 - 20) * 3,
        **{key + "_fraction_zyx": mask.reshape(40, 2, 100, 2, 168, 2).mean((1, 3, 5))
           for key, mask in masks.items()})
    np.savez_compressed(out / "masks_1p5mm.npz", xcat_codes_zyx=codes,
                        **{key + "_zyx": mask for key, mask in masks.items()})
    # A source box for every nonzero native voxel is exact but unnecessarily
    # large. Greedily merge equal-valued 3-mm cells; subdivide only at edge.
    def boxes(energy, target, native):
        result = []
        used = np.zeros(target.shape, bool)
        yy3, xx3 = np.ogrid[:100, :168]
        x3 = (xx3 + .5 - 84) * 3
        y3 = (yy3 + .5 - 50) * 3
        whole = ((abs(x3) + 1.5) / 250)**2 + ((abs(y3) + 1.5) / 150)**2 <= 1
        for k in range(40):
            for j in range(100):
                for i in range(168):
                    if used[k,j,i] or not whole[j,i] or target[k,j,i] <= 0: continue
                    val = target[k,j,i]
                    i1 = i+1
                    while i1 < 168 and whole[j,i1] and not used[k,j,i1] and target[k,j,i1] == val: i1 += 1
                    j1 = j+1
                    while j1 < 100 and np.all(whole[j1,i:i1] & ~used[k,j1,i:i1] & (target[k,j1,i:i1] == val)): j1 += 1
                    k1 = k+1
                    while k1 < 40 and np.all(whole[j:j1,i:i1] & ~used[k1,j:j1,i:i1] & (target[k1,j:j1,i:i1] == val)): k1 += 1
                    used[k:k1,j:j1,i:i1] = True
                    result.append(old.Box(energy, (i+i1-168)*1.5, (j+j1-100)*1.5,
                              (k+k1-40)*1.5, (i1-i)*1.5, (j1-j)*1.5,
                              (k1-k)*1.5, float(val)))
        for k,j,i in np.argwhere((~whole)[None] & (target > 0)):
            for dk in range(2):
                for dj in range(2):
                    for di in range(2):
                        nk,nj,ni=2*k+dk,2*j+dj,2*i+di
                        val=float(native[nk,nj,ni])
                        if val > 0 and support[nk,nj,ni]:
                            result.append(old.Box(energy, (ni+.5-168)*1.5,
                              (nj+.5-100)*1.5,(nk+.5-40)*1.5,.75,.75,.75,val))
        expected=float(target.sum(dtype=np.float64)*27*old.YIELDS[energy])
        actual=math.fsum(box.intensity for box in result)
        if not math.isclose(actual,expected,rel_tol=1e-6):
            raise AssertionError((energy,actual,expected))
        return result
    fr_boxes, bi_boxes = boxes(218, fr3, fr), boxes(440, bi3, bi)
    manifests=[]
    for view in range(20):
        path=out / f"view_{view+1:02d}_mixed.mac"
        with path.open("w",encoding="ascii") as stream:
            stream.write(f"/xcat/clear\n/xcat/centerY -345\n/xcat/angle {view*18}\n")
            for box in fr_boxes+bi_boxes:
                stream.write(f"/xcat/add {box.energy} {box.x:.9f} {box.y:.9f} {box.z:.9f} {box.hx:.9f} {box.hy:.9f} {box.hz:.9f} {box.intensity:.15g}\n")
            stream.write(f"/run/beamOn {total//(20*workers)}\n")
        manifests.append({"view":view+1,"file":path.name,"sha256":old.sha256_file(path)})
    kidney_full=int(np.count_nonzero(raw == 75))
    coverage={key:{"retained_voxels":int(mask.sum()),"touches_z_min":bool(mask[0].any()),
                    "touches_z_max":bool(mask[-1].any())} for key,mask in masks.items()}
    coverage["kidney"]["fraction_of_full_xcat_kidneys"]=int(masks["kidney"].sum())/kidney_full
    manifest={"experiment":cfg["experiment_id"],"xif_sha256":old.sha256_file(xif),
              "canvas_native_zyx":list(codes.shape),"truth_zyx":list(fr3.shape),
              "ellipse_semi_axes_mm":[250,150],"centerY_mm":-345,
              "crop_z_half_open":[z0,z1],"padding_native_x_each_side":40,
              "total_primary_photons":total,"workers_per_view":workers,
              "gamma_yields":cfg["gamma_yields"],"organ_coverage":coverage,
              "integrated_activity_mm3":{"fr":float(fr3.sum()*27),"bi":float(bi3.sum()*27)},
              "source_counts":{"fr":len(fr_boxes),"bi":len(bi_boxes)},"macros":manifests}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("source_counts","organ_coverage","integrated_activity_mm3")},indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--xif",type=Path,default=old.DEFAULT_XIF)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--total",type=int,default=1_000_000_000)
    parser.add_argument("--workers",type=int,default=10)
    a=parser.parse_args()
    build(a.xif,a.output,a.total,a.workers)
