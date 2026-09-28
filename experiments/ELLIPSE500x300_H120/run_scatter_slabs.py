"""Run legacy ScatterGen in four exact axial slabs to avoid 32-bit overflow.

ScatterGen indexes each (detector, voxel) matrix with signed 32-bit integers.
The complete 85x85x40 matrix has 3.33 billion elements and overflows; each
85x85x10 slab has 832.32 million elements. Source voxel world z is preserved
by setting Params_Image[10] to the slab centre. No spatial sampling changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = ROOT / "Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
SUFFIX = "_pe_v4_ELLIPSE500x300_H120"
STEM = "shift_0.000000_0.000000_0.000000"
NDET, NZ, NY, NX = 11520, 40, 85, 85
SLABS = 4


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def input_paths(response):
    name = {"A218": "JSCC_218keV", "A440": "JSCC_440keV",
            "C440to218": "JSCC_440keV_to_218keVwin"}[response]
    folder = ENGINE / "runs" / (name + SUFFIX)
    pe_folder = ENGINE / "runs" / (("JSCC_218keV" if response == "A218" else "JSCC_440keV") + SUFFIX)
    return folder, pe_folder / f"PE_SysMat_{STEM}_v4.sysmat"


def prepare(response, index):
    if index not in range(SLABS):
        raise ValueError("Slab index must be 0..3")
    folder, pe = input_paths(response)
    if pe.stat().st_size != NDET * NZ * NY * NX * 4:
        raise ValueError("Complete unwindowed PE matrix required")
    output = HERE / "generated/ScatterSlabs" / response / f"slab{index:02d}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    parameters = {}
    for source in sorted(folder.glob("Params_*.dat")):
        if source.name == "Params_Image.dat":
            image = np.fromfile(source, dtype="<f4")
            if not np.array_equal(image[:7], [85,85,40,6,6,3,1]) or image[11] != 270:
                raise ValueError("Source Params_Image does not match ellipse experiment")
            image[2] = 10
            image[10] = -45 + 30 * index
            image.tofile(output / source.name)
        else:
            shutil.copy2(source, output / source.name)
        parameters[source.name] = digest(output / source.name)
    if len(parameters) < 4:
        raise ValueError("Missing ScatterGen parameters")
    data = np.memmap(pe, dtype="<f4", mode="r", shape=(NDET, NZ, NY, NX))
    sliced_pe = output / f"PE_SysMat_slab{index:02d}.sysmat"
    z0, z1 = index * 10, (index + 1) * 10
    with sliced_pe.open("wb") as sink:
        for detector in range(0, NDET, 64):
            data[detector:detector+64,z0:z1].tofile(sink)
    del data
    if sliced_pe.stat().st_size != NDET * 10 * NY * NX * 4:
        raise ValueError("Slab PE byte count mismatch")
    (output / "input_manifest.json").write_text(json.dumps({
        "response": response, "slab_index": index, "z_source_slices_half_open": [z0,z1],
        "world_z_centres_mm": [-58.5+3*z0,-58.5+3*(z1-1)],
        "pe_parent": str(pe), "pe_parent_sha256": digest(pe),
        "sliced_pe_sha256": digest(sliced_pe), "params_sha256": parameters}, indent=2)+"\n")
    print(output, flush=True)
    return output


def run(response, index, binary, cuda):
    folder = HERE / "generated/ScatterSlabs" / response / f"slab{index:02d}"
    receipt = json.loads((folder / "input_manifest.json").read_text())
    pe = folder / f"PE_SysMat_slab{index:02d}.sysmat"
    if digest(pe) != receipt["sliced_pe_sha256"]:
        raise ValueError("Slab input changed")
    with (folder / "Scatter_console.log").open("w") as log:
        subprocess.run([str(binary.resolve()), "-PE", str(pe.resolve()),
                        "-cuda", str(cuda)], cwd=folder, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    shift = -45 + 30 * index
    stem = f"shift_0.000000_0.000000_{shift:.6f}"
    scatter = folder / f"Scatter_SysMat_{stem}.sysmat"
    combined = folder / f"SysMat_withScatter_{stem}.sysmat"
    expected = NDET * 10 * NY * NX * 4
    if scatter.stat().st_size != expected:
        raise ValueError("Incomplete scatter slab")
    if response != "C440to218" and combined.stat().st_size != expected:
        raise ValueError("Incomplete combined slab")
    (folder / "complete.json").write_text(json.dumps({
        "response": response, "slab_index": index,
        "scatter_sha256": digest(scatter),
        "combined_sha256": digest(combined) if combined.exists() else None,
        "binary_sha256": digest(binary)}, indent=2)+"\n")
    print(f"Scatter slab complete: {response}/{index}", flush=True)


def stitch(response):
    folder, _ = input_paths(response)
    base = HERE / "generated/ScatterSlabs" / response
    expected = NDET * NZ * NY * NX * 4
    for kind in (("Scatter_SysMat",) if response == "C440to218" else
                 ("Scatter_SysMat", "SysMat_withScatter")):
        target = folder / f"{kind}_{STEM}.sysmat"
        if target.exists():
            raise FileExistsError(target)
        staging = target.with_suffix(target.suffix + ".building")
        output = np.memmap(staging, mode="w+", dtype="<f4", shape=(NDET,NZ,NY,NX))
        for index in range(SLABS):
            slab = base / f"slab{index:02d}"
            receipt = json.loads((slab / "complete.json").read_text())
            shift = -45 + 30 * index
            source = slab / f"{kind}_shift_0.000000_0.000000_{shift:.6f}.sysmat"
            key = "scatter_sha256" if kind == "Scatter_SysMat" else "combined_sha256"
            if digest(source) != receipt[key]:
                raise ValueError(f"Changed scatter slab: {source}")
            data = np.memmap(source, mode="r", dtype="<f4", shape=(NDET,10,NY,NX))
            for detector in range(0,NDET,64):
                output[detector:detector+64,index*10:(index+1)*10] = data[detector:detector+64]
            output.flush()
            del data
        del output
        if staging.stat().st_size != expected:
            raise ValueError("Stitched response has wrong byte count")
        os.replace(staging,target)
        print(f"Stitched {target} sha256={digest(target)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("prepare", "run", "stitch"))
    p.add_argument("response", choices=("A218", "A440", "C440to218"))
    p.add_argument("index", type=int, nargs="?")
    p.add_argument("--binary", type=Path,
      default=ENGINE / "ScatterGen_RayTracing_CircularHole/ScatterGen_CircularHole_detector_local")
    p.add_argument("--cuda", type=int, default=0)
    a = p.parse_args()
    if a.action == "stitch":
        stitch(a.response)
    elif a.index is None:
        p.error("prepare/run requires an index")
    elif a.action == "prepare":
        prepare(a.response,a.index)
    else:
        run(a.response,a.index,a.binary,a.cuda)
