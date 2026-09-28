"""Produce two PE-v4 matrices; use run_scatter_slabs.py for scatter.

The legacy monolithic ScatterGen uses signed 32-bit detector×voxel indices;
85×85×40×11520 exceeds that range. Never run it on this complete grid.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
SUFFIX = "_pe_v4_ELLIPSE500x300_H120"
STEM = "shift_0.000000_0.000000_0.000000"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def run(pe, scatter, cuda):
    config = json.loads((Path(__file__).with_name("config.json")).read_text())
    expected = 11520 * 85 * 85 * 40 * 4
    folders = [ENGINE / "runs" / (name + SUFFIX) for name in
               ("JSCC_218keV", "JSCC_440keV", "JSCC_440keV_to_218keVwin")]
    binaries = {"pe": digest(pe), "scatter": digest(scatter)}
    for folder in folders:
        params = np.fromfile(folder / "Params_Image.dat", dtype="<f4")
        if (not np.array_equal(params[:7], [85, 85, 40, 6, 6, 3, 1]) or
                params[11] != config["fov2collimator0_mm"]):
            raise ValueError(f"Wrong ellipse Params: {folder}")
        receipt = folder / "ELLIPSE_inputs.json"
        provenance = {"executables": binaries,
                      "parameters": {p.name: digest(p) for p in sorted(folder.glob("Params_*.dat"))}}
        if receipt.exists():
            if json.loads(receipt.read_text()) != provenance:
                raise ValueError(f"Input changed: {folder}")
        else:
            if list(folder.glob("*.sysmat")):
                raise ValueError(f"Existing matrix lacks provenance: {folder}")
            receipt.write_text(json.dumps(provenance, indent=2) + "\n")

    def execute(command, folder, log):
        with (folder / log).open("w") as stream:
            subprocess.run(command, cwd=folder, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    def check(path):
        if path.stat().st_size != expected:
            raise ValueError(f"Incomplete matrix: {path}")

    for folder in folders[:2]:
        raw = folder / f"PE_SysMat_{STEM}_v4.sysmat"
        windowed = folder / f"PE_Windowed_SysMat_{STEM}_v4.sysmat"
        if raw.exists() or windowed.exists():
            check(raw)
            check(windowed)
            metadata = json.loads((folder / "PE_v4_manifest.json").read_text())
            if (metadata["voxel_count"] != 85 * 85 * 40 or
                    metadata["face_subdivisions"] != 16 or
                    metadata["model"] != "PE_v4_visible_surface_symmetric_halton_layer_grid"):
                raise ValueError("Existing PE output has incompatible model/grid")
        else:
            execute([str(pe.resolve()), "--cuda", str(cuda), "--face-subdiv", "16",
                     "--rows-per-chunk", "4", "--samples-per-launch", "32",
                     "--output-unwindowed", str(raw), "--output-windowed", str(windowed),
                     "--manifest", str(folder / "PE_v4_manifest.json"),
                     "--progress", str(folder / "PE_progress.json"),
                     "--log", str(folder / "PE_progress.tsv")], folder, "PE_console.log")
            check(raw)
            check(windowed)
    print("Both PE-v4 matrices completed and size checked. "
          "Run four axial slabs per response with run_scatter_slabs.py, "
          "then stitch and validate before Factors conversion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe", type=Path, required=True)
    parser.add_argument("--scatter", type=Path, required=True)
    parser.add_argument("--cuda", type=int, default=0)
    args = parser.parse_args()
    run(args.pe, args.scatter, args.cuda)
