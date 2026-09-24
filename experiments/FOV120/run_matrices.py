"""Run existing PE-v4 and detector-local scatter binaries on isolated FOV120 Params."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import hashlib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fov_config import load_config

ENGINE = ROOT/"Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
STEM = "shift_0.000000_0.000000_0.000000"


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def run(pe, scatter, cuda):
    cfg = load_config()
    expected = 11520*51*51*cfg["z_layers"]*4
    folders = [ENGINE/"runs"/(name+"_pe_v4_FOV120") for name in
               ("JSCC_218keV","JSCC_440keV","JSCC_440keV_to_218keVwin")]
    executable_hashes={"pe":checksum(pe),"scatter":checksum(scatter)}
    for folder in folders:
        params = np.fromfile(folder/"Params_Image.dat",dtype="<f4")
        if len(params)!=12 or not np.array_equal(params[:7], [51,51,40,6,6,3,1]) or params[11]!=170:
            raise ValueError(f"Wrong FOV120 Params: {folder}")
        provenance={"executables":executable_hashes,
                    "parameters":{path.name:checksum(path) for path in sorted(folder.glob('Params_*.dat'))}}
        receipt=folder/'FOV120_inputs.json'
        if receipt.exists():
            if json.loads(receipt.read_text())!=provenance:
                raise ValueError(f"Inputs changed since matrix calculation: {folder}; use fresh outputs")
        else:
            if list(folder.glob('*.sysmat')):
                raise ValueError(f"Cannot reuse matrices without input provenance: {folder}")
            receipt.write_text(json.dumps(provenance,indent=2)+'\n')
    def execute(command, folder, log):
        with (folder/log).open("w") as stream:
            subprocess.run(command,cwd=folder,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def check(path):
        if path.stat().st_size != expected:
            raise ValueError(f"Incomplete matrix: {path}")
    for folder in folders[:2]:
        raw = folder/f"PE_SysMat_{STEM}_v4.sysmat"
        windowed = folder/f"PE_Windowed_SysMat_{STEM}_v4.sysmat"
        if raw.exists() or windowed.exists():
            check(raw); check(windowed)
            metadata = json.loads((folder/"PE_v4_manifest.json").read_text())
            if metadata["voxel_count"] != 104040 or metadata["face_subdivisions"] != 16 or metadata["model"] != "PE_v4_visible_surface_symmetric_halton_layer_grid":
                raise ValueError("Existing PE output has different physics/grid")
        else:
            execute([str(pe.resolve()),"--cuda",str(cuda),"--face-subdiv","16","--rows-per-chunk","4",
                     "--samples-per-launch","32","--output-unwindowed",str(raw),"--output-windowed",str(windowed),
                     "--manifest",str(folder/"PE_v4_manifest.json"),"--progress",str(folder/"PE_progress.json"),
                     "--log",str(folder/"PE_progress.tsv")],folder,"PE_console.log")
            check(raw);check(windowed)
    for index,folder in enumerate(folders):
        source = folders[min(index,1)]/f"PE_SysMat_{STEM}_v4.sysmat"
        result = folder/f"Scatter_SysMat_{STEM}.sysmat"
        combined = folder/f"SysMat_withScatter_{STEM}.sysmat"
        if result.exists():
            check(result)
            if result.stat().st_mtime < source.stat().st_mtime:
                raise ValueError("Scatter predates PE; use a fresh run directory")
        else:
            execute([str(scatter.resolve()),"-PE",str(source),"-cuda",str(cuda)],folder,"Scatter_console.log")
            check(result)
        if index < 2:
            check(combined)
    print("All three FOV120 response matrices completed and size checked")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pe",type=Path,required=True)
    parser.add_argument("--scatter",type=Path,required=True)
    parser.add_argument("--cuda",type=int,default=0)
    args=parser.parse_args()
    run(args.pe,args.scatter,args.cuda)
