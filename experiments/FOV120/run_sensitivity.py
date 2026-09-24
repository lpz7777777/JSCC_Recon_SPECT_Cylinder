"""Compute new FOV120 K*B Sensi_d and validate against independent workers."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from fov_config import load_config, validate_factor_geometry
from workflow import digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root",type=Path,default=Path(__file__).resolve().parent/"generated")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    args=parser.parse_args()
    root=args.data_root.resolve()
    factor=root/"Factors/440keV_RotateNum20"
    validate_factor_geometry({"440":factor},load_config(),scan_matrix=True)
    records=[]
    for dataset in ("sensitivity_440","sensitivity_validation_440"):
        record=json.loads((root/"collections"/f"{dataset}_all.json").read_text())
        if record["primary_counts"][0] or record["primary_counts"][2]:
            raise ValueError("Sensi_d requires pure 440 primary data")
        records.append(record)
    if set(records[0]["seeds"]) & set(records[1]["seeds"]):
        raise ValueError("Sensitivity and validation use overlapping seeds")
    tool=ROOT/"Auxiliary_Studies/Sensitivity_SPECT_PolarCoor"
    out=root/"Sensitivity"
    lists=[root/"List/218-440keV_RotateNum20_Geant4JSCC"/f"List_{d}_all"/"1.csv"
           for d in ("sensitivity_440","sensitivity_validation_440")]
    subprocess.run([sys.executable,str(tool/"run_compton_sensitivity.py"),"--factor-dir",str(factor),
                    "--compton-list",str(lists[0]),"--source-photons",str(records[0]["primary_counts"][1]),
                    "--energy-mev","0.440","--rotate-num","20","--energy-resolution-662kev","0.13",
                    "--energy-resolution-reference-kev","511","--energy-threshold-sum-mev","0.350",
                    "--input-energies-already-smeared","--device",args.device,"--output-dir",str(out)],check=True)
    subprocess.run([sys.executable,str(tool/"validate_uniform_compton_closure.py"),"--factor-dir",str(factor),
                    "--compton-list",str(lists[1]),"--source-photons",str(records[1]["primary_counts"][1]),
                    "--sensi-d",str(out/"Sensi_d"),"--event-start-fraction","0","--event-fraction","1",
                    "--device",args.device,"--output-dir",str(out/"IndependentClosure")],check=True)
    # Preserve the independent closure metrics; they are scientific evidence, not a pass flag.
    import numpy as np
    import shutil
    values=np.fromfile(out/"Sensi_d",dtype="<f4")
    if len(values)!=51240 or not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError("Invalid computed sensitivity")
    if (factor/"Sensi_d").exists():
        raise FileExistsError("Refusing to replace an installed sensitivity")
    shutil.copy2(out/"Sensi_d",factor/"Sensi_d")
    provenance={"experiment":"FOV120","operator":"K*B","pixel_count":len(values),
                "resolution_fwhm":.13,"reference_keV":511,"sum_threshold_MeV":.350,
                "input_already_smeared":True,"source_photons":records[0]["primary_counts"][1],
                "independent_validation_seeds":records[1]["seeds"],
                "hashes":{name:digest(factor/name) for name in ("Sensi_d","factor_manifest.json",
                    "coor_polar_full.csv","Detector.csv","polar_cell_volume_mm3.float64","SysMat_polar")}}
    (factor/"Sensi_d_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")


if __name__=="__main__":
    main()
