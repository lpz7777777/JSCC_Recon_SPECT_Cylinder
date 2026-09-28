"""Compute full-circle K*B Sensi_d, then record ellipse-effective sensitivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from geometry import generate
from validate_factors import validate


def digest(path):
    from hashlib import sha256
    h=sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-root",type=Path,required=True)
    p.add_argument("--factor-root",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    a=p.parse_args()
    validate(a.factor_root)
    cfg=json.loads((HERE/"config.json").read_text())
    records=[]
    for dataset in ("sensitivity_440","sensitivity_validation_440"):
        record=json.loads((a.data_root/"collections"/f"{dataset}_all.json").read_text())
        if record["primary_counts"][0] or record["primary_counts"][2]:
            raise ValueError("Sensitivity source must be pure 440-keV photons")
        if sum(record["primary_counts"])<1_000_000_000:
            raise ValueError("At least 1e9 photons per sensitivity dataset required")
        records.append(record)
    if set(records[0]["seeds"]) & set(records[1]["seeds"]):
        raise ValueError("Sensitivity training and validation seeds overlap")
    factor=a.factor_root/"440keV_RotateNum20"
    output=a.data_root/"Sensitivity"
    if output.exists() or (factor/"Sensi_d").exists():
        raise FileExistsError("Sensitivity output already exists")
    tool=ROOT/"Auxiliary_Studies/Sensitivity_SPECT_PolarCoor"
    def list_path(dataset):
        return a.data_root/"List/218-440keV_RotateNum20_Geant4JSCC"/f"List_{dataset}_all/1.csv"
    subprocess.run([sys.executable,str(tool/"run_compton_sensitivity.py"),
      "--factor-dir",str(factor),"--compton-list",str(list_path("sensitivity_440")),
      "--source-photons",str(records[0]["primary_counts"][1]),
      "--energy-mev","0.440","--rotate-num","20",
      "--energy-resolution-662kev","0.13",
      "--energy-resolution-reference-kev","511",
      "--energy-threshold-sum-mev","0.350",
      "--input-energies-already-smeared","--device",a.device,
      "--output-dir",str(output)],check=True)
    subprocess.run([sys.executable,str(tool/"validate_uniform_compton_closure.py"),
      "--factor-dir",str(factor),"--compton-list",str(list_path("sensitivity_validation_440")),
      "--source-photons",str(records[1]["primary_counts"][1]),
      "--sensi-d",str(output/"Sensi_d"),"--event-start-fraction","0",
      "--event-fraction","1","--device",a.device,
      "--output-dir",str(output/"IndependentCircleClosure")],check=True)
    values=np.fromfile(output/"Sensi_d",dtype="<f4")
    if len(values)!=132040 or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError("Computed complete-grid Sensi_d is invalid")
    _,volume,fraction,active,_,_=generate(cfg)
    effective_volume=float(np.dot(volume,fraction))
    full_volume=float(volume.sum())
    report={"full_circle_mean_efficiency":float(values.sum()/full_volume),
            "ellipse_predicted_mean_efficiency":float(np.dot(values,fraction)/effective_volume),
            "full_circle_volume_mm3":full_volume,"ellipse_effective_volume_mm3":effective_volume,
            "active_columns":len(active),
            "zero_sensitivity_columns":int(np.count_nonzero(values==0)),
            "note":"Ellipse prediction is independent of the validation source; compare to accepted independent ellipse events after filtering."}
    (output/"ellipse_effective_sensitivity.json").write_text(json.dumps(report,indent=2)+"\n")
    shutil.copy2(output/"Sensi_d",factor/"Sensi_d")
    provenance={"experiment":cfg["experiment_id"],"operator":"K*B",
                "pixel_count":len(values),"resolution_fwhm":.13,
                "reference_keV":511,"sum_threshold_MeV":.350,
                "input_already_smeared":True,
                "source_photons":records[0]["primary_counts"][1],
                "independent_validation_seeds":records[1]["seeds"],
                "ellipse_effective_sensitivity":report,
                "hashes":{name:digest(factor/name) for name in
                ("Sensi_d","factor_manifest.json","coor_polar_full.csv","Detector.csv",
                 "polar_cell_volume_mm3.float64","SysMat_polar")}}
    (factor/"Sensi_d_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
