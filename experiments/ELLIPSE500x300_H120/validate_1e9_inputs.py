"""Gate XCAT transport on complete three-way 1e9 rule-phantom input closure."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from simulation import rods


def expected_218_fraction(dataset,config):
    volume=(math.pi*150**2*120 if dataset=="CircleNewDist" else
            math.pi*250*150*120)
    weights={energy:config["gamma_yields"][str(energy)]*volume for energy in (218,440)}
    if dataset=="EllipseContrast":
        for rod in rods():
            weights[rod["energy"]]+=5*config["gamma_yields"][str(rod["energy"])]*\
                math.pi*rod["radius_mm"]**2*rod["height_mm"]
    return weights[218]/sum(weights.values())


def validate(root):
    config=json.loads((HERE/"config.json").read_text())
    results={}
    for dataset in ("CircleNewDist","EllipseUniform","EllipseContrast"):
        collection=json.loads((root/"collections"/f"{dataset}_1e9.json").read_text())
        if collection["views"]!=list(range(1,21)) or len(collection["worker_indices"])!=200:
            raise ValueError(f"Incomplete views/workers: {dataset}")
        primary=collection["primary_counts"]
        if sum(primary)!=1_000_000_000 or primary[2]:
            raise ValueError(f"Primary count closure failure: {dataset}")
        expected=expected_218_fraction(dataset,config)
        observed=primary[0]/sum(primary)
        if abs(observed-expected)>.001:
            raise ValueError(f"Gamma yield ratio mismatch: {dataset} {observed} vs {expected}")
        counts={}
        for energy in (218,440):
            path=root/"CntStat"/f"{energy}keV_RotateNum20_Geant4JSCC"/\
                 f"CntStat_{dataset}_1e9.csv"
            values=np.loadtxt(path,delimiter=",",dtype=np.int64)
            if values.shape!=(20,10496) or np.any(values<0) or np.any(values.sum(axis=1)<=0):
                raise ValueError(f"Invalid detector counts: {path}")
            counts[str(energy)]=int(values.sum())
        list_dir=root/"List/218-440keV_RotateNum20_Geant4JSCC"/f"List_{dataset}_1e9"
        if any(not (list_dir/f"{view}.csv").is_file() for view in range(1,21)):
            raise ValueError(f"Missing Compton List view: {dataset}")
        results[dataset]={"primary_counts":primary,"expected_218_fraction":expected,
                          "observed_218_fraction":observed,"window_counts":counts,
                          "collection_manifest_sha256":collection["manifest_sha256"]}
    return results


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    if a.report.exists():raise FileExistsError(a.report)
    report=validate(a.root)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+"\n")
    print("ELLIPSE_RULE_1E9_INPUT_CLOSURE_PASSED")
