"""Report measured four-layer Geant4 pilot counts before response fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    raw=np.fromfile(a.input/"Params_Detector.dat",dtype="<f4")
    detector=raw[1:].reshape(int(raw[0]),12)
    layer=detector[detector[:,11]==1,1]+270
    if len(layer)!=10496 or not np.array_equal(np.unique(layer),[300,330,360,390]):
        raise ValueError("Detector layer geometry mismatch")
    records={}
    sources=(("A218",218,"218_direct","calibration_218_pilot"),
             ("A440",440,"440_direct","calibration_440_pilot"),
             ("C440to218",440,"440_cross","calibration_440_pilot"))
    for response,energy,name,collection in sources:
        counts=np.loadtxt(a.input/f"CntStat_{name}.csv",delimiter=",").reshape(-1)
        meta=json.loads((a.input/f"{collection}.json").read_text())
        primary=meta["primary_counts"]
        if int(sum(primary))!=100_000_000 or primary[0 if energy==440 else 1]:
            raise ValueError("Pilot source count or mono energy mismatch")
        layers=[]
        for position in (300,330,360,390):
            n=int(counts[layer==position].sum())
            layers.append({"normal_distance_mm":position,"counts":n,
                           "measured_efficiency":n/sum(primary),
                           "poisson_relative_se":n**-.5 if n else None})
        records[response]={"energy_keV":energy,"primary_counts":primary,
                           "layers":layers,"meets_1_percent_statistical_gate":
                           all(row["counts"]>=10000 for row in layers)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    report={"experiment":"ELLIPSE500x300_H120",
            "calibration_installed":False,"responses":records,
            "input_sha256":{path.name:digest(path) for path in sorted(a.input.iterdir()) if path.is_file()}}
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v["meets_1_percent_statistical_gate"] for k,v in records.items()}))


if __name__=="__main__":main()
