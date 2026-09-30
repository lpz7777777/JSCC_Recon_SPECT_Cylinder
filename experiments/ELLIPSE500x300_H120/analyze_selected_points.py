"""Validate selected Geant4 point collections and compare detection efficiency."""
import argparse
import json
from pathlib import Path

import numpy as np

DATASETS=("Point_center_z+00_218keV","Point_center_z+00_440keV",
          "Point_rho98_a00_z-57_218keV","Point_rho98_a00_z-57_440keV",
          "Point_rho98_a00_z+57_218keV","Point_rho98_a00_z+57_440keV",
          "Point_rho98_a02_z-57_218keV","Point_rho98_a02_z-57_440keV",
          "Point_rho98_a02_z+57_218keV","Point_rho98_a02_z+57_440keV")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("point_root",type=Path)
    args=p.parse_args()
    root=args.point_root.resolve()
    collected=root/"SelectedCollected"
    truth={row["dataset"]:row for row in json.loads((root/"truth.json").read_text())["sites"]}
    rows=[]
    all_workers=[]
    all_seeds=[]
    for dataset in DATASETS:
        marker=json.loads((collected/"collections"/f"{dataset}_1e+07.json").read_text())
        if (marker["dataset"]!=dataset or marker["level"]!="1e+07" or
            marker["views"]!=list(range(1,21)) or
            sum(marker["primary_counts"])!=10**7 or
            len(marker["worker_indices"])!=20 or len(marker["seeds"])!=20):
            raise ValueError(f"Incomplete point collection {dataset}")
        all_workers.extend(marker["worker_indices"])
        all_seeds.extend(marker["seeds"])
        counts={}
        for energy in (218,440):
            file=collected/"CntStat"/f"{energy}keV_RotateNum20_Geant4JSCC"/f"CntStat_{dataset}_1e+07.csv"
            data=np.loadtxt(file,delimiter=",",dtype=np.int64)
            if data.shape!=(20,10496) or np.any(data<0):
                raise ValueError(f"Invalid detector counts: {file}")
            counts[str(energy)]=int(data.sum())
        source_energy=truth[dataset]["energy_keV"]
        direct=counts[str(source_energy)]
        rows.append({"dataset":dataset,"source_energy_keV":source_energy,
                     "object_mm":truth[dataset]["object_mm"],
                     "normalized_ellipse_radius":truth[dataset]["normalized_ellipse_radius"],
                     "primary_photons":10**7,"detected_counts":counts,
                     "direct_counts":direct,"direct_detection_efficiency":direct/10**7,
                     "cross_window_counts":counts[str(218 if source_energy==440 else 440)]})
    if len(set(all_workers))!=200 or len(set(all_seeds))!=200:
        raise ValueError("Duplicate point workers or seeds")
    centers={row["source_energy_keV"]:row["direct_detection_efficiency"]
             for row in rows if row["dataset"].startswith("Point_center_")}
    for row in rows:
        row["direct_efficiency_over_center"]=(row["direct_detection_efficiency"]/
                                               centers[row["source_energy_keV"]])
    report={"experiment":"ELLIPSE500x300_H120","point_datasets":len(rows),
            "views_per_dataset":20,"workers":len(all_workers),
            "total_primary_photons":10**8,"reference":"same-energy center at z=0",
            "rows":rows}
    out=collected/"selected_point_efficiency.json"
    out.write_text(json.dumps(report,indent=2)+"\n")
    print("ELLIPSE_SELECTED_POINTS_OK",out)
    for row in rows:
        print(row["dataset"],"counts",row["direct_counts"],
              "ratio",f"{row['direct_efficiency_over_center']:.4f}")

if __name__=="__main__":main()
