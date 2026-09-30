"""Compare independent point-source detector patterns to calibrated factors."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "FOV120"))
from reconstruction_ssh import connect

REMOTE = ("/data/run01/scxi717/lpz/"
          "20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/"
          "experiments/ELLIPSE500x300_H120/generated/FactorsCalibrated")
FOLDERS = {218: "218keV_RotateNum20", 440: "440keV_RotateNum20"}
SELECTED = ("Point_center_z+00_218keV", "Point_center_z+00_440keV",
            *(f"Point_rho98_{axis}_z{z}_{energy}keV"
              for axis in ("a00", "a02") for z in ("-57", "+57")
              for energy in (218, 440)))


def similarity(measured, modeled):
    a = measured.reshape(-1).astype(np.float64)
    b = modeled.reshape(-1).astype(np.float64)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    centered = float(np.corrcoef(a, b)[0, 1])
    scale = float(a.sum() / b.sum())
    return {"cosine": cosine, "pearson": centered, "count_scale": scale,
            "modeled_scaled_sum": float((b * scale).sum())}


def main():
    truth = {row["dataset"]: row for row in json.loads(
        (HERE / "generated/PointScan/truth.json").read_text())["sites"]}
    with np.load(HERE / "generated/Geometry/geometry.npz") as geometry:
        coordinates = geometry["coordinates_mm"]
        inverse = geometry["inverse_rotation"]
        forward = geometry["rotation"]
    rows = []
    with connect() as ssh, ssh.open_sftp() as sftp:
        streams = {energy: sftp.open(f"{REMOTE}/{folder}/SysMat_polar", "rb")
                   for energy, folder in FOLDERS.items()}
        try:
            for energy, stream in streams.items():
                if stream.stat().st_size != 132040 * 10496 * 4:
                    raise ValueError(f"Calibrated factor size mismatch: {energy}")
            for dataset in SELECTED:
                source = truth[dataset]
                energy = source["energy_keV"]
                point = np.asarray(source["object_mm"], dtype=np.float64)
                index = int(np.argmin(np.sum((coordinates - point) ** 2, axis=1)))
                observed_file = (HERE / "generated/PointSelectedCntStat" /
                    f"{energy}keV_RotateNum20_Geant4JSCC" /
                    f"CntStat_{dataset}_1e+07.csv")
                measured = np.loadtxt(observed_file, delimiter=",", dtype=np.float32)
                if measured.shape != (20, 10496) or int(measured.sum()) <= 0:
                    raise ValueError(f"Invalid independent detector counts: {dataset}")
                patterns = {}
                for sense, mapping in (("physical_negative_angle", inverse),
                                       ("opposite_positive_angle", forward)):
                    modeled = np.empty((20, 10496), dtype=np.float32)
                    stream = streams[energy]
                    for view in range(20):
                        stream.seek(int(mapping[index, view]) * 10496 * 4)
                        block = stream.read(10496 * 4)
                        if len(block) != 10496 * 4:
                            raise ValueError("Short factor row")
                        modeled[view] = np.frombuffer(block, dtype="<f4")
                    patterns[sense] = similarity(measured, modeled)
                rows.append({"dataset": dataset, "energy_keV": energy,
                             "object_mm": point.tolist(),
                             "nearest_factor_mm": coordinates[index].tolist(),
                             "nearest_distance_mm": float(np.linalg.norm(coordinates[index] - point)),
                             "measured_counts": int(measured.sum()),
                             "cntstat_sha256": hashlib.sha256(observed_file.read_bytes()).hexdigest(),
                             "orientation": patterns})
                print(dataset, "cosine negative", round(patterns["physical_negative_angle"]["cosine"], 4),
                      "positive", round(patterns["opposite_positive_angle"]["cosine"], 4))
        finally:
            for stream in streams.values():
                stream.close()
    report = {"experiment": "ELLIPSE500x300_H120", "datasets": len(rows),
              "factor_interpolation": "nearest polar center; no interpolation",
              "comparison": "flattened 20-view direct-window detector pattern",
              "rows": rows}
    output = HERE / "reports/selected_point_factor_alignment.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
