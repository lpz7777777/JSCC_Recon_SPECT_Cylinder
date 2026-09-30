"""On scxi717, compare independent point CntStat with interpolated factor rows."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE / "generated"
FOLDERS = {218: "218keV_RotateNum20", 440: "440keV_RotateNum20"}
SELECTED = ("Point_center_z+00_218keV", "Point_center_z+00_440keV",
            *(f"Point_rho98_{axis}_z{z}_{energy}keV"
              for axis in ("a00", "a02") for z in ("-57", "+57")
              for energy in (218, 440)))


def similarity(observed, modeled):
    a = observed.ravel().astype(np.float64)
    b = modeled.ravel().astype(np.float64)
    return {"cosine": float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))),
            "pearson": float(np.corrcoef(a, b)[0, 1])}


def main():
    bundle = ROOT / "PointQA/selected_point_counts.zip"
    with np.load(ROOT / "Geometry/geometry.npz") as geometry:
        coords = geometry["coordinates_mm"]
        rotations = {"negative_angle": geometry["inverse_rotation"],
                     "positive_angle": geometry["rotation"]}
    factors = {}
    factor_manifests = {}
    detector_layers = {}
    for energy, folder in FOLDERS.items():
        base = ROOT / "FactorsCalibrated" / folder
        matrix = base / "SysMat_polar"
        if matrix.stat().st_size != 132040 * 10496 * 4:
            raise ValueError(f"Factor size mismatch: {energy}")
        factors[energy] = np.memmap(matrix, mode="r", dtype="<f4", shape=(132040, 10496))
        detector = np.loadtxt(base / "Detector.csv", delimiter=",", skiprows=1)
        detector_layers[energy] = np.abs(detector[:, 2])
        manifest = base / "factor_manifest.json"
        factor_manifests[str(energy)] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with zipfile.ZipFile(bundle) as archive:
        truth = {row["dataset"]: row for row in json.loads(archive.read("truth.json"))["sites"]}
        rows = []
        for dataset in SELECTED:
            site = truth[dataset]
            energy = site["energy_keV"]
            point = np.asarray(site["object_mm"], dtype=np.float64)
            file = f"CntStat/{energy}keV_RotateNum20_Geant4JSCC/CntStat_{dataset}_1e+07.csv"
            payload = archive.read(file)
            observed = np.loadtxt(io.BytesIO(payload), delimiter=",", dtype=np.float32)
            if observed.shape != (20, 10496) or observed.sum() <= 0:
                raise ValueError(f"Invalid point CntStat: {dataset}")
            squared = np.sum((coords - point) ** 2, axis=1)
            neighbors = np.argpartition(squared, 8)[:8]
            neighbors = neighbors[np.argsort(squared[neighbors])]
            distance = np.sqrt(squared[neighbors])
            if np.any(distance < 1e-9):
                weight = (distance < 1e-9).astype(np.float64)
            else:
                weight = 1 / np.maximum(distance, 1e-9) ** 2
            weight /= weight.sum()
            patterns = {}
            layer_comparison = None
            for sense, rotation in rotations.items():
                modeled_nearest = np.empty((20, 10496), np.float64)
                modeled_weighted = np.zeros((20, 10496), np.float64)
                factor = factors[energy]
                for view in range(20):
                    modeled_nearest[view] = factor[rotation[neighbors[0], view]]
                    for i, w in zip(neighbors, weight):
                        modeled_weighted[view] += w * factor[rotation[i, view]]
                patterns[sense] = {"nearest": similarity(observed, modeled_nearest),
                                   "weighted8": similarity(observed, modeled_weighted)}
                if sense == "negative_angle":
                    layer_comparison = []
                    for distance_mm in np.unique(detector_layers[energy]):
                        mask = detector_layers[energy] == distance_mm
                        layer_comparison.append({"detector_distance_mm": float(distance_mm),
                            "observed_counts": float(observed[:, mask].sum()),
                            "modeled_sum": float(modeled_weighted[:, mask].sum()),
                            "within_layer_cosine": similarity(observed[:, mask],
                                                              modeled_weighted[:, mask])["cosine"]})
                    observed_total = sum(row["observed_counts"] for row in layer_comparison)
                    modeled_total = sum(row["modeled_sum"] for row in layer_comparison)
                    for layer in layer_comparison:
                        layer["observed_fraction"] = layer["observed_counts"] / observed_total
                        layer["modeled_fraction"] = layer["modeled_sum"] / modeled_total
            rows.append({"dataset": dataset, "energy_keV": energy,
                         "object_mm": point.tolist(), "neighbors_mm": coords[neighbors].tolist(),
                         "neighbor_weights": weight.tolist(),
                         "cntstat_sha256": hashlib.sha256(payload).hexdigest(),
                         "observed_counts": int(observed.sum()), "orientation": patterns,
                         "layers": layer_comparison})
            print(dataset, "negative nearest", round(patterns["negative_angle"]["nearest"]["cosine"], 4),
                  "weighted8", round(patterns["negative_angle"]["weighted8"]["cosine"], 4),
                  flush=True)
    report = {"experiment": "ELLIPSE500x300_H120", "method": "8 nearest 3D polar centers, inverse squared-distance weighted factor rows",
              "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
              "factor_manifest_sha256": factor_manifests, "rows": rows}
    output = ROOT / "PointQA/point_factor_interpolation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
