"""Verify NEMA Geant4 macro transfer and completed worker accounting."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def validate(manifest_path: Path, stage: str):
    manifest = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    jobs = manifest["jobs"]
    doses = {"1e9": 1_000_000_000, "5e9": 5_000_000_000, "1e10": 10_000_000_000}
    expected_total = doses[manifest["level"]]
    worker_count = 20*manifest["workers_per_view"]
    if (manifest["dataset"] != "NEMA_Body_H60" or
        manifest["total_primary_photons"] != expected_total or
        len(jobs) != worker_count or
        len(manifest["macros"]) != 20 or
        [row["index"] for row in jobs] != list(range(worker_count)) or
        len({row["seed"] for row in jobs}) != worker_count or
        sorted({row["view"] for row in jobs}) != list(range(1, 21)) or
        sum(row["photons"] for row in jobs) != expected_total or
        any(row["level"] != manifest["level"] or row["dataset"] != manifest["dataset"] for row in jobs)):
        raise ValueError("NEMA primary/view/seed closure failed")
    for item in manifest["macros"]:
        path = root / item["path"]
        if (path.stat().st_size != item["bytes"] or
            digest(path) != item["sha256"]):
            raise ValueError(f"Transferred macro changed: {path}")
        text = path.read_text(encoding="ascii")
        if (text.count("/xcat/angle ") != 1 or
            text.splitlines().count(f"/run/beamOn {jobs[0]['photons']}") != 1 or
            text.count("/xcat/add ") !=
              sum(manifest["source_boxes_by_energy"].values())):
            raise ValueError(f"Malformed source macro: {path}")
    result = {"stage": stage, "macro_count": 20,
              "total_primary_photons": manifest["total_primary_photons"],
              "expected_primary_energy_fraction":
                  manifest["expected_primary_energy_fraction"]}
    if stage != "prepared":
        selected = jobs[:1] if stage in ("smoke", "first") else jobs
        category = "smoke" if stage == "smoke" else "workers"
        primaries = np.zeros(3, dtype=np.int64)
        records = []
        for job in selected:
            folder = root / category / f"{job['index']:05d}"
            record = json.loads((folder / "worker.json").read_text())
            if (record["status"] != "complete" or
                record["seed"] != job["seed"] or
                record["macro_sha256"] != job["macro_sha256"] or
                record["view"] != job["view"]):
                raise ValueError(f"Worker provenance failed: {folder}")
            actual = np.loadtxt(folder / "PrimaryCount.csv", delimiter=",",
                                dtype=np.int64, ndmin=2)
            photons = min(job["photons"], 10000) if stage == "smoke" else job["photons"]
            if actual.shape != (1, 3) or actual[0, 2] != 0 or actual.sum() != photons:
                raise ValueError(f"Worker primary count failed: {folder}")
            for name, expected in record["output_sha256"].items():
                if digest(folder / name) != expected:
                    raise ValueError(f"Worker output hash failed: {folder/name}")
            for energy in (218, 440):
                counts = np.loadtxt(folder / f"CntStat_{energy}.csv", delimiter=",",
                                    dtype=np.int64, ndmin=2)
                if counts.shape != (1, 10496) or np.any(counts < 0):
                    raise ValueError(f"Worker detector counts invalid: {folder}")
            primaries += actual[0]
            records.append(record)
        total = int(primaries.sum())
        fraction = float(primaries[0] / total)
        expected = manifest["expected_primary_energy_fraction"]["218"]
        standard_error = math.sqrt(expected * (1-expected) / total)
        if abs(fraction-expected) > 6*standard_error:
            raise ValueError("Observed 218/440 mixture differs from yield-weighted truth")
        if stage == "complete" and (len(records) != worker_count or total != expected_total):
            raise ValueError("Production workers did not close")
        result.update({"verified_workers": len(records),
                       "observed_primary_counts_218_440_other": primaries.tolist(),
                       "observed_218_fraction": fraction,
                       "expected_218_fraction": expected})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--stage", choices=("prepared", "smoke", "first", "complete"),
                        default="prepared")
    args = parser.parse_args()
    print(json.dumps(validate(args.manifest, args.stage), indent=2))


if __name__ == "__main__":
    main()
