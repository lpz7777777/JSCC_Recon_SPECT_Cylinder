"""Package four collected 1e9 imaging datasets with per-file SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile

DATASETS = ("CircleNewDist", "EllipseUniform", "EllipseContrast", "XCAT")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generated", type=Path, required=True)
    a = p.parse_args()
    root = a.generated.resolve()
    archive = root / "imaging_1e9.tar.gz"
    manifest = root / "imaging_1e9_files.json"
    sha_file = root / "imaging_1e9.tar.gz.sha256"
    if any(path.exists() for path in (archive, manifest, sha_file)):
        raise FileExistsError("Imaging transfer package already exists")
    files = []
    all_seeds = set()
    for dataset in DATASETS:
        collection = root / "collections" / f"{dataset}_1e9.json"
        record = json.loads(collection.read_text())
        if (record["dataset"] != dataset or record["level"] != "1e9" or
                record["views"] != list(range(1, 21)) or
                len(record["worker_indices"]) != 200 or len(record["seeds"]) != 200 or
                len(set(record["worker_indices"])) != 200 or
                sum(record["primary_counts"]) != 1_000_000_000):
            raise ValueError(f"Invalid collection: {dataset}")
        seeds = set(record["seeds"])
        if len(seeds) != 200 or all_seeds & seeds:
            raise ValueError(f"Repeated random seed: {dataset}")
        all_seeds.update(seeds)
        files.append(collection)
        files.extend(root / "CntStat" / f"{energy}keV_RotateNum20_Geant4JSCC" /
                     f"CntStat_{dataset}_1e9.csv" for energy in (218, 440))
        list_root = root / "List/218-440keV_RotateNum20_Geant4JSCC" / f"List_{dataset}_1e9"
        files.extend(list_root / f"{view}.csv" for view in range(1, 21))
    records = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        records[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size,
                                                        "sha256": digest(path)}
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    temporary = archive.with_name(archive.name + ".building")
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=1) as package:
            for path in files:
                package.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
            package.add(manifest, arcname=manifest.name, recursive=False)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    sha_file.write_text(digest(archive) + "\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": digest(archive), "files": len(records),
                      "unique_seeds": len(all_seeds)}))


if __name__ == "__main__":
    main()
