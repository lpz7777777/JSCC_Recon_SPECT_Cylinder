"""Package selected collected imaging datasets with per-file SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
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
    p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    p.add_argument("--level", choices=("1e9","5e9","1e10"), default="1e9")
    p.add_argument("--archive-stem")
    a = p.parse_args()
    a.archive_stem = a.archive_stem or f"imaging_{a.level}"
    if (not re.fullmatch(r"[A-Za-z0-9_-]+", a.archive_stem) or
            len(a.datasets) != len(set(a.datasets)) or
            not all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in a.datasets)):
        raise ValueError("Unsafe or repeated dataset/archive names")
    root = a.generated.resolve()
    archive = root / f"{a.archive_stem}.tar.gz"
    manifest = root / f"{a.archive_stem}_files.json"
    sha_file = root / f"{a.archive_stem}.tar.gz.sha256"
    if any(path.exists() for path in (archive, manifest, sha_file)):
        raise FileExistsError("Imaging transfer package already exists")
    files = []
    all_seeds = set()
    total = {"1e9":1_000_000_000,"5e9":5_000_000_000,"1e10":10_000_000_000}[a.level]
    for dataset in a.datasets:
        collection = root / "collections" / f"{dataset}_{a.level}.json"
        record = json.loads(collection.read_text())
        if (record["dataset"] != dataset or record["level"] != a.level or
                record["views"] != list(range(1, 21)) or
                len(record["worker_indices"]) != 200 or len(record["seeds"]) != 200 or
                len(set(record["worker_indices"])) != 200 or
                sum(record["primary_counts"]) != total):
            raise ValueError(f"Invalid collection: {dataset}")
        seeds = set(record["seeds"])
        if len(seeds) != 200 or all_seeds & seeds:
            raise ValueError(f"Repeated random seed: {dataset}")
        all_seeds.update(seeds)
        files.append(collection)
        files.extend(root / "CntStat" / f"{energy}keV_RotateNum20_Geant4JSCC" /
                     f"CntStat_{dataset}_{a.level}.csv" for energy in (218, 440))
        list_root = root / "List/218-440keV_RotateNum20_Geant4JSCC" / f"List_{dataset}_{a.level}"
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
