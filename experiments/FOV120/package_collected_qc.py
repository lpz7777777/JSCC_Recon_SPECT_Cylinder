"""Package collected FOV120 Uniform/Contrast data with file-level SHA256 checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("1e9", "1e10"), required=True)
    parser.add_argument("--generated", type=Path, default=Path(__file__).with_name("generated"))
    args = parser.parse_args()
    root = args.generated.resolve()
    archive = root / f"qc_{args.level}_collected.tar.gz"
    file_manifest = root / f"qc_{args.level}_files.json"
    sha_file = root / f"qc_{args.level}_collected.sha256"
    if any(path.exists() for path in (archive, file_manifest, sha_file)):
        raise FileExistsError("Refusing to replace an existing QC archive or manifest")

    files = []
    all_seeds = set()
    for dataset in ("Uniform", "Contrast"):
        record = root / "collections" / f"{dataset}_{args.level}.json"
        info = json.loads(record.read_text(encoding="utf-8"))
        assert info["dataset"] == dataset and info["level"] == args.level
        assert info["views"] == list(range(1, 21))
        assert len(info["worker_indices"]) == len(info["seeds"]) == 200
        assert sum(info["primary_counts"]) == int(float(args.level))
        if all_seeds.intersection(info["seeds"]):
            raise ValueError("Duplicate random seed across datasets")
        all_seeds.update(info["seeds"])
        files.append(record)
        files.extend(root / "CntStat" / f"{energy}keV_RotateNum20_Geant4JSCC" /
                     f"CntStat_{dataset}_{args.level}.csv" for energy in (218, 440))
        list_dir = (root / "List" / "218-440keV_RotateNum20_Geant4JSCC" /
                    f"List_{dataset}_{args.level}")
        files.extend(list_dir / f"{view}.csv" for view in range(1, 21))

    records = {}
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        records[path.relative_to(root).as_posix()] = {
            "bytes": path.stat().st_size, "sha256": digest(path),
        }
    file_manifest.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary = archive.with_name(archive.name + ".part")
    try:
        with tarfile.open(temporary, mode="w:gz", compresslevel=1) as package:
            for path in files:
                package.add(path, arcname=path.relative_to(root).as_posix())
            package.add(file_manifest, arcname=file_manifest.name)
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    sha = digest(archive)
    sha_file.write_text(sha + "\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": sha, "files": len(records), "seeds": len(all_seeds)}))


if __name__ == "__main__":
    main()
