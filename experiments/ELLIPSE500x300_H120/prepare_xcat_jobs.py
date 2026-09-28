"""Freeze independently seeded Geant4 workers for an existing XCAT source."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    target = source / "jobs.json"
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((source / "manifest.json").read_text())
    workers = manifest["workers_per_view"]
    total = manifest["total_primary_photons"]
    if total % (20 * workers):
        raise ValueError("Total does not divide over views/workers")
    jobs = []
    for item in manifest["macros"]:
        macro = source / item["file"]
        if digest(macro) != item["sha256"]:
            raise ValueError(f"Changed source macro: {macro}")
        for worker in range(workers):
            index = len(jobs)
            jobs.append({"index": index, "dataset": "XCAT", "level": f"1e{len(str(total))-1}",
                         "view": item["view"], "worker": worker,
                         "photons": total // (20 * workers), "seed": 29092801 + index,
                         "mono_keV": None, "role": "imaging", "macro": macro.name,
                         "macro_sha256": item["sha256"]})
    target.write_text(json.dumps({"format_version": 1, "experiment_id": manifest["experiment"],
                                  "source_manifest_sha256": digest(source / "manifest.json"),
                                  "jobs": jobs}, indent=2) + "\n")
    print(f"Frozen {len(jobs)} workers: {target}")


if __name__ == "__main__":
    main()
