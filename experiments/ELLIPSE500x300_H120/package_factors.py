"""Package validated calibrated Factors and Sensi_d for scxi717 transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    root = a.generated.resolve()
    factor = root / "FactorsCalibrated"
    sensitivity = root / "Sensitivity"
    archive = root / "factors_transfer.tar.zst"
    manifest = root / "factors_transfer_files.json"
    sha_file = root / "factors_transfer.tar.zst.sha256"
    if any(x.exists() for x in (archive, manifest, sha_file)):
        raise FileExistsError("Factor transfer package already exists")
    names = ("218keV_RotateNum20", "440keV_RotateNum20",
             "440keV_to218win_RotateNum20")
    for name in names:
        folder = factor / name
        record = json.loads((folder / "factor_manifest.json").read_text())
        if not record.get("calibration", {}).get("enabled"):
            raise ValueError(f"Uncalibrated Factor: {name}")
        if (folder / "SysMat_polar").stat().st_size != 5_543_567_360:
            raise ValueError(f"Wrong Factor byte count: {name}")
    if (factor / names[1] / "Sensi_d").stat().st_size != 528_160:
        raise ValueError("Missing or wrong-size Sensi_d")
    if not (factor / names[1] / "Sensi_d_provenance.json").is_file():
        raise ValueError("Missing sensitivity provenance")
    if not (sensitivity / "ellipse_effective_sensitivity.json").is_file():
        raise ValueError("Sensitivity run is not complete")
    if not (sensitivity / "IndependentCircleClosure").is_dir():
        raise ValueError("Independent Compton closure output is missing")
    files = sorted(x for base in (factor, sensitivity) for x in base.rglob("*") if x.is_file())
    if any(x.is_symlink() for x in files):
        raise ValueError("Factor package must not contain symlinks")
    records = {x.relative_to(root).as_posix(): {"bytes": x.stat().st_size,
              "sha256": digest(x)} for x in files}
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    temporary = archive.with_name(archive.name + ".building")
    try:
        producer = subprocess.Popen(["tar", "-C", str(root), "-cf", "-",
                                     "FactorsCalibrated", "Sensitivity", manifest.name],
                                    stdout=subprocess.PIPE)
        consumer = subprocess.run(["zstd", f"-T{a.threads}", "-3", "-o", str(temporary)],
                                  stdin=producer.stdout, check=False)
        producer.stdout.close()
        if producer.wait() or consumer.returncode:
            raise RuntimeError("Factor tar/zstd packaging failed")
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    sha = digest(archive)
    sha_file.write_text(sha + "\n")
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                      "sha256": sha, "files": len(records)}))


if __name__ == "__main__":
    main()
