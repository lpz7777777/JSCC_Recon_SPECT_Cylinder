"""Fetch six selected frames per channel from a verified 1e9 formal result."""
import argparse
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
          "experiments/ELLIPSE500x300_H120/generated/Results")
ITERATIONS = (50, 500, 1000, 3000, 5000, 10000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result_name")
    args = p.parse_args()
    if "/" in args.result_name or "\\" in args.result_name:
        p.error("result_name must be one directory name")
    report = json.loads((HERE / "reports" / f"{args.result_name}_integrity.json").read_text())
    if (report["job_result"] != args.result_name or
        report["history_frames_per_channel"] != 200 or len(report["outputs"]) != 6):
        raise ValueError("Complete verified formal result required")
    with np.load(HERE / "generated/Geometry/geometry.npz") as geometry:
        active = geometry["active_indices"]
    if len(active) != 82040:
        raise ValueError("Active geometry mismatch")
    output = HERE / "generated/GalleryFrames" / args.result_name
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"result": args.result_name, "iterations": ITERATIONS, "channels": {}}
    with connect() as ssh, ssh.open_sftp() as sftp:
        for channel in report["outputs"]:
            name = channel["channel"]
            file = f"{REMOTE}/{args.result_name}/Image_{name}_history.float32"
            hashes = {}
            with sftp.open(file, "rb") as stream:
                if stream.stat().st_size != 200 * len(active) * 4:
                    raise ValueError(f"History size mismatch: {name}")
                for iteration in ITERATIONS:
                    stream.seek((iteration // 50 - 1) * len(active) * 4)
                    data = stream.read(len(active) * 4)
                    if len(data) != len(active) * 4:
                        raise ValueError(f"Short history frame: {name}/{iteration}")
                    values = np.frombuffer(data, dtype="<f4")
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise ValueError(f"Invalid values: {name}/{iteration}")
                    full = np.zeros(132040, dtype="<f4")
                    full[active] = values
                    digest = hashlib.sha256(full.tobytes()).hexdigest()
                    if iteration == 10000 and digest != channel["sha256"]["full"]:
                        raise ValueError(f"Final frame/full-image hash mismatch: {name}")
                    path = output / f"Image_{name}_iter{iteration:05d}_full.float32"
                    full.tofile(path)
                    hashes[str(iteration)] = digest
            manifest["channels"][name] = {
                "source_history_sha256": channel["sha256"]["history"],
                "frames_sha256": hashes}
            print(name, "six frames checked", flush=True)
    path = output / "selected_frames_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
