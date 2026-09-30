"""Fetch selected verified active history frames without copying 200 full frames."""
import argparse
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
CHANNELS = ("218_SinglePhoton_CrossTalkCorrected", "440_SinglePlusCompton")
ITERATIONS = (50, 500, 1000, 3000, 10000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result_name")
    args = p.parse_args()
    if "/" in args.result_name or "\\" in args.result_name:
        p.error("result_name must be one directory name")
    report = json.loads((HERE / "reports" / f"{args.result_name}_integrity.json").read_text())
    if report["job_result"] != args.result_name or report["history_frames_per_channel"] != 200:
        raise ValueError("Verified 200-frame formal result required")
    with np.load(HERE / "generated/Geometry/geometry.npz") as geometry:
        active = geometry["active_indices"]
    if len(active) != 82040:
        raise ValueError("Unexpected active pixel count")
    dest = HERE / "generated/HistorySelected" / args.result_name
    dest.mkdir(parents=True, exist_ok=True)
    record = {"result": args.result_name, "iterations": ITERATIONS,
              "channels": CHANNELS, "source_history_sha256": {}}
    with connect() as ssh, ssh.open_sftp() as sftp:
        for channel in CHANNELS:
            source = f"{REMOTE}/{args.result_name}/Image_{channel}_history.float32"
            expected = next(row["sha256"]["history"] for row in report["outputs"]
                            if row["channel"] == channel)
            record["source_history_sha256"][channel] = expected
            with sftp.open(source, "rb") as stream:
                if stream.stat().st_size != 200 * len(active) * 4:
                    raise ValueError(f"History size mismatch: {channel}")
                for iteration in ITERATIONS:
                    stream.seek((iteration // 50 - 1) * len(active) * 4)
                    data = stream.read(len(active) * 4)
                    if len(data) != len(active) * 4:
                        raise ValueError(f"Short history frame: {channel}/{iteration}")
                    values = np.frombuffer(data, dtype="<f4")
                    if not np.isfinite(values).all() or np.any(values < 0):
                        raise ValueError(f"Invalid history frame: {channel}/{iteration}")
                    full = np.zeros(132040, dtype="<f4")
                    full[active] = values
                    if iteration == 10000:
                        final = HERE / "generated/RemoteResults" / args.result_name / f"Image_{channel}_full.float32"
                        if final.exists() and not np.array_equal(full, np.fromfile(final, dtype="<f4")):
                            raise ValueError(f"Final history frame differs from verified image: {channel}")
                    full.tofile(dest / f"Image_{channel}_iter{iteration:05d}_full.float32")
                    print(channel, iteration, float(values.sum()))
    (dest / "source_manifest.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
