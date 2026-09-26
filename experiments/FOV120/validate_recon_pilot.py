"""Gate a full FOV120 run on a complete, memory-safe short reconstruction."""
import argparse
import json
from pathlib import Path

import numpy as np


CHANNELS = (
    "Image_440_SinglePhoton", "Image_440_ComptonOnly",
    "Image_440_SinglePlusCompton", "Image_218_SinglePhoton_CrossTalkCorrected",
    "Image_440SinglePlus218Single", "Image_440SingleComptonPlus218Single",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", choices=("Uniform", "Contrast"), required=True)
    parser.add_argument("--level", choices=("1e9", "1e10"), required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--max-gpu-fraction", type=float, default=0.8)
    args = parser.parse_args()
    root = args.result
    manifest = json.loads((root / "run_manifest.json").read_text())
    for key, expected in (("dataset", args.dataset), ("count_level", args.level),
                          ("world_size", args.world_size), ("pixel_count", 51240),
                          ("z_layers", 40), ("iterations", 10)):
        if manifest.get(key) != expected:
            raise ValueError(f"Pilot {key}: expected {expected}, got {manifest.get(key)}")
    if manifest.get("accepted_compton_events", 0) <= 0:
        raise ValueError("Pilot accepted no Compton events")
    resources = manifest.get("resources_by_rank", [])
    if len(resources) != args.world_size:
        raise ValueError("Incomplete pilot GPU resource records")
    peak = 0.0
    for resource in resources:
        total = resource["device_total_bytes"]
        fraction = max(resource["peak_allocated_bytes"],
                       resource["peak_reserved_bytes"]) / total
        peak = max(peak, fraction)
        if fraction > args.max_gpu_fraction:
            raise ValueError(f"Pilot GPU peak {fraction:.1%} exceeds "
                             f"{args.max_gpu_fraction:.1%}")
    for channel in CHANNELS:
        path = root / channel
        if path.stat().st_size != 51240 * 4:
            raise ValueError(f"Missing or wrong-size pilot image: {channel}")
        image = np.fromfile(path, dtype=np.float32)
        if not np.isfinite(image).all() or np.any(image < 0):
            raise ValueError(f"Invalid pilot image: {channel}")
    report = {"dataset": args.dataset, "level": args.level,
              "accepted_compton_events": manifest["accepted_compton_events"],
              "max_gpu_fraction": peak, "channels": len(CHANNELS), "status": "ok"}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
