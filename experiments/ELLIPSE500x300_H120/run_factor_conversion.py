"""Wait for raw production, stream three uncalibrated Factors, then validate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = ROOT / "Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main"
STEM = "shift_0.000000_0.000000_0.000000"
SUFFIX = "_pe_v4_ELLIPSE500x300_H120"
EXPECTED = 13_317_120_000


def running(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-pid", type=int)
    p.add_argument("--wait-hours", type=float, default=12)
    a = p.parse_args()
    deadline = time.monotonic() + a.wait_hours * 3600
    if a.matrix_pid:
        while running(a.matrix_pid):
            if time.monotonic() > deadline:
                raise TimeoutError("Raw matrix production did not finish before deadline")
            time.sleep(60)
    folders = {"A218": ENGINE / "runs" / ("JSCC_218keV" + SUFFIX),
               "A440": ENGINE / "runs" / ("JSCC_440keV" + SUFFIX),
               "C440to218": ENGINE / "runs" / ("JSCC_440keV_to_218keVwin" + SUFFIX)}
    outputs = {"A218": "218keV_RotateNum20", "A440": "440keV_RotateNum20",
               "C440to218": "440keV_to218win_RotateNum20"}
    destination = HERE / "generated/FactorsRaw"
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError(destination)
        destination.rmdir()
    destination.mkdir(parents=True)
    for response, source in folders.items():
        if response == "C440to218":
            matrix = source / f"Scatter_SysMat_{STEM}.sysmat"
        else:
            matrix = source / f"SysMat_withScatter_{STEM}.sysmat"
            progress = json.loads((source / "PE_progress.json").read_text())
            if progress["status"] != "complete":
                raise ValueError(f"PE response incomplete: {response}")
        if matrix.stat().st_size != EXPECTED:
            raise ValueError(f"Wrong Cartesian response size: {matrix}")
        command = [sys.executable, str(HERE / "convert_factors.py"),
                   "--sysmat", str(matrix),
                   "--params-image", str(source / "Params_Image.dat"),
                   "--params-detector", str(source / "Params_Detector.dat"),
                   "--output", str(destination / outputs[response]),
                   "--response", response]
        subprocess.run(command, check=True)
    subprocess.run([sys.executable, str(HERE / "validate_factors.py"),
                    str(destination), "--full-scan"], check=True)
    (destination / "conversion_complete.json").write_text(json.dumps({
        "experiment": "ELLIPSE500x300_H120", "matrix_pid": a.matrix_pid,
        "factors": outputs, "full_scan": True}, indent=2) + "\n")
    print("ELLIPSE_FACTOR_CONVERSION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
