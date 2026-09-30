"""Fetch and provenance-check the interpolated point-factor comparison."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "FOV120"))
from reconstruction_ssh import connect

REMOTE = ("/data/run01/scxi717/lpz/"
          "20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/"
          "experiments/ELLIPSE500x300_H120/generated/PointQA/point_factor_interpolation.json")


def main():
    with connect() as ssh, ssh.open_sftp() as sftp:
        with sftp.open(REMOTE, "rb") as stream:
            payload = stream.read()
    report = json.loads(payload)
    bundle = HERE / "generated/PointQA/selected_point_counts.zip"
    if (report["experiment"] != "ELLIPSE500x300_H120" or
        len(report["rows"]) != 10 or
        report["bundle_sha256"] != hashlib.sha256(bundle.read_bytes()).hexdigest()):
        raise ValueError("Remote point analysis provenance mismatch")
    output = HERE / "reports/selected_point_factor_interpolation.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
