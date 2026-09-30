"""Fetch a small verified result report from scxi717 for Git-tracked evidence."""
import argparse
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"FOV120"))
from reconstruction_ssh import connect

REMOTE=("/data/run01/scxi717/lpz/"
        "20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/"
        "experiments/ELLIPSE500x300_H120/generated/Results")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("result_name")
    p.add_argument("--kind",choices=("integrity","circle_uniformity"),default="integrity")
    args=p.parse_args()
    if "/" in args.result_name or "\\" in args.result_name:
        p.error("result_name must be a directory name")
    with connect() as ssh, ssh.open_sftp() as sftp:
        with sftp.open(f"{REMOTE}/{args.result_name}/{args.kind}_report.json","r") as stream:
            report=json.load(stream)
    if args.kind=="integrity":
        if report["job_result"]!=args.result_name or len(report["outputs"])!=6:
            raise ValueError("Incomplete or mismatched remote integrity report")
    elif report["result"]!=args.result_name or len(report["channels"])!=4:
        raise ValueError("Incomplete or mismatched uniformity report")
    dest=HERE/"reports"/f"{args.result_name}_{args.kind}.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(report,indent=2)+"\n")
    print(dest)

if __name__=="__main__": main()
