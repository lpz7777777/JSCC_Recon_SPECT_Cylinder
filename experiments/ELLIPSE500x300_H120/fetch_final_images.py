"""Download six small final frames and verify remote output hashes."""
import argparse
import hashlib
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
    args=p.parse_args()
    if "/" in args.result_name or "\\" in args.result_name:
        p.error("result_name must be a single directory name")
    report=json.loads((HERE/"reports"/f"{args.result_name}_integrity.json").read_text())
    if report["job_result"]!=args.result_name or len(report["outputs"])!=6:
        raise ValueError("Verified integrity report required")
    dest=HERE/"generated/RemoteResults"/args.result_name
    dest.mkdir(parents=True,exist_ok=True)
    with connect() as ssh, ssh.open_sftp() as sftp:
        for channel in report["outputs"]:
            name=f"Image_{channel['channel']}_full.float32"
            expected=channel["sha256"]["full"]
            local=dest/name
            if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest()==expected:
                continue
            temporary=local.with_suffix(".downloading")
            sha=hashlib.sha256()
            with sftp.open(f"{REMOTE}/{args.result_name}/{name}","rb") as src, temporary.open("wb") as sink:
                while block:=src.read(1<<20):
                    sha.update(block)
                    sink.write(block)
            if sha.hexdigest()!=expected or temporary.stat().st_size!=132040*4:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Downloaded image checksum mismatch: {name}")
            temporary.replace(local)
            print(local)

if __name__=="__main__": main()
