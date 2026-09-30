"""Upload the small ellipse reconstruction code and geometry to scxi717.

Uses the existing Windows DPAPI-protected credential through the project's
reconstruction_ssh helper. Never transfers raw Factors, images, or secrets.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"experiments/FOV120"))
from reconstruction_ssh import connect

REMOTE_ROOT="/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor"
FILES=("config.json","geometry.py","torch_active_operator.py",
       "validate_factors.py","run_reconstruction.py","run_sensitivity.py",
       "resource_budget.py","reconstruct.sh","test_active_dist.py",
       "diagnose_geometry_payload.py","diagnose_nccl.py","diagnose_nccl.sh",
       "check_remote_pilot.py","verify_formal_result.py","analyze_circle_uniform.py",
       "README.md")


def digest(path):
    sha=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--include-geometry",action="store_true")
    a=p.parse_args()
    target=f"{REMOTE_ROOT}/experiments/ELLIPSE500x300_H120"
    with connect() as client:
        cmd=f"mkdir -p -- {shlex.quote(target)}/generated/Geometry {shlex.quote(target)}/logs"
        _,out,err=client.exec_command(cmd)
        if out.channel.recv_exit_status():
            raise RuntimeError(err.read().decode(errors="replace"))
        with client.open_sftp() as sftp:
            paths=[(HERE/name,f"{target}/{name}") for name in FILES]
            if a.include_geometry:
                paths.append((HERE/"generated/Geometry/geometry.npz",
                              f"{target}/generated/Geometry/geometry.npz"))
                paths.append((HERE/"generated/Geometry/manifest.json",
                              f"{target}/generated/Geometry/manifest.json"))
            for source,destination in paths:
                if not source.is_file(): raise FileNotFoundError(source)
                temporary=destination+".uploading"
                sftp.put(str(source),temporary)
                remote_hash_cmd=f"sha256sum -- {shlex.quote(temporary)}"
                _,out,err=client.exec_command(remote_hash_cmd)
                returned=out.read().decode().split()[0]
                if out.channel.recv_exit_status() or returned!=digest(source):
                    raise RuntimeError(f"Transfer checksum mismatch: {source}")
                _,moved,move_error=client.exec_command(
                    f"mv -f -- {shlex.quote(temporary)} {shlex.quote(destination)}")
                if moved.channel.recv_exit_status():
                    raise RuntimeError(move_error.read().decode(errors="replace"))
                print(f"Uploaded {source.name}: {returned}")


if __name__=="__main__":
    main()
