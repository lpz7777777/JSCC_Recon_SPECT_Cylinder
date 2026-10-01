"""Install immutable NEMA macros on maty through the unlocked SSH agent."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

from validate_nema_simulation import validate

HERE = Path(__file__).resolve().parent
HOST = "maty@192.168.11.1"
REMOTE = "/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120"
SCRIPTS = ("validate_nema_simulation.py", "maty_nema_array.sh", "maty_nema_collect.sh", "package_imaging.py", "check_simulation.py")

def run(args):
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return result.stdout.strip()

def remote(command):
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", HOST, command])

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("1e9","5e9","1e10"), required=True)
    a = parser.parse_args()
    source = HERE / f"generated/NEMA_Body_H60/Simulation_{a.level}"
    validate(source / "jobs.json", "prepared")
    stage = HERE / "generated" / f"NEMA_upload_{a.level}"
    stage.mkdir(exist_ok=True)
    archive = stage / "prepared.tar.gz"
    files = [source / "jobs.json", *sorted((source / "macros").glob("*.mac"))]
    with tarfile.open(archive, "w:gz") as package:
        for path in files:
            package.add(path, arcname=path.relative_to(HERE).as_posix())
    target = f"{REMOTE}/generated/NEMA_Body_H60/Simulation_{a.level}"
    remote(f"test ! -e {shlex.quote(target)}")
    remote(f"mkdir -p {shlex.quote(REMOTE+'/generated')} ")
    remote_archive = REMOTE + f"/generated/nema_prepared_{a.level}.tar.gz"
    remote(f"test ! -e {shlex.quote(remote_archive)}")
    run(["scp", "-o", "BatchMode=yes", str(archive), HOST+":"+remote_archive])
    if remote(f"sha256sum {shlex.quote(remote_archive)}").split()[0] != digest(archive):
        raise ValueError("Prepared archive transfer hash mismatch")
    hashes = {}
    for name in SCRIPTS:
        path = stage / name
        path.write_bytes((HERE / name).read_bytes().replace(b"\r\n",b"\n"))
        temp = REMOTE + "/" + name + ".uploading"
        run(["scp", "-o", "BatchMode=yes", str(path), HOST+":"+temp])
        if remote(f"sha256sum {shlex.quote(temp)}").split()[0] != digest(path):
            raise ValueError(f"Script transfer hash mismatch: {name}")
        remote(f"mv -f {shlex.quote(temp)} {shlex.quote(REMOTE+'/'+name)}")
        hashes[name] = digest(path)
    remote(f"tar --no-same-owner -xzf {shlex.quote(remote_archive)} -C {shlex.quote(REMOTE)}")
    remote(f"mkdir -p {shlex.quote(target+'/logs')}")
    print(remote(f"python3 {shlex.quote(REMOTE+'/validate_nema_simulation.py')} {shlex.quote(target+'/jobs.json')}"))
    (stage / "transfer.json").write_text(json.dumps({"level":a.level,
        "jobs_sha256":digest(source/'jobs.json'),"archive_sha256":digest(archive),
        "remote":target,"script_sha256":hashes},indent=2)+"\n")

if __name__ == "__main__": main()
