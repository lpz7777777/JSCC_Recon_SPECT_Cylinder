"""Deploy a hash-verified ellipse 1e9 projection package to scxi717."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments/FOV120"))
from reconstruction_ssh import connect

REMOTE = ("/data/run01/scxi717/lpz/"
          "20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/"
          "experiments/ELLIPSE500x300_H120/generated")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(client, command: str) -> str:
    _, stdout, stderr = client.exec_command(command, timeout=300)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if code:
        raise RuntimeError(f"Remote command failed ({code}): {err or out}")
    return out.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-stem", default="imaging_1e9")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.archive_stem):
        raise ValueError("Unsafe archive stem")
    base = HERE / "generated"
    archive = base / f"{args.archive_stem}.tar.gz"
    expected = (base / f"{args.archive_stem}.tar.gz.sha256").read_text().strip()
    if digest(archive) != expected:
        raise ValueError("Local imaging archive SHA-256 mismatch")
    manifest_name = f"{args.archive_stem}_files.json"
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or manifest_name not in names:
            raise ValueError("Duplicate or missing imaging archive member")
        for member in members:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe imaging archive member: {member.name}")
        manifest = json.load(package.extractfile(manifest_name))
        if set(names) != set(manifest) | {manifest_name}:
            raise ValueError("Archive members differ from manifest")
        for member in members:
            if member.name in manifest and member.size != manifest[member.name]["bytes"]:
                raise ValueError(f"Wrong size: {member.name}")
    with connect() as client:
        run(client, f"mkdir -p -- {shlex.quote(REMOTE)}")
        with client.open_sftp() as sftp:
            remote_archive = f"{REMOTE}/{archive.name}"
            partial = remote_archive + ".uploading"
            for path in (remote_archive, partial, f"{REMOTE}/{manifest_name}"):
                try:
                    sftp.stat(path)
                except FileNotFoundError:
                    continue
                raise FileExistsError(f"Refusing to replace remote file: {path}")
            print(f"Uploading {archive.stat().st_size} bytes", flush=True)
            sftp.put(str(archive), partial, confirm=True)
            if run(client, f"sha256sum -- {shlex.quote(partial)}").split()[0] != expected:
                raise ValueError("Remote imaging archive hash mismatch")
            sftp.rename(partial, remote_archive)
        # Every file is new within the independent experiment root.
        for relative in manifest:
            path = f"{REMOTE}/{relative}"
            check = run(client, f"test ! -e {shlex.quote(path)} && echo NEW")
            if check != "NEW":
                raise FileExistsError(path)
        run(client, f"tar --no-same-owner -xzf {shlex.quote(remote_archive)} "
                    f"-C {shlex.quote(REMOTE)}")
        checker = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=json.loads((root/sys.argv[2]).read_text())
for relative,record in manifest.items():
    path=root/relative
    assert path.stat().st_size==record["bytes"],relative
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            h.update(block)
    assert h.hexdigest()==record["sha256"],relative
print(f"ELLIPSE_IMAGING_TRANSFER_VERIFIED {len(manifest)} files")'''
        print(run(client, "python3 -c " + shlex.quote(checker) + " " +
                  shlex.quote(REMOTE) + " " + shlex.quote(manifest_name)))


if __name__ == "__main__":
    main()
