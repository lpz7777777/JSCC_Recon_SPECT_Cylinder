"""Deploy calibrated ellipse Factors and Sensi_d to scxi717 with SHA-256 checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import sys

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


def run(client, command: str, timeout: int = 3600) -> str:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f"Remote command failed ({status}): {err or out}")
    return out.strip()


def main() -> None:
    local = HERE / "generated"
    archive = local / "factors_transfer.tar.zst"
    expected = (local / "factors_transfer.tar.zst.sha256").read_text().strip()
    if digest(archive) != expected:
        raise ValueError("Local Factor archive hash mismatch")
    manifest = json.loads((local / "factors_transfer_files.json").read_text())
    for relative in manifest:
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or path.parts[0] not in (
                "FactorsCalibrated", "Sensitivity"):
            raise ValueError(f"Unsafe Factor path: {relative}")
    with connect() as client:
        run(client, f"mkdir -p -- {shlex.quote(REMOTE)}")
        with client.open_sftp() as sftp:
            remote_archive = f"{REMOTE}/{archive.name}"
            partial = remote_archive + ".uploading"
            for name in (archive.name, archive.name + ".uploading",
                         "FactorsCalibrated", "Sensitivity", "factors_transfer_files.json"):
                try:
                    sftp.stat(f"{REMOTE}/{name}")
                except FileNotFoundError:
                    continue
                raise FileExistsError(f"Remote Factor target already exists: {name}")
            last = [0]
            def progress(sent: int, total: int) -> None:
                point = sent // (100 << 20)
                if point > last[0]:
                    last[0] = point
                    print(f"Uploaded {sent}/{total} bytes", flush=True)
            print(f"Uploading {archive.stat().st_size} Factor bytes", flush=True)
            sftp.put(str(archive), partial, callback=progress, confirm=True)
            if run(client, f"sha256sum -- {shlex.quote(partial)}").split()[0] != expected:
                raise ValueError("Remote Factor archive hash mismatch")
            sftp.rename(partial, remote_archive)
        pipeline = f"zstd -dc -- {shlex.quote(remote_archive)} | tar -tf -"
        names = run(client, "bash -o pipefail -c " + shlex.quote(pipeline)).splitlines()
        allowed = set(manifest) | {"factors_transfer_files.json"}
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or (
                    name.rstrip("/") not in allowed and
                    name.rstrip("/") not in ("FactorsCalibrated", "Sensitivity",
                                                 "FactorsCalibrated/218keV_RotateNum20",
                                                 "FactorsCalibrated/440keV_RotateNum20",
                                                 "FactorsCalibrated/440keV_to218win_RotateNum20",
                                                 "Sensitivity/IndependentCircleClosure")):
                raise ValueError(f"Unexpected Factor archive member: {name}")
        extract = (f"zstd -dc -- {shlex.quote(remote_archive)} | "
                   f"tar --no-same-owner -xf - -C {shlex.quote(REMOTE)}")
        run(client, "bash -o pipefail -c " + shlex.quote(extract))
        checker = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); records=json.loads((root/'factors_transfer_files.json').read_text())
for relative,record in records.items():
    path=root/relative
    assert path.stat().st_size==record['bytes'],relative
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    assert h.hexdigest()==record['sha256'],relative
print(f'ELLIPSE_FACTOR_TRANSFER_VERIFIED {len(records)} files')'''
        print(run(client, "python3 -c " + shlex.quote(checker) + " " +
                  shlex.quote(REMOTE)))


if __name__ == "__main__":
    main()
