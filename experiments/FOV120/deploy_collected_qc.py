"""Deploy a verified FOV120 QC archive to the reconstruction cluster.

The reconstruction credential stays in Windows DPAPI storage. The archive is
uploaded to a temporary name, checked remotely, and extracted only when its
members do not already exist. Every extracted file is then checked against the
file-level SHA256 manifest produced on the Geant4 cluster.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import tarfile

from reconstruction_ssh import connect


REMOTE_ROOT = ("/data/run01/scxi717/lpz/"
               "20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/"
               "experiments/FOV120_20260924/experiments/FOV120/generated")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(client, command):
    _, stdout, stderr = client.exec_command(command, timeout=120)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f"Remote command failed ({status}): {err or out}")
    return out.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("1e9", "1e10"), required=True)
    parser.add_argument("--generated", type=Path,
                        default=Path(__file__).with_name("generated"))
    args = parser.parse_args()
    archive = args.generated / f"qc_{args.level}_collected.tar.gz"
    expected = (args.generated / f"qc_{args.level}_collected.sha256").read_text().strip()
    actual = sha256(archive)
    if actual != expected:
        raise ValueError("Local archive checksum mismatch")
    manifest_name = f"qc_{args.level}_files.json"
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or manifest_name not in names:
            raise ValueError("Archive has duplicate or missing manifest members")
        for member in members:
            path = PurePosixPath(member.name)
            if (not member.isfile() or path.is_absolute() or
                    ".." in path.parts or not path.parts):
                raise ValueError(f"Unsafe archive member: {member.name}")
        manifest = json.load(bundle.extractfile(manifest_name))
        if set(names) != set(manifest) | {manifest_name}:
            raise ValueError("Archive contents differ from file manifest")
        for member in members:
            if member.name in manifest and member.size != manifest[member.name]["bytes"]:
                raise ValueError(f"Archive size mismatch: {member.name}")

    with connect() as client:
        sftp = client.open_sftp()
        try:
            remote_archive = f"{REMOTE_ROOT}/{archive.name}"
            remote_part = remote_archive + ".part"
            for name in [archive.name, archive.name + ".part", manifest_name, *manifest]:
                try:
                    sftp.stat(f"{REMOTE_ROOT}/{name}")
                except FileNotFoundError:
                    continue
                raise FileExistsError(f"Refusing to replace remote file: {name}")
            print(f"Uploading {archive.stat().st_size} bytes; {len(manifest)} data files")
            last_report = [0]

            def progress(sent, total):
                point = sent // (100 << 20)
                if point > last_report[0]:
                    last_report[0] = point
                    print(f"Uploaded {sent}/{total} bytes", flush=True)

            sftp.put(str(archive), remote_part, callback=progress, confirm=True)
            remote_hash = run(client, f"sha256sum {shlex.quote(remote_part)}").split()[0]
            if remote_hash != expected:
                raise ValueError("Remote archive checksum mismatch; leaving .part for inspection")
            sftp.rename(remote_part, remote_archive)
            run(client, f"tar --no-same-owner -xzf {shlex.quote(remote_archive)} "
                        f"-C {shlex.quote(REMOTE_ROOT)}")
            checker = '''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest=json.loads((root/sys.argv[2]).read_text())
for relative,record in manifest.items():
    path=root/relative
    assert path.stat().st_size==record["bytes"], relative
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(8<<20),b""):
            digest.update(block)
    assert digest.hexdigest()==record["sha256"], relative
print(f"Verified {len(manifest)} files and {sum(v['bytes'] for v in manifest.values())} bytes")'''
            print(run(client, "python3 -c " + shlex.quote(checker) + " " +
                      shlex.quote(REMOTE_ROOT) + " " + shlex.quote(manifest_name)))
            print(f"Remote archive SHA256: {remote_hash}")
        finally:
            sftp.close()


if __name__ == "__main__":
    main()
