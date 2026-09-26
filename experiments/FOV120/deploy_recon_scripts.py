"""Upload the versioned FOV120 reconstruction launcher and pilot gate."""
import hashlib
from pathlib import Path

from reconstruction_ssh import connect


ROOT = Path(__file__).parent
REMOTE = ("/data/run01/scxi717/lpz/"
          "20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/"
          "experiments/FOV120_20260924/experiments/FOV120")
FILES = ("reconstruct.sh", "validate_recon_pilot.py")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    with connect() as client:
        sftp = client.open_sftp()
        try:
            for name in FILES:
                data = (ROOT / name).read_bytes()
                if name.endswith(".sh") and b"\r\n" in data:
                    raise ValueError(f"Shell script has Windows line endings: {name}")
                target = f"{REMOTE}/{name}"
                temporary = target + ".new"
                try:
                    sftp.stat(temporary)
                except FileNotFoundError:
                    pass
                else:
                    raise FileExistsError(temporary)
                with sftp.file(temporary, "wb") as output:
                    output.write(data)
                with sftp.file(temporary, "rb") as source:
                    if digest(source.read()) != digest(data):
                        raise ValueError(f"Remote upload hash mismatch: {name}")
                sftp.posix_rename(temporary, target)
                print(f"{name}: {digest(data)}")
        finally:
            sftp.close()


if __name__ == "__main__":
    main()
