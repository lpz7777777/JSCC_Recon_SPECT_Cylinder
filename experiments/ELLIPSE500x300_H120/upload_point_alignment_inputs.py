"""Upload selected public experiment counts and analysis code to scxi717."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "FOV120"))
from reconstruction_ssh import connect

REMOTE = ("/data/run01/scxi717/lpz/"
          "20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/"
          "experiments/ELLIPSE500x300_H120")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reference = json.loads((HERE / "reports/selected_point_factor_alignment.json").read_text())
    source = HERE / "generated/PointSelectedCntStat"
    bundle = HERE / "generated/PointQA/selected_point_counts.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(HERE / "generated/PointScan/truth.json", "truth.json")
        for row in reference["rows"]:
            energy, dataset = row["energy_keV"], row["dataset"]
            relative = (f"{energy}keV_RotateNum20_Geant4JSCC/"
                        f"CntStat_{dataset}_1e+07.csv")
            file = source / relative
            if digest(file) != row["cntstat_sha256"]:
                raise ValueError(f"Independent CntStat changed: {dataset}")
            archive.write(file, "CntStat/" + relative)
    code = HERE / "point_factor_interpolation_remote.py"
    with connect() as ssh:
        _, stdout, stderr = ssh.exec_command(f"mkdir -p '{REMOTE}/generated/PointQA'")
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError(stderr.read().decode())
        with ssh.open_sftp() as sftp:
            for local, remote in ((code, f"{REMOTE}/{code.name}"),
                                  (bundle, f"{REMOTE}/generated/PointQA/{bundle.name}")):
                sftp.put(str(local), remote)
                if sftp.stat(remote).st_size != local.stat().st_size:
                    raise ValueError(f"Upload size mismatch: {remote}")
                print(local.name, local.stat().st_size, digest(local))


if __name__ == "__main__":
    main()
