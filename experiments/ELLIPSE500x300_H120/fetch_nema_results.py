"""Fetch immutable NEMA outputs, checking every downloaded SHA-256."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "FOV120"))
from reconstruction_ssh import connect

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()

def main():
    result = sys.argv[1]
    if "/" in result or "\\" in result:
        raise ValueError("Result must be one directory name")
    report = json.loads((HERE / "reports" / f"{result}_integrity.json").read_text())
    if report["dataset"] != "NEMA_Body_H60" or report["job_result"] != result:
        raise ValueError("Verified NEMA result required")
    remote = "/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/Results/" + result
    dest = HERE / "generated/RemoteResults" / result
    dest.mkdir(parents=True, exist_ok=True)
    expected = {f"Image_{row['channel']}_{kind}.float32": row["sha256"][kind]
                for row in report["outputs"] for kind in ("full", "history")}
    expected["PredictedCntStat_218_From440.float32"] = report["predicted_sha256"]
    with connect() as ssh, ssh.open_sftp() as sftp:
        for name, sha in expected.items():
            path = dest / name
            if not path.exists() or digest(path) != sha:
                temporary = path.with_suffix(".downloading")
                sftp.get(remote + "/" + name, str(temporary))
                if digest(temporary) != sha:
                    raise ValueError(f"Transfer checksum mismatch: {name}")
                temporary.replace(path)
            print("VERIFIED", name, flush=True)
        sftp.get(remote + "/run_manifest.json", str(dest / "run_manifest.json"))
        run = json.loads((dest / "run_manifest.json").read_text())
        name = f"{run['dataset']}_{run['count_level']}.json"
        collection = HERE / "generated/collections" / name
        collection.parent.mkdir(parents=True, exist_ok=True)
        temporary = collection.with_suffix(".downloading")
        sftp.get(remote.rsplit('/Results/',1)[0]+"/collections/"+name, str(temporary))
        if digest(temporary) != report["collection_sha256"]:
            raise ValueError("Transferred collection hash mismatch")
        if collection.exists() and digest(collection) != report["collection_sha256"]:
            raise ValueError("Refusing to replace a different frozen collection")
        temporary.replace(collection)
    (dest / "transfer_manifest.json").write_text(json.dumps(expected, indent=2)+"\n")

if __name__ == "__main__":
    main()
