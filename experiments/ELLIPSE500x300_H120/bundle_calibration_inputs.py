"""Hash and transfer only collected monoenergetic calibration/sensitivity data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


DATASETS=("calibration_218","calibration_440",
          "sensitivity_440","sensitivity_validation_440")


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def paths():
    result=[Path("collections")/f"{name}_all.json" for name in DATASETS]
    for name,window in (("calibration_218",218),("calibration_440",440),
                        ("calibration_440",218)):
        result.append(Path("CntStat")/f"{window}keV_RotateNum20_Geant4JSCC"/
                      f"CntStat_{name}_all.csv")
    for name in DATASETS[2:]:
        result.append(Path("List/218-440keV_RotateNum20_Geant4JSCC")/
                      f"List_{name}_all/1.csv")
    return result


def create(data,archive):
    if archive.exists():raise FileExistsError(archive)
    receipts=[]
    seed_sets=[]
    for dataset in DATASETS:
        record=json.loads((data/"collections"/f"{dataset}_all.json").read_text())
        if sum(record["primary_counts"])!=1_000_000_000:
            raise ValueError(f"Wrong primary count: {dataset}")
        if dataset.endswith("218"):
            if record["primary_counts"][1] or record["primary_counts"][2]:
                raise ValueError("218 source is not monoenergetic")
        elif record["primary_counts"][0] or record["primary_counts"][2]:
            raise ValueError("440 source is not monoenergetic")
        seed_sets.append(set(record["seeds"]))
    for i,left in enumerate(seed_sets):
        if any(left & right for right in seed_sets[i+1:]):
            raise ValueError("Calibration and validation reuse seeds")
    for relative in paths():
        source=data/relative
        if not source.is_file():raise FileNotFoundError(source)
        receipts.append({"path":relative.as_posix(),"bytes":source.stat().st_size,
                         "sha256":digest(source)})
    record={"experiment":"ELLIPSE500x300_H120","source":"maty collected Geant4",
            "data_level":"all","files":receipts}
    archive.parent.mkdir(parents=True,exist_ok=True)
    manifest=archive.with_suffix(archive.suffix+".json")
    manifest.write_text(json.dumps(record,indent=2)+"\n")
    with tarfile.open(archive,"w:gz") as out:
        for relative in paths():
            out.add(data/relative,arcname=relative.as_posix(),recursive=False)
    print(json.dumps({"archive":str(archive),"bytes":archive.stat().st_size,
                      "sha256":digest(archive),"files":len(receipts)}))


def extract(archive,manifest,destination):
    record=json.loads(manifest.read_text())
    expected={entry["path"]:entry for entry in record["files"]}
    if set(expected)!={p.as_posix() for p in paths()}:
        raise ValueError("Unexpected calibration bundle contents")
    with tarfile.open(archive,"r:gz") as source:
        actual={member.name:member for member in source.getmembers()}
        if set(actual)!=set(expected):
            raise ValueError("Archive members do not match manifest")
        for name,entry in expected.items():
            member=actual[name]
            if not member.isfile() or member.size!=entry["bytes"]:
                raise ValueError(f"Wrong archive entry: {name}")
            target=destination/name
            if target.exists():raise FileExistsError(target)
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+".extracting")
            with source.extractfile(member) as input_stream,temporary.open("wb") as output:
                for block in iter(lambda:input_stream.read(8<<20),b""):
                    output.write(block)
            if digest(temporary)!=entry["sha256"]:
                raise ValueError(f"Checksum mismatch after extraction: {name}")
            temporary.rename(target)
    print("ELLIPSE_CALIBRATION_BUNDLE_VERIFIED",len(expected))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("create")
    c.add_argument("--data",type=Path,required=True)
    c.add_argument("--archive",type=Path,required=True)
    e=sub.add_parser("extract")
    e.add_argument("--archive",type=Path,required=True)
    e.add_argument("--manifest",type=Path,required=True)
    e.add_argument("--destination",type=Path,required=True)
    a=p.parse_args()
    if a.command=="create":create(a.data,a.archive)
    else:extract(a.archive,a.manifest,a.destination)
