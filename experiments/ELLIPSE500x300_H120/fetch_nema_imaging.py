"""Fetch the immutable maty imaging package through the existing SSH agent."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from deploy_nema_simulation import HOST, REMOTE, run, remote, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level",choices=("1e9","5e9","1e10"),required=True)
    args=parser.parse_args()
    base=Path(__file__).resolve().parent/"generated"
    stem=f"nema_h60_imaging_{args.level}"
    names=[stem+".tar.gz.sha256",stem+"_files.json",stem+".tar.gz"]
    hashes={name:remote(f"sha256sum {REMOTE}/generated/{name}").split()[0] for name in names}
    for name in names:
        dest=base/name
        if dest.exists():
            if digest(dest)!=hashes[name]:
                raise ValueError(f"Refusing to replace a different frozen package: {name}")
            continue
        temp=dest.with_name(dest.name+".downloading")
        run(["scp","-o","BatchMode=yes",HOST+":"+REMOTE+"/generated/"+name,str(temp)])
        if digest(temp)!=hashes[name]:
            raise ValueError(f"Transfer checksum mismatch: {name}")
        temp.replace(dest)
    expected=(base/names[0]).read_text().strip()
    if digest(base/names[2])!=expected:
        raise ValueError("Imaging archive checksum mismatch")
    manifest=json.loads((base/names[1]).read_text())
    with tarfile.open(base/names[2],"r:gz") as archive:
        members=archive.getmembers()
        if len({m.name for m in members})!=len(members) or set(m.name for m in members)!=set(manifest)|{names[1]}:
            raise ValueError("Archive members differ from the file manifest")
        for member in members:
            if not member.isfile():
                raise ValueError("Unexpected non-file archive member")
            if member.name in manifest:
                sha=hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda:stream.read(8<<20),b""):
                        sha.update(block)
                if member.size!=manifest[member.name]["bytes"] or sha.hexdigest()!=manifest[member.name]["sha256"]:
                    raise ValueError(f"Archive member checksum mismatch: {member.name}")
        with archive.extractfile(f"collections/NEMA_Body_H60_{args.level}.json") as stream:
            collection=json.load(stream)
    expected_total={"1e9":10**9,"5e9":5*10**9,"1e10":10**10}[args.level]
    if (collection["level"]!=args.level or sum(collection["primary_counts"])!=expected_total or
        collection["views"]!=list(range(1,21)) or sorted(collection["worker_indices"])!=list(range(200)) or
        len(collection["seeds"])!=200 or len(set(collection["seeds"]))!=200):
        raise ValueError("Transport collection primary/view/seed closure failed")
    print(json.dumps({"level":args.level,"archive_sha256":expected,"files":len(manifest),
                      "primary_counts":collection["primary_counts"],"views":20,"workers":200},indent=2))


if __name__=="__main__": main()
