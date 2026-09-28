"""Compact status snapshot for immutable Geant4 worker manifests."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("manifest",type=Path)
    a=p.parse_args()
    plan=json.loads(a.manifest.read_text())["jobs"]
    base=a.manifest.parent/"workers"
    stats=Counter()
    failures=[]
    for job in plan:
        record=base/f"{job['index']:05d}"/"worker.json"
        if not record.is_file():
            stats[job["dataset"],job["level"],"pending"]+=1
            continue
        actual=json.loads(record.read_text())
        status=actual.get("status","unknown")
        if status=="complete" and actual.get("simulated_photons")!=job["photons"]:
            status="count_mismatch"
        stats[job["dataset"],job["level"],status]+=1
        if status!="complete": failures.append(job["index"])
    result={"manifest":str(a.manifest),
            "counts":[{"dataset":dataset,"level":level,"state":state,"workers":count}
                      for (dataset,level,state),count in sorted(stats.items())],
            "failed_indices":failures[:50]}
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
