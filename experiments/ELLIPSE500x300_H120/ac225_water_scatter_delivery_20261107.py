"""Read the new formal jobs and strictly deliver each successful independent route."""

import argparse
import json
from pathlib import Path
import shutil
import tarfile

import numpy as np

from ac225_corrections.io import digest, read, verify, write
from ac225_gpu_archive_stream_20261105 import download_gpu_archive_stream
from ac225_water_scatter_20261107 import DATA, GPU, GPU_PYTHON, REPORT
from campaign_reliable_io_20261104 import accounting
from campaign_20261102_io import python_json


PACK = """
import hashlib,json,os,pathlib,shutil,sys,tarfile
c=json.loads(sys.argv[1]); root=pathlib.Path(c['output']); dest=pathlib.Path(c['delivery'])
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for data in iter(lambda:f.read(8<<20),b''):h.update(data)
 return h.hexdigest()
def safe(name):
 return not pathlib.Path(name).is_absolute() and '..' not in pathlib.Path(name).parts
receipt=root/'completion.json'; result=json.loads(receipt.read_bytes())
assert result['passed'] and result['condition']==c['name'] and result['checkpoint_count']==200
assert result['release_sha256']==c['release_sha256'] and result['collection_sha256']==c['collection_sha256']
files={str(n):root/n for n in result['files']}; expected=dict(result['files'])
files['completion.json']=receipt; expected['completion.json']=sha(receipt)
base=dest.parent
label=c['label']; root_job, index=c['job_id'].split('_')
for ext in ('out','err'):
 p=base/'logs'/f'{label}_{root_job}_{index}.{ext}'; files['scheduler.'+ext]=p; expected['scheduler.'+ext]=sha(p)
evidence=base/'cache_evidence'/(c['name']+'.json')
assert json.loads(evidence.read_bytes())['plan_sha256']==c['plan_sha256']
files['compiler_cache.json']=evidence;expected['compiler_cache.json']=sha(evidence)
assert all(safe(n) for n in files)
dest.mkdir(exist_ok=True)
marker=dest/(c['name']+'.json'); archive=dest/(c['name']+'.tar.gz')
if marker.exists():
 meta=json.loads(marker.read_bytes())
 assert meta['files']==expected and sha(archive)==meta['archive_sha256']
else:
 partial=archive.with_suffix('.partial')
 assert not archive.exists() and not partial.exists()
 total=sum(p.stat().st_size for p in files.values())
 assert shutil.disk_usage(dest).free >= total*1.05/.8+(16<<20)
 with tarfile.open(partial,'x:gz',compresslevel=1) as tar:
  for n,p in files.items():
   assert sha(p)==expected[n],n
   tar.add(p,arcname=n,recursive=False)
 os.replace(partial,archive)
 meta=dict(archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
           files=expected,uncompressed_bytes=total)
 with marker.open('x') as f:json.dump(meta,f,indent=2)
print('CAMPAIGN_JSON '+json.dumps(meta))
"""


def snapshot():
    submitted = read(REPORT / "submitted.json")
    jobs = submitted["jobs"]
    rows = {row[0]: row[1:] for row in accounting("gpu", jobs.values(), GPU_PYTHON)}
    live = python_json("gpu", GPU_PYTHON, """
import sys,json,pathlib,subprocess,os
p=json.loads(sys.argv[1]); out={}
for name,root in p['outputs'].items():
 f=pathlib.Path(root)/'progress.json'
 if f.is_file():out[name]=json.loads(f.read_bytes())
queue=subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i|%T|%R'],text=True)
entries={}
for line in queue.splitlines():
 if '|' in line:
  job,state,reason=line.split('|',2)
  if job.split('_',1)[0] in p['jobs']:entries[job]=dict(state=state,reason_or_nodes=reason)
print('CAMPAIGN_JSON '+json.dumps(dict(progress=out,queue=entries)))
""", dict(outputs={n: r["output"] for n, r in submitted["routes"].items()},
          jobs=list(map(str, jobs.values()))), timeout=45)
    results = {}
    for name, route in submitted["routes"].items():
        job_id = f"{jobs[str(route['energy'])]}_{route['index']}"
        required = {job_id + ext for ext in ("", ".batch", ".extern")}
        related = {job: state for job, state in rows.items() if job in required or job.startswith(job_id + ".")}
        results[name] = dict(job_id=job_id, scheduler=related, queue=live["queue"].get(job_id),
                             progress=live["progress"].get(name),
                             completed=required <= related.keys() and all(v == ["COMPLETED", "0:0"] for v in related.values()))
    return submitted, results


def validate_delivery(folder, route, name, job_id):
    completion = read(folder / "completion.json")
    if (completion["iterations"], completion["save_step"], completion["checkpoint_count"]) != (
            route["iterations"], route["save_step"], 200):
        raise ValueError("Formal save loop differs")
    if read(folder / "allocation.json")["array_element"] != job_id:
        raise ValueError("Formal output allocation differs from the accounting identity")
    numerical = read(folder / "numerical_QA.json")
    if not numerical["passed"] or not all(row["passed"] for row in numerical["operators"].values()):
        raise ValueError("Operator numerical audit failed")
    frozen_body = DATA / "payload/backgrounds" / f"{name}.npy"
    if digest(frozen_body) != completion["background_sha256"] or digest(folder / "fixed_body_background.npy") != digest(frozen_body):
        raise ValueError("Delivered background differs from frozen template")
    history = np.load(folder / "active_history.npy", mmap_mode="r")
    body = np.load(folder / "fixed_body_background.npy")
    cross = np.load(folder / "fixed_instrument_C440to218.npy") if route["energy"] == 218 else np.zeros_like(body)
    if not np.isfinite(cross).all() or (cross < 0).any():
        raise ValueError("Invalid 440-to-218 background")
    if route["energy"] == 218:
        parent_name = route["parent440_condition"]
        parent = read(REPORT / "delivery" / f"{parent_name}.json")
        if parent["files"]["completion.json"] != completion["parent440_completion_sha256"]:
            raise ValueError("218 parent differs from the accepted same-branch 440 route")
    if history.shape != (200, 78920) or not np.isfinite(history).all() or (history < 0).any():
        raise ValueError("Incomplete or invalid active history")
    for frame, iteration in enumerate(range(route["save_step"], route["iterations"] + 1, route["save_step"])):
        with np.load(folder / "checkpoints" / f"iter_{iteration:05d}.npz") as state:
            if int(state["iteration"]) != iteration or int(state["background_last_update"]) != 0 or not (
                    np.array_equal(state["image"].reshape(-1), history[frame])
                    and np.array_equal(state["body_background"], body)
                    and np.array_equal(state["instrument_cross_background"], cross)):
                raise ValueError("Fixed background or checkpoint history does not match")
    with np.load(DATA / "payload/whole_geometry.npz") as g:
        active = g["active_indices"]
    final = np.load(folder / "final_full.npy")
    if final.shape != (132040,) or np.count_nonzero(final) != np.count_nonzero(final[active]) or not (
            np.array_equal(final[active], history[-1]) and
            np.array_equal(history[-1], np.load(folder / "final_active.npy").reshape(-1))):
        raise ValueError("Wrong final domain or last checkpoint")


def deliver():
    submission, statuses = snapshot()
    for name, state in statuses.items():
        if not state["completed"]:
            continue
        route = submission["routes"][name]
        marker = REPORT / "delivery" / (name + ".json")
        target = DATA / "formal_delivery" / name
        if marker.exists():
            verify(target, read(marker)["files"])
            continue
        proof = python_json("gpu", GPU_PYTHON, PACK, dict(
            name=name, output=route["output"], delivery=GPU + "/delivery",
            job_id=state["job_id"], label=f"ACW_v2_{route['energy']}_1107",
            release_sha256=submission["release_sha256"], collection_sha256=route["collection_sha256"],
            plan_sha256=submission["plan_sha256"]), timeout=180, readonly=False)
        transfer = DATA / "delivery"
        transfer.mkdir(exist_ok=True)
        needed = proof["archive_bytes"] + proof["uncompressed_bytes"]
        if shutil.disk_usage(transfer).free < needed / .8:
            raise OSError("Local archive and extraction need 20% headroom")
        archive = transfer / (name + ".tar.gz")
        if not archive.exists():
            partial = archive.with_suffix(".partial")
            download_gpu_archive_stream(proof["archive"], partial, proof["archive_sha256"], proof["archive_bytes"])
            partial.replace(archive)
        if digest(archive) != proof["archive_sha256"]:
            raise ValueError("Archive SHA differs")
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if len(members) != len(proof["files"]) or {m.name for m in members} != set(proof["files"]):
                raise ValueError("Archive member list differs")
            for member in members:
                path = Path(member.name)
                if not member.isfile() or path.is_absolute() or ".." in path.parts:
                    raise ValueError("Unsafe archive member")
                local = target / path
                if not local.exists():
                    local.parent.mkdir(parents=True, exist_ok=True)
                    partial = local.with_name(local.name + ".partial")
                    with tar.extractfile(member) as src, partial.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    if digest(partial) != proof["files"][member.name]:
                        raise ValueError("Partial file SHA differs")
                    partial.replace(local)
        verify(target, proof["files"])
        validate_delivery(target, route, name, state["job_id"])
        write(marker, dict(passed=True, local=str(target), scheduler=state["scheduler"], all_200_frames_verified=True, **proof))
        print("AC225_SCATTER_V2_STRICT_ROUTE_DELIVERED", name, flush=True)
    receipts = {n: read(REPORT / "delivery" / (n + ".json")) for n in statuses
                if (REPORT / "delivery" / (n + ".json")).exists()}
    if len(receipts) == len(statuses) and not (REPORT / "strict_delivery.json").exists():
        write(REPORT / "strict_delivery.json", dict(passed=True, routes=len(receipts), checkpoints=200 * len(receipts),
              release_sha256=submission["release_sha256"], receipts={k: v["files"]["completion.json"] for k, v in receipts.items()}))
    return dict(delivered=list(receipts), status=statuses)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("status", "deliver"))
    args = p.parse_args()
    print(json.dumps(snapshot()[1] if args.action == "status" else deliver(), indent=2))
