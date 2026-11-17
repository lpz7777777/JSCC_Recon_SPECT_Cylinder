"""Freeze and submit an independent formal scatter reconstruction on the accepted water data."""

import argparse
import json
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ac225_corrections.io import digest, hashes, immutable, read, verify, write
from ac225_corrections.windows import validate_tagged_counts
from ac225_water_cache_repair_20261104 import controller_absent
from ac225_water_scatter_model_20261107 import SIDE_WINDOWS, fit_source_ratios, fit_template
from ac225_water_workflow_20261103 import DATA as ORIGINAL, REPORT as OLD_REPORT
from campaign_20261102_io import execute, python_json, upload_files
from derenzo_campaign_common import GPU_PROJECT, GPU_PYTHON, HERE, ROOT
from ehe_5e9_workflow import q


TAG = "ac225_water_scatter_v2_20261107"
DATA = HERE / "generated" / TAG
REPORT = HERE / "reports/NEMA_Body_H60" / TAG
GPU = GPU_PROJECT + "/generated/" + TAG
FROZEN = ORIGINAL / "repair_payload_20261104"
ROUTE_ORDER = ("EHE_5e9", "EHE_5e10", "JSCC_5e9", "JSCC_5e10")
RUNTIME = "ac225_water_scatter_runtime_20261107.py"
LAUNCHER = "ac225_water_scatter_launch_20261107.py"


def input_path(dataset):
    return ORIGINAL / "water_inputs" / dataset


def read_tags(dataset, window, collection):
    path = input_path(dataset)
    names = [window + suffix for suffix in (".npy", "_tags.npy")]
    verify(path, {n: collection["files"][n] for n in names})
    observed, tags = (np.load(path / n) for n in names)
    validate_tagged_counts(observed, tags)
    return tags


def make_closure_truth(payload):
    path = HERE / "generated/NEMA_Body_H60/truth_3mm.npz"
    expected = read(HERE / "reports/NEMA_Body_H60/manifest.json")["truth_sha256"]
    verify(path.parent, {path.name: expected})
    folder = payload / "closure_truth"
    folder.mkdir()
    with np.load(payload / "whole_geometry.npz") as g:
        active = g["active_indices"]
        xyz = g["coordinates_mm"][active]
        volumes = (g["cell_volume_mm3"] * g["ellipse_fraction"])[active]
    with np.load(path) as truth:
        axes = tuple(truth[f"{axis}_mm"] for axis in "zyx")
        for energy in (218, 440):
            source = truth[f"activity_{energy}_zyx"]
            evaluate = RegularGridInterpolator(axes, source, bounds_error=False, fill_value=0.)
            density = evaluate(xyz[:, ::-1])
            density /= np.dot(density, volumes)
            np.save(folder / f"{energy}.npy", density[:, None])
    return expected


def freeze():
    marker = REPORT / "freeze.json"
    if marker.exists():
        receipt = read(marker)
        verify(DATA, receipt["local_files"])
        return receipt
    controller_absent()
    strict = read(OLD_REPORT / "strict_delivery.json")
    if not strict["passed"] or not strict["all_files_SHA_verified"] or strict["routes"] != 32:
        raise ValueError("Full previous water delivery required")
    old_release = read(FROZEN / "release_manifest.json")
    cfg = read(FROZEN / "contract.json")
    payload = DATA / "payload"
    payload.mkdir(parents=True, exist_ok=False)
    members = [n for n in old_release["files"] if n.startswith(("ac225_corrections/", "params/")) or n in
               ("whole_geometry.npz", "nema_body_h60_config.json", "side_contract.json", "ehe_common.py",
                "torch_active_operator.py", "single_checkpoint_mlem.py", "test_ac225_corrections.py")]
    verify(FROZEN, {n: old_release["files"][n] for n in members})
    for name in members:
        path = payload / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(FROZEN / name, path)
    shutil.copyfile(HERE / RUNTIME, payload / RUNTIME)
    shutil.copyfile(HERE / "ac225_water_scatter_model_20261107.py", payload / "ac225_water_scatter_model_20261107.py")
    shutil.copyfile(HERE / "ac225_formal_cache_launcher_20261104.py", payload / "ac225_formal_cache_launcher_20261104.py")
    shutil.copyfile(HERE / LAUNCHER, DATA / "launcher.py")
    truth_sha = make_closure_truth(payload)
    (payload / "backgrounds").mkdir()

    collections = {dataset: read(input_path(dataset) / "collection.json") for dataset in ROUTE_ORDER}
    old_routes = read(OLD_REPORT / "formal_effective_routes_r5.json")["routes"]
    routes, fits = {}, {}
    for index, dataset in enumerate(ROUTE_ORDER):
        system, dose = dataset.split("_")
        donor = f"{system}_{'5e10' if dose == '5e9' else '5e9'}"
        target, reference = collections[dataset], collections[donor]
        if not all(c["passed"] and c["actual_Geant4"] for c in (target, reference)):
            raise ValueError("Independent real Geant4 collections required")
        if set(target["worker_receipts"].values()) & set(reference["worker_receipts"].values()):
            raise ValueError("Target and donor reused workers")
        verify(input_path(dataset), {n + ".npy": target["files"][n + ".npy"] for n in SIDE_WINDOWS})
        target_sides = {n: np.load(input_path(dataset) / (n + ".npy")) for n in SIDE_WINDOWS}
        donor_sides = {n: read_tags(donor, n, reference) for n in SIDE_WINDOWS}
        ratios, counts, predicted = fit_source_ratios(target_sides, donor_sides)
        for energy in (440, 218):
            name = f"{dataset}_fixed_mc_scatter_{energy}"
            old = f"{dataset}_attenuation_{energy}"
            old_route = old_routes[old]
            old_output = Path(strict["local"]) / old
            verify(old_output, {"completion.json": strict["conditions"][old]["completion_sha256"]})
            old_receipt = read(old_output / "completion.json")
            layout = np.load(payload / "params" / system / str(energy) / "physical_layout.npy")
            template, bandwidth = fit_template(read_tags(donor, f"{energy}_main", reference), layout)
            background = np.einsum("edv,e->dv", template, ratios)
            np.save(payload / "backgrounds" / f"{name}.npy", background)
            fits[name] = dict(donor=donor, donor_collection_sha256=digest(input_path(donor) / "collection.json"),
                              fitted_target_donor_emission_ratios=ratios.tolist(),
                              known_target_donor_emission_ratios=(np.array(target["primary_counts"]) /
                                                                 reference["primary_counts"]).tolist(),
                              side_windows=SIDE_WINDOWS, observed_side_counts=counts.tolist(),
                              model_side_counts=predicted.tolist(), body_counts=float(background.sum()),
                              **bandwidth)
            routes[name] = dict(dataset=dataset, system=system, index=index, energy=energy, input=old_route["input"],
                                output=GPU + "/formal/" + name, iterations=cfg["conditions"][old]["iterations"],
                                save_step=cfg["conditions"][old]["save_step"],
                                collection_sha256=digest(input_path(dataset) / "collection.json"),
                                old_matched_S_path=old_route["output"] + "/matched_sensitivity.npy",
                                old_matched_S_sha256=old_receipt["files"]["matched_sensitivity.npy"])
            if energy == 218:
                parent = f"{dataset}_fixed_mc_scatter_440"
                routes[name].update(parent440=routes[parent]["output"], parent440_condition=parent)
        print("FROZEN_MODEL", dataset, ratios.tolist(), flush=True)
    write(payload / "scatter_model.json", dict(routes=fits, original_truth_sha256=truth_sha,
          parent_manifest_sha256=digest(FROZEN / "release_manifest.json"),
          original_strict_delivery_sha256=digest(OLD_REPORT / "strict_delivery.json"),
          target_truth_tags_used_to_fit=False, source_shape="same known NEMA phantom in independent MC"))
    write(payload / "release_manifest.json", dict(files=hashes(payload)))
    archive = DATA / "bundle.tar.gz"
    with tarfile.open(archive, "x:gz", compresslevel=3) as tar:
        for name in hashes(payload):
            tar.add(payload / name, arcname=name)
    result = dict(local_files={"bundle.tar.gz": digest(archive), "launcher.py": digest(DATA / "launcher.py")},
                  release_sha256=digest(payload / "release_manifest.json"), routes=routes,
                  side_root=read(OLD_REPORT / "repair_20261104/freeze.json")["output"])
    write(marker, result)
    return result


def preflight(frozen):
    probe = """
import hashlib,json,os,pathlib,sys
p=json.loads(sys.argv[1]);r=pathlib.Path(p['parent']);s=os.statvfs(r)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
assert os.access(r,os.W_OK) and s.f_bavail*s.f_frsize >= 12*(1<<30)
root=pathlib.Path(p['new']); formal=root/'formal'
assert not formal.exists() or (formal.is_dir() and not any(formal.iterdir()))
side=pathlib.Path(p['side'])/'completion.json'; sc=json.loads(side.read_bytes())
assert sc['passed'] and len(sc['factors'])==12
for route in p['routes'].values():
 f=pathlib.Path(route['input'])/'collection.json'
 assert sha(f)==route['collection_sha256']
 assert sha(pathlib.Path(route['old_matched_S_path']))==route['old_matched_S_sha256']
 assert not pathlib.Path(route['output']).exists()
print('CAMPAIGN_JSON '+json.dumps(dict(passed=True,available_bytes=s.f_bavail*s.f_frsize,
 total_bytes=s.f_blocks*s.f_frsize,side_completion_sha256=sha(side))))
"""
    return python_json("gpu", GPU_PYTHON, probe, dict(parent=GPU.rsplit("/", 1)[0], new=GPU,
                       side=frozen["side_root"], routes=frozen["routes"]), timeout=90)


def audit_frozen_projection():
    frozen = read(REPORT / "freeze.json")
    payload = DATA / "payload"
    manifest = payload / "release_manifest.json"
    verify(payload, {"release_manifest.json": frozen["release_sha256"]})
    prior = read(ROOT / "Analysis_Plots/research/ac225_water_scatter_audit_20261107/audit.json")
    checks = {}
    for name, route in frozen["routes"].items():
        dataset, energy = route["dataset"], route["energy"]
        key = f"backgrounds/{name}.npy"
        expected = read(manifest)["files"][key]
        verify(payload, {key: expected})
        pred = np.load(payload / key)
        tagged = read_tags(dataset, f"{energy}_main", read(input_path(dataset) / "collection.json"))
        truth = tagged[:, 1:].sum(axis=(0, 1))
        old = prior["results"][f"{dataset}_{energy}"]["routes"]["attenuation_tew_pool3"]["background_track"][-1]
        checks[name] = dict(predicted=float(pred.sum()), truth=int(truth.sum()),
                            ratio=float(pred.sum() / truth.sum()),
                            L2=float(np.linalg.norm(pred - truth) / np.linalg.norm(truth)),
                            old_pool_ratio=old["ratio_to_tagged_body"], old_pool_L2=old["relative_L2_to_tagged_body"],
                            sha256=expected)
    result = dict(release_sha256=frozen["release_sha256"], target_tags_evaluated_after_model_freeze=True,
                  target_tags_used_in_fit=False, results=checks)
    marker = REPORT / "frozen_projection_QA.json"
    if marker.exists():
        recorded = read(marker)
        for name in checks:
            if recorded["results"][name]["sha256"] != checks[name]["sha256"] or not np.isclose(
                    recorded["results"][name]["L2"], checks[name]["L2"], rtol=1e-10, atol=1e-12):
                raise ValueError("Held-out scatter audit differs")
        return recorded
    write(marker, result)
    return result


def submit_array(energy, dependency, plan_sha):
    label = f"ACW_v2_{energy}_1107"
    marker = REPORT / f"job_{energy}.json"
    intent = REPORT / f"submission_intent_{energy}.json"
    if marker.exists():
        return read(marker)
    if intent.exists():
        raise ValueError("Unresolved scheduler intent; reconcile with Slurm")
    script = DATA / f"formal_{energy}.sh"
    body = """#!/usr/bin/env bash
set -euo pipefail
source /etc/profile.d/modules.sh
module load cuda/12.9
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"""
    body += f"{q(GPU_PYTHON)} -B {q(GPU + '/launcher.py')} {q(GPU + '/plan.json')} {plan_sha} {energy}\n"
    immutable(script, body.encode())
    upload_files("gpu", GPU, {script.name: script})
    execute("gpu", "bash -n " + q(GPU + "/" + script.name), readonly=True)
    queue = execute("gpu", 'squeue -h -u "$USER" -o "%i|%j|%T"', readonly=True)
    if any(line.split("|")[1:2] == [label] for line in queue.splitlines()):
        raise ValueError("Same named active formal job already exists")
    options = "-p gpu_5090 --account=scxi717 --qos=gpugpu -N1 -n1 -c8 --gres=gpu:1 --time=1440 --array=0-3%4"
    if dependency:
        options += f" --dependency=aftercorr:{dependency}"
    command = (f"sbatch --parsable {options} --job-name={label} --output={q(GPU + '/logs/' + label + '_%A_%a.out')}"
               f" --error={q(GPU + '/logs/' + label + '_%A_%a.err')} {q(GPU + '/' + script.name)}")
    record = dict(label=label, command=command, plan_sha256=plan_sha, script_sha256=digest(script),
                  submitted_epoch=time.time())
    write(intent, record)
    response = execute("gpu", command, timeout=60).strip()
    job = response.split(";")[0]
    if not job.isdigit():
        raise ValueError("Uncertain sbatch result; intent preserved")
    record.update(job=int(job), response=response)
    write(marker, record)
    return record


def deploy_and_submit():
    controller_absent()
    frozen = freeze()
    marker = REPORT / "deployment.json"
    if not marker.exists():
        proof = preflight(frozen)
        helper = "make_nema_body_h60.py"
        verify(FROZEN, {helper: read(FROZEN / "release_manifest.json")["files"][helper]})
        plan = dict(frozen, side_completion_sha256=proof["side_completion_sha256"],
                    archive=GPU + "/bundle.tar.gz", launcher_sha256=digest(DATA / "launcher.py"),
                    per_route_required_free_bytes=2 << 30)
        immutable(DATA / "plan.json", (json.dumps(plan, indent=2, allow_nan=False) + "\n").encode())
        transfer = {name: DATA / name for name in ("bundle.tar.gz", "launcher.py", "plan.json")}
        transfer["check_support/" + helper] = FROZEN / helper
        upload_files("gpu", GPU, transfer)
        prepare = """
import hashlib,json,pathlib,sys
p=json.loads(sys.argv[1]);root=pathlib.Path(p['path'])
for n,h in p['files'].items(): assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
for name in ('formal','logs','cache_evidence'): (root/name).mkdir(exist_ok=True)
print('CAMPAIGN_JSON '+json.dumps(dict(passed=True)))
"""
        files = {name: digest(path) for name, path in transfer.items()}
        proof["remote_file_check"] = python_json("gpu", GPU_PYTHON, prepare, dict(path=GPU, files=files),
                                                  timeout=120, readonly=False)
        check = (f"PYTHONPATH={q(GPU + '/check_support')} {q(GPU_PYTHON)} -B {q(GPU + '/launcher.py')} {q(GPU + '/plan.json')} "
                 f"{files['plan.json']} check")
        proof["linux_checks"] = execute("gpu", "timeout 660s " + q("env") + " " + check,
                                        timeout=690, readonly=False).strip()
        if "AC225_SCATTER_V2_LINUX_CHECKS_PASSED" not in proof["linux_checks"]:
            raise ValueError("Linux checks marker missing")
        proof["files"] = files
        write(marker, proof)
    plan_sha = read(marker)["files"]["plan.json"]
    job440 = submit_array(440, None, plan_sha)
    job218 = submit_array(218, job440["job"], plan_sha)
    result = dict(jobs={"440": job440["job"], "218": job218["job"]}, routes=frozen["routes"],
                  release_sha256=frozen["release_sha256"], plan_sha256=plan_sha)
    write(REPORT / "submitted.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "audit", "submit"))
    args = parser.parse_args()
    result = {"freeze": freeze, "audit": audit_frozen_projection, "submit": deploy_and_submit}[args.action]()
    print(json.dumps({k: v for k, v in result.items() if k != "routes"}, indent=2))
