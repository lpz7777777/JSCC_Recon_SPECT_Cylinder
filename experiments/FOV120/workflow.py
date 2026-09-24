"""Prepare, run, and collect isolated FOV120 Geant4 tasks (local or SLURM array)."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fov_config import load_config

GENERATED = Path(__file__).resolve().parent / "generated"


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def gps_source(energy, center, radius=None, height=None, intensity=1, first=False):
    lines = [f"/gps/source/intensity {intensity:.15g}" if first else f"/gps/source/add {intensity:.15g}",
             "/gps/particle gamma", f"/gps/energy {energy} keV", "/gps/number 1",
             f"/gps/pos/centre {center[0]:.9g} {center[1]:.9g} {center[2]:.9g} mm"]
    if radius is None:
        lines += ["/gps/pos/type Point"]
    else:
        lines += ["/gps/pos/type Volume", "/gps/pos/shape Cylinder",
                  f"/gps/pos/radius {radius} mm", f"/gps/pos/halfz {height/2} mm"]
    return "\n".join(lines + ["/gps/ang/type iso", "/gps/ang/mintheta 0 deg", "/gps/ang/maxtheta 180 deg"]) + "\n"


def prepare(output, cfg):
    if (output / "jobs.json").exists():
        raise FileExistsError(f"Task manifest already exists: {output}")
    macros = output / "macros"
    macros.mkdir(parents=True, exist_ok=True)
    jobs = []

    def add(dataset, level, view, body, total, workers, mono=None, role="imaging"):
        if total % workers or not 1 <= total//workers <= 2147483647:
            raise ValueError("Invalid worker photon allocation")
        macro = macros / f"{dataset}_{level}_v{view:02d}.mac"
        text = body + f"\n/run/beamOn {total//workers}\n"
        macro.write_text(text, encoding="ascii")
        for worker in range(workers):
            index = len(jobs)
            jobs.append({"index": index, "dataset": dataset, "level": level, "view": view,
                         "worker": worker, "photons": total//workers, "seed": 26092401 + index,
                         "mono_keV": mono, "role": role,
                         "macro": macro.relative_to(output).as_posix(), "macro_sha256": digest(macro)})

    # Independent calibration pilot, added production counts, and sensitivity validation.
    for role in ("calibration", "sensitivity", "sensitivity_validation"):
        for energy in (218, 440):
            if role.startswith("sensitivity") and energy == 218:
                continue
            body = gps_source(energy, cfg["geant4_center_mm"], cfg["support_radius_mm"], cfg["height_mm"], first=True)
            for level, photons, workers in (("pilot", 100_000_000, 10), ("extension", 900_000_000, 90)):
                add(f"{role}_{energy}", level, 1, body, photons, workers, energy, role)

    # A point is its own dataset. Four azimuths for each noncentral radius.
    for energy in (218, 440):
        for z in (0, -15, 15, -30, 30, -45, 45, -57, 57):
            for radius in (0, 75, 135):
                for angle in ((0,) if radius == 0 else (0, 90, 180, 270)):
                    radians = math.radians(angle)
                    xyz = (radius*math.cos(radians), -245+radius*math.sin(radians), z)
                    add(f"Point_E{energy}_r{radius}_a{angle}_z{z:+d}", "1e7", 1,
                        gps_source(energy, xyz, first=True), 10_000_000, 1, energy, "point")

    rods = []
    for z in (-45, 0, 45):
        for index, diameter in enumerate(range(10, 31, 4)):
            theta = index*math.pi/3
            rods.append({"energy": 218 if index % 2 == 0 else 440,
                         "center_mm": [60*math.cos(theta), 60*math.sin(theta), z],
                         "radius_mm": diameter/2, "height_mm": 30, "excess_activity": 5})
    truth_spec = {"height_mm": cfg["height_mm"], "physical_radius_mm": 150,
                  "background_activity": 1, "rods": rods,
                  "gamma_yields": cfg["gamma_yields"],
                  "normalization": "common activity scale; yield times volume; no per-energy renormalization"}
    write_json(output / "Contrast_truth.json", truth_spec)
    for dataset in ("Uniform", "Contrast", "XCAT"):
        for photons in cfg["count_levels"]:
            level = f"1e{int(math.log10(photons))}"
            for view in range(1, 21):
                if dataset == "XCAT":
                    source = GENERATED / "XCAT" / f"view_{view:02d}_mixed.mac"
                    body = re.sub(r"/run/beamOn\s+\d+", "", source.read_text(encoding="ascii"))
                else:
                    body = "# FOV120; gamma proxy; no object material\n/gps/source/multiplevertex false\n"
                    for index, energy in enumerate((218, 440)):
                        weight = cfg["gamma_yields"][str(energy)] * math.pi*150**2*120
                        body += gps_source(energy, (0, -245, 0), 150, 120, weight, first=index == 0)
                    if dataset == "Contrast":
                        theta = math.radians((view-1)*18)
                        for rod in rods:
                            x,y,z = rod["center_mm"]
                            center = (x*math.cos(theta)+y*math.sin(theta), -245+y*math.cos(theta)-x*math.sin(theta), z)
                            weight = 5 * cfg["gamma_yields"][str(rod["energy"]) ] * math.pi*rod["radius_mm"]**2*30
                            body += gps_source(rod["energy"], center, rod["radius_mm"], 30, weight)
                add(dataset, level, view, body, photons//20, 10)
    write_json(output / "jobs.json", {"format_version": 1, "config_sha256": digest(Path(__file__).with_name("config.json")),
                                     "jobs": jobs})
    print(f"Prepared {len(jobs)} uniquely seeded tasks: {output / 'jobs.json'}")


def load_jobs(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = data["jobs"]
    if len({j["seed"] for j in jobs}) != len(jobs) or [j["index"] for j in jobs] != list(range(len(jobs))):
        raise ValueError("Duplicate seeds or malformed task indices")
    return jobs


def prepare_point_imaging(output):
    """Independent 20-view point datasets, separate from single-view response scans."""
    output.mkdir(parents=True,exist_ok=False)
    macros=output/"macros";macros.mkdir()
    jobs=[]
    for energy in (218,440):
        for z in (0,-15,15,-30,30,-45,45,-57,57):
            for radius in (0,75,135):
                for angle in ((0,) if radius==0 else (0,90,180,270)):
                    phi=math.radians(angle)
                    xyz=(radius*math.cos(phi),radius*math.sin(phi),z)
                    dataset=f"PointRecon_E{energy}_r{radius}_a{angle}_z{z:+d}"
                    for view in range(1,21):
                        theta=math.radians((view-1)*18)
                        x,y,z=xyz
                        center=(x*math.cos(theta)+y*math.sin(theta),-245+y*math.cos(theta)-x*math.sin(theta),z)
                        macro=macros/f"{dataset}_v{view:02d}.mac"
                        macro.write_text(gps_source(energy,center,first=True)+"/run/beamOn 500000\n",encoding="ascii")
                        index=len(jobs)
                        jobs.append({"index":index,"dataset":dataset,"level":"1e7","view":view,
                            "worker":0,"photons":500000,"seed":26192401+index,"mono_keV":energy,
                            "role":"point_imaging","position_mm":list(xyz),
                            "macro":macro.relative_to(output).as_posix(),"macro_sha256":digest(macro)})
    write_json(output/"jobs.json",{"format_version":1,"jobs":jobs})
    print(f"Prepared {len(jobs)} point imaging tasks")


def run_job(manifest, index, executable, crystal, smoke=False):
    jobs = load_jobs(manifest)
    if index < 0 or index >= len(jobs):
        raise ValueError("Task index out of range")
    job = jobs[index]
    base = manifest.parent
    macro = base / job["macro"]
    if digest(macro) != job["macro_sha256"]:
        raise ValueError("Macro changed after manifest generation")
    folder = base / ("smoke" if smoke else "workers") / f"{index:05d}"
    folder.mkdir(parents=True, exist_ok=False)  # never append to old Geant4 output
    shutil.copy2(crystal, folder / "CrystalMatrix.txt")
    body = macro.read_text(encoding="ascii")
    photons = min(job["photons"], 10000) if smoke else job["photons"]
    body = re.sub(r"/run/beamOn\s+\d+", f"/run/beamOn {photons}", body)
    (folder / "run.mac").write_text(body, encoding="ascii")
    environment = os.environ.copy()
    environment["JSCC_RANDOM_SEED"] = str(job["seed"])
    start = time.time()
    with (folder / "console.log").open("w", encoding="utf-8") as log:
        process = subprocess.run([str(executable.resolve()), "run.mac"], cwd=folder, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
    record = dict(job, status="failed", exit_code=process.returncode,
                  elapsed_seconds=time.time()-start, simulated_photons=photons,
                  executable_sha256=digest(executable), crystal_sha256=digest(crystal))
    write_json(folder / "worker.json", record)
    if process.returncode:
        raise RuntimeError(f"Geant4 failed: {folder / 'console.log'}")
    primary = np.loadtxt(folder / "PrimaryCount.csv", delimiter=",", dtype=np.int64, ndmin=2)
    if primary.shape != (1, 3) or int(primary.sum()) != photons or primary[0, 2] != 0:
        raise ValueError(f"Primary count mismatch: {folder}")
    if job["mono_keV"] and primary[0, 0 if job["mono_keV"] == 440 else 1] != 0:
        raise ValueError("Monoenergetic task emitted wrong primary energy")
    for energy in (218, 440):
        counts = np.loadtxt(folder / f"CntStat_{energy}.csv", delimiter=",", dtype=np.int64, ndmin=2)
        if counts.shape != (1, 10496) or np.any(counts < 0):
            raise ValueError("Detector count output shape/value mismatch")
    record.update(status="complete", primary_counts=primary[0].tolist(),
                  photons_per_second=photons/max(record["elapsed_seconds"], 1e-9))
    record["output_sha256"] = {name: digest(folder/name) for name in
                              ("CntStat_218.csv", "CntStat_440.csv", "List.csv", "PrimaryCount.csv")}
    write_json(folder / "worker.json", record)
    print(json.dumps(record, indent=2))


def collect(manifest, dataset, level, destination):
    jobs = [j for j in load_jobs(manifest) if j["dataset"] == dataset and (level == "all" or j["level"] == level)]
    if not jobs:
        raise ValueError("No matching tasks")
    views = sorted({j["view"] for j in jobs})
    if jobs[0]["role"] in ("imaging","point_imaging") and views != list(range(1, 21)):
        raise ValueError("Missing imaging views")
    records = []
    for job in jobs:
        folder = manifest.parent / "workers" / f"{job['index']:05d}"
        record = json.loads((folder / "worker.json").read_text(encoding="utf-8"))
        if record["status"] != "complete" or any(record[k] != job[k] for k in job):
            raise ValueError(f"Failed or mismatched task: {folder}")
        if record["simulated_photons"] != job["photons"]:
            raise ValueError("Smoke data cannot enter production collection")
        for name, checksum in record["output_sha256"].items():
            if digest(folder / name) != checksum:
                raise ValueError(f"Worker output changed: {folder / name}")
        records.append((folder, record))
    if len({r["crystal_sha256"] for _,r in records}) != 1 or len({r["executable_sha256"] for _,r in records}) != 1:
        raise ValueError("Worker detector/executable mismatch")
    marker = destination / "collections" / f"{dataset}_{level}.json"
    if marker.exists():
        raise FileExistsError(marker)
    projections = {e: np.zeros((len(views), 10496), dtype=np.int64) for e in (218,440)}
    listdir = destination / "List" / "218-440keV_RotateNum20_Geant4JSCC" / f"List_{dataset}_{level}"
    listdir.mkdir(parents=True, exist_ok=False)
    for view_index,view in enumerate(views):
        with (listdir / f"{view}.csv").open("wb") as sink:
            for folder,record in records:
                if record["view"] != view:
                    continue
                for energy in projections:
                    projections[energy][view_index] += np.loadtxt(folder/f"CntStat_{energy}.csv",delimiter=",",dtype=np.int64).reshape(-1)
                with (folder / "List.csv").open("rb") as source:
                    shutil.copyfileobj(source, sink)
    for energy,counts in projections.items():
        directory = destination / "CntStat" / f"{energy}keV_RotateNum20_Geant4JSCC"
        directory.mkdir(parents=True, exist_ok=True)
        # Loader reshapes (views, detectors), then transposes internally.
        np.savetxt(directory / f"CntStat_{dataset}_{level}.csv", counts, delimiter=",", fmt="%d")
    write_json(marker, {"dataset": dataset, "level": level, "views": views,
                        "primary_counts": np.sum([r["primary_counts"] for _,r in records],axis=0).tolist(),
                        "worker_indices": [r["index"] for _,r in records],
                        "seeds": [r["seed"] for _,r in records],
                        "manifest_sha256": digest(manifest), "list_dir": str(listdir.resolve())})
    print(marker)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, default=GENERATED / "Simulation")
    points=sub.add_parser("prepare-points")
    points.add_argument("--output",type=Path,default=GENERATED/"PointImaging")
    run = sub.add_parser("run")
    run.add_argument("--manifest", type=Path, default=GENERATED / "Simulation/jobs.json")
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--executable", type=Path, required=True)
    run.add_argument("--crystal", type=Path, required=True)
    run.add_argument("--smoke", action="store_true")
    merge = sub.add_parser("collect")
    merge.add_argument("--manifest", type=Path, default=GENERATED / "Simulation/jobs.json")
    merge.add_argument("--dataset", required=True)
    merge.add_argument("--level", required=True)
    merge.add_argument("--destination", type=Path, default=GENERATED)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output.resolve(), load_config())
    elif args.command == "prepare-points":
        prepare_point_imaging(args.output.resolve())
    elif args.command == "run":
        run_job(args.manifest.resolve(), args.index, args.executable, args.crystal, args.smoke)
    else:
        collect(args.manifest.resolve(), args.dataset, args.level, args.destination)


if __name__ == "__main__":
    main()
