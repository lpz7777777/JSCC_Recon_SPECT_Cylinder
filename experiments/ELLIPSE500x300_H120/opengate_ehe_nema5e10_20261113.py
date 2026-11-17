"""Independent 50B-gamma vacuum NEMA acquisition using the corrected source.

Reuse the frozen geometry, source images, MC response and matched projector.
Never reuse imaging observations from 5e9 or rerun successful stages.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from opengate_ehe_common_20261113 import HERE, read, save, save_array, sha, now

TAG = "opengate_ehe_nema5e10_20261113"
DATA = HERE / "generated" / TAG
REPORT = HERE / "reports/NEMA_Body_H60" / TAG
CPU = "/WORK/maty_work/lpz/" + TAG
PARENT_TAG = "opengate_ehe_nema5e9_rotfix_20261113"
PARENT_DATA = HERE / "generated" / PARENT_TAG
PARENT_REPORT = HERE / "reports/NEMA_Body_H60" / PARENT_TAG
RESPONSE_CPU = "/WORK/maty_work/lpz/opengate_ehe_nema5e9_20261113"
LOCAL_ENV = HERE / "generated/opengate_ehe_nema5e9_20261113/local_env/Scripts/python.exe"
TOTAL = 50_000_000_000
WORKERS = 1000
CONCURRENCY = 160


def prepare():
    if (REPORT / "freeze.json").exists():
        return read(REPORT / "freeze.json")
    parent = read(PARENT_REPORT / "freeze.json")
    assert parent["release"] == "f7231adc45ed8607"
    source = PARENT_DATA / parent["payload_relative"]
    gate = read(PARENT_REPORT / "released_remaining.json")
    assert gate["actual_primary_vertex_check"]["actual_primary_vertex_rotation_verified"]
    accepted = read(PARENT_REPORT / "strict_delivery.json")
    assert accepted["passed"] and accepted["actual_imaging_primaries"] == 5_000_000_000
    payload = DATA / "payload"
    payload.mkdir(parents=True, exist_ok=False)
    for name in ("opengate_ehe_common_20261113.py", "opengate_ehe_transport_20261113.py",
                 "opengate_ehe_projector_20261113.py", "opengate_ehe_guard_20261113.py", "cpu_python.sh"):
        assert sha(source / name) == parent["files"][name]["sha256"]
        shutil.copy2(source / name, payload / name)
    inputs = payload / "inputs"
    shutil.copytree(source / "inputs", inputs)
    cfg = read(inputs / "config.json")
    p218 = cfg["imaging_primary_energy_fraction_expected"]["218"]
    rng = np.random.default_rng(615030001)
    emitted = np.zeros((20, 2), dtype=np.int64)
    jobs = []
    for view in range(20):
        n218 = int(rng.binomial(TOTAL // 20, p218))
        emitted[view] = [n218, TOTAL // 20 - n218]
        for ei, energy in enumerate((218, 440)):
            for part in range(25):
                index = len(jobs)
                jobs.append(dict(index=index, kind="imaging", view=view, angle_deg=18*view,
                    energy_keV=energy, primaries=int(emitted[view, ei]//25 + (part < emitted[view, ei]%25)),
                    seed=615130001+index, smear_seed=615430001+index, importance_weight=1.))
    assert len(jobs) == WORKERS and sum(j["primaries"] for j in jobs) == TOTAL
    assert len({j["seed"] for j in jobs} | {j["smear_seed"] for j in jobs}) == 2*WORKERS
    for key in ("repair_of", "reused_views", "recomputed_views", "reused_response_characterization_primaries"):
        cfg.pop(key, None)
    cfg.update(study=TAG, created_utc=now(), imaging_primary_gamma_total=TOTAL,
        emitted_primaries_by_view_energy=emitted.tolist(), independent_acquisition=True,
        imaging_observations_inherited=False, source_parent_release=parent["release"],
        source_rotation_implementation_sha256=sha(payload/"opengate_ehe_transport_20261113.py"),
        reused_source_coordinate_proof_sha256=sha(PARENT_REPORT/"released_remaining.json"),
        reused_response_kernel_sha256=sha(PARENT_DATA/"delivery/collected/response_kernels.npz"),
        max_response_or_imaging_concurrency=CONCURRENCY,
        response_characterization="88 already accepted GenericSource workers / 4.4B histories, no new response jobs")
    save(inputs / "config.json", cfg)
    save(inputs / "imaging_jobs.json", jobs)
    # Retain the source-characterization plan as provenance, never submit it.
    cpu_source = PARENT_DATA / "delivery/local_execution/opengate_ehe_recon_cpu_20261113.py"
    text = cpu_source.read_text(encoding="utf-8")
    assert text.count("5_000_000_000") == 1
    text = text.replace("5_000_000_000", "50_000_000_000").replace("actual 5e9 input required", "actual 5e10 input required")
    (payload/"opengate_ehe_recon_cpu_20261113.py").write_text(text, encoding="utf-8", newline="\n")
    shutil.copy2(PARENT_DATA/"delivery/local_execution/opengate_ehe_common_20261113.py", payload/"local_io_common.py")
    shutil.copy2(HERE/"opengate_ehe_archive_20261113.py", payload/"opengate_ehe_archive_20261113.py")
    shutil.copy2(HERE/"opengate_ehe_nema5e10_archive_20261113.py", payload/"opengate_ehe_nema5e10_archive_20261113.py")
    shutil.copy2(Path(__file__), payload/Path(__file__).name)
    inherited = dict(passed=True, source_study=PARENT_TAG, source_release=parent["release"],
        source_freeze_sha256=sha(PARENT_REPORT/"freeze.json"),
        source_rotation_check=gate["actual_primary_vertex_check"],
        actual_primary_vertex_rotation_verified=True,
        source_rotation_gate_receipt_sha256=sha(PARENT_REPORT/"released_remaining.json"),
        response_kernel_sha256=cfg["reused_response_kernel_sha256"],
        response_source_transport_receipt_sha256=sha(HERE/"reports/NEMA_Body_H60/opengate_ehe_nema5e9_20261113/maty_transport_strict_fetch.json"),
        parent_strict_delivery_sha256=sha(PARENT_REPORT/"strict_delivery.json"),
        inherited_imaging_primaries=0, inherited_imaging_frames=0,
        response_workers_reused=88, response_characterization_primaries=4_400_000_000)
    save(payload/"inherited_science.json", inherited)
    save(REPORT/"inherited_science.json", inherited)
    members = {p.relative_to(payload).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(payload.rglob("*")) if p.is_file()}
    identity = hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest()[:16]
    result = dict(passed=True, study=TAG, release=identity, created_utc=now(), files=members,
        payload_relative="payload", imaging_total=TOTAL, workers=WORKERS,
        concurrency=CONCURRENCY, source_parent_release=parent["release"],
        transport_byte_identical_to_corrected_parent=True,
        projector_byte_identical_to_parent=True, reused_response_workers=88,
        imaging_observations_inherited=False, independent_validation_jobs=False,
        reconstruction_CPU_only_change="accepted input budget 5e9 -> 5e10; numerical loops unchanged",
        reconstruction_CPU_parent_sha256=sha(cpu_source))
    save(payload/"release.json", result)
    save(REPORT/"freeze.json", result)
    return result


def remote_release():
    return CPU + "/releases/" + read(REPORT/"freeze.json")["release"]


def deploy():
    from campaign_20261102_io import upload_files, python_json
    frozen = read(REPORT/"freeze.json")
    payload = DATA/frozen["payload_relative"]
    for n, r in frozen["files"].items():
        assert sha(payload/n) == r["sha256"]
    upload_files("maty", remote_release(), {p.relative_to(payload).as_posix(): p for p in payload.rglob("*") if p.is_file()})
    source = '''import pathlib,json,hashlib,os,sys,subprocess
a=json.loads(sys.argv[1]);p=pathlib.Path(a['release']);f=json.loads((p/'release.json').read_text())
for n,r in f['files'].items():
 q=p/n
 if q.stat().st_size!=r['bytes'] or hashlib.sha256(q.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Changed immutable release '+n)
r=pathlib.Path(a['root']);(r/'control').mkdir(parents=True,exist_ok=True)
receipt=r/'control/output_capacity.json';reserve=r/'control/output_reservation.bin'
if receipt.exists():result=json.loads(receipt.read_text())
else:
 with reserve.open('xb') as h:os.posix_fallocate(h.fileno(),0,2*(1<<30));h.flush();os.fsync(h.fileno())
 s=reserve.stat()
 if s.st_size!=2*(1<<30) or s.st_blocks*512<s.st_size:raise ValueError('Real output allocation was not confirmed')
 result={'passed':True,'release':f['release'],'files':len(f['files']),'actual_output_directory':str(r),'real_reserved_bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'planned_bounded_output_bytes':1<<30,'reserve_fraction_minimum':.2}
 with receipt.open('x') as h:json.dump(result,h);h.flush();os.fsync(h.fileno())
print('CAMPAIGN_JSON '+json.dumps(result))'''
    result = python_json("maty", "python3", source, dict(root=CPU, release=remote_release()), timeout=120, readonly=False)
    save(REPORT/"deployment.json", result)
    return result


def submit(stage, dependency=None):
    from campaign_20261102_io import upload_files, python_json
    from ehe_5e9_workflow import q
    receipt = REPORT/(stage+"_job.json")
    if receipt.exists():
        return read(receipt)
    assert read(REPORT/"deployment.json")["passed"]
    r = remote_release()
    body = "#!/bin/bash\nset -euo pipefail\n" + f"cd {q(r)}\n"
    flags = ["--partition=cnmix", "-N", "1", "-n", "1", "-c", "1", "--time=72:00:00"]
    if stage == "imaging":
        body += 'index="${SLURM_ARRAY_TASK_ID:?}"\nprintf -v index4 "%04d" "$index"\n'
        body += f'mkdir -p {q(CPU+"/resources")}\nallocation={q(CPU+"/resources/imaging_")}"$index4".allocation.txt\n'
        body += 'scontrol show job "${SLURM_JOB_ID:?}" > "$allocation"\n'
        body += (f'bash {q(r+"/cpu_python.sh")} {q(r+"/opengate_ehe_guard_20261113.py")} '
                 f'--record {q(CPU+"/resources/imaging_")}"$index4".json --allocation "$allocation" --seconds 250000 -- '
                 f'{q(RESPONSE_CPU+"/environment/venv/bin/python")} {q(r+"/opengate_ehe_transport_20261113.py")} '
                 f'run --inputs {q(r+"/inputs")} --kind imaging --index "$index" --output {q(CPU+"/workers/imaging/")}"$index4"\n')
        flags += [f"--array=0-{WORKERS-1}%{CONCURRENCY}"]
    elif stage == "collect":
        body += f'bash {q(r+"/cpu_python.sh")} {q(r+"/"+Path(__file__).name)} collect --root {q(CPU)} --inputs {q(r+"/inputs")}\n'
    else:
        raise ValueError("Only actual imaging and collection are submitted")
    if dependency:
        flags += ["--dependency=afterok:"+str(dependency)]
    flags += ["--job-name=og_ehe5e10_"+stage, "--output="+CPU+"/logs/"+stage+"_%A_%a.log"]
    local = DATA/(stage+".sh")
    local.write_text(body, encoding="ascii", newline="\n")
    target = CPU+"/scripts/"+read(REPORT/"freeze.json")["release"]
    upload_files("maty", target, {local.name: local})
    plan = dict(stage=stage, root=CPU, release=read(REPORT/"freeze.json")["release"], flags=flags,
                script=target+"/"+local.name, script_sha256=sha(local), created_utc=now())
    save(REPORT/(stage+"_submission_intent.json"), plan)
    source = '''import pathlib,json,os,sys,fcntl,hashlib,subprocess,time
a=json.loads(sys.argv[1]);r=pathlib.Path(a['root']);(r/'logs').mkdir(exist_ok=True)
lock=(r/'control'/('submit_'+a['stage']+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=r/'control'/(a['stage']+'_job.json');intent=r/'control'/(a['stage']+'_intent.json')
def save(p,d):
 t=p.with_suffix('.writing')
 with t.open('w') as h:json.dump(d,h);h.flush();os.fsync(h.fileno())
 t.replace(p)
if p.exists():
 result=json.loads(p.read_text())
 if result['release']!=a['release'] or result['script_sha256']!=a['script_sha256']:raise ValueError('Different registered submit')
else:
 if intent.exists():raise ValueError('Unresolved scheduler intent; inspect Slurm without blind submit')
 if hashlib.sha256(pathlib.Path(a['script']).read_bytes()).hexdigest()!=a['script_sha256']:raise ValueError('Changed batch file')
 save(intent,a)
 if a['stage']=='imaging':
  capacity=json.loads((r/'control/output_capacity.json').read_text());reserve=r/'control/output_reservation.bin'
  if not capacity['passed'] or reserve.stat().st_size!=capacity['real_reserved_bytes']:raise ValueError('Real allocation gate absent')
  reserve.unlink();save(r/'control/output_capacity_released.json',dict(capacity,released_epoch=time.time()))
 x=subprocess.run(['sbatch','--parsable']+a['flags']+[a['script']],capture_output=True,text=True,timeout=60)
 if x.returncode:raise RuntimeError(x.stderr)
 number=x.stdout.strip().split(';')[0]
 if not number.isdigit():raise ValueError('Uncertain submit result')
 result=dict(a,job=int(number),submitted_epoch=time.time());save(p,result)
print('CAMPAIGN_JSON '+json.dumps(result))'''
    result = python_json("maty", "python3", source, plan, timeout=100, readonly=False)
    save(receipt, result)
    return result


def collect(root, inputs):
    root, inputs = Path(root), Path(inputs)
    if (root/"collected/complete.json").exists():
        raise FileExistsError("Successful 5e10 collection exists")
    cfg, jobs = read(inputs/"config.json"), read(inputs/"imaging_jobs.json")
    assert cfg["imaging_primary_gamma_total"] == TOTAL and len(jobs) == WORKERS
    counts = np.zeros((2,20,68,34), dtype=np.int64)
    tags = np.zeros((2,2,20,68,34), dtype=np.int64)
    emitted = np.zeros((20,2), dtype=np.int64)
    records = {}
    for job in jobs:
        folder = root/"workers/imaging"/f"{job['index']:04d}"
        meta = read(folder/"complete.json")
        if not meta["passed"] or meta["job"] != job or meta["actual_primary_events"] != job["primaries"]:
            raise ValueError("Actual worker/primary identity differs")
        for n, r in meta["files"].items():
            if (folder/n).stat().st_size != r["bytes"] or sha(folder/n) != r["sha256"]:
                raise ValueError("Changed worker output")
        with np.load(folder/"counts.npz") as z:
            c = z["counts"]
        if c.shape != (2,68,34) or c.dtype != np.int64 or np.any(c<0):
            raise ValueError("Invalid crystal window counts")
        ei = (218,440).index(job["energy_keV"])
        counts[:,job["view"]] += c
        tags[:,ei,job["view"]] += c
        emitted[job["view"],ei] += job["primaries"]
        records[str(job["index"])] = dict(sha256=sha(folder/"complete.json"), primaries=job["primaries"], seed=job["seed"])
    np.testing.assert_array_equal(tags.sum(1), counts)
    np.testing.assert_array_equal(emitted, np.array(cfg["emitted_primaries_by_view_energy"]))
    assert emitted.sum() == TOTAL
    inherited = read(inputs.parent/"inherited_science.json")
    kernel = Path(RESPONSE_CPU)/"collected/response_kernels.npz"
    if sha(kernel) != inherited["response_kernel_sha256"]:
        raise ValueError("Changed accepted response kernel")
    dest = root/"collected"
    dest.mkdir()
    save_array(dest/"observations.npz", counts=counts, tagged=tags, emitted=emitted)
    shutil.copy2(kernel, dest/"response_kernels.npz")
    response_counts = read(Path(RESPONSE_CPU)/"collected/complete.json")["response_counts_by_job"]
    save(dest/"complete.json", dict(passed=True, created_utc=now(), actual_imaging_primaries=int(emitted.sum()),
        actual_primary_by_view_energy=emitted.tolist(), counts_by_window=counts.sum((1,2,3)).tolist(),
        workers=records, config_sha256=sha(inputs/"config.json"), response_counts_by_job=response_counts,
        inherited_science_sha256=sha(inputs.parent/"inherited_science.json"), inherited_imaging_primaries=0,
        actual_primary_vertex_rotation_verified=True,
        files={n:dict(bytes=(dest/n).stat().st_size, sha256=sha(dest/n)) for n in ("observations.npz","response_kernels.npz")},
        physical_calibration_passed=False))
    if not (root/"inputs").exists():
        os.symlink(inputs, root/"inputs")
    print("ACTUAL_5E10_COLLECTION_COMPLETE", counts.sum((1,2,3)).tolist(), flush=True)


def seal(root, kind):
    import re
    import opengate_ehe_archive_20261113 as archive
    root = Path(root)
    authority = root/"control/transport_delivery_authority.json"
    if authority.exists():
        return read(authority)
    complete = read(root/"collected/complete.json")
    if not complete["passed"] or complete["actual_imaging_primaries"] != TOTAL:
        raise ValueError("Full actual 50B collection required")
    freeze = read(root/"releases"/read(root/"control/imaging_job.json")["release"]/"release.json")
    source = root/"releases"/freeze["release"]
    inherited = read(source/"inherited_science.json")
    if sha(root/"collected/response_kernels.npz") != inherited["response_kernel_sha256"]:
        raise ValueError("Changed reused response")
    acct = dict(imaging=archive.accounting(root,"imaging",WORKERS), collect=archive.accounting(root,"collect"))
    resources, peak = {}, 0.
    for index in range(WORKERS):
        p = root/"resources"/f"imaging_{index:04d}.json"
        record = read(p)
        if record["status"] != "COMPLETED" or record["rss_fraction"] > .8:
            raise ValueError("Worker actual resource/exit failed")
        snapshot = (root/"resources"/f"imaging_{index:04d}.allocation.txt").read_text()
        if f"ArrayJobId={acct['imaging']['job']}" not in snapshot or not re.search(r'\bArrayTaskId='+str(index)+r'\b',snapshot):
            raise ValueError("Actual allocation identity differs")
        resources[str(index)] = dict(sha256=sha(p), job=record["job"])
        peak = max(peak,record["rss_fraction"])
    result = dict(passed=True, actual_primary_gamma_total=TOTAL, new_imaging_primaries=TOTAL,
        inherited_imaging_primaries=0, new_workers=WORKERS, inherited_response_workers=88,
        actual_primary_vertex_rotation_verified=True, source_rotation_check=inherited["source_rotation_check"],
        inherited_science=inherited, inherited_science_sha256=sha(source/"inherited_science.json"),
        accounting=acct, resources=resources, max_worker_rss_fraction=peak,
        collected_sha256=sha(root/"collected/complete.json"), physical_calibration_passed=False)
    save(authority,result)
    return result


def bundle(root, kind):
    import opengate_ehe_archive_20261113 as archive
    archive.seal = seal
    return archive.bundle(Path(root), kind)


def fetch(kind):
    from campaign_20261102_io import execute
    from ehe_5e9_workflow import q
    import opengate_ehe_delivery_20261113 as delivery
    receipt = REPORT/("maty_"+kind+"_strict_fetch.json")
    if receipt.exists():
        return read(receipt)
    r = remote_release()
    raw = execute("maty", "python3 "+q(r+"/opengate_ehe_nema5e10_archive_20261113.py")+" --root "+q(CPU)+" --kind "+kind,timeout=1800)
    record = json.loads(raw.rsplit("CAMPAIGN_JSON ",1)[1])
    save(REPORT/("maty_"+kind+"_bundle.json"),record)
    delivery.DATA, delivery.REPORT = DATA, REPORT
    archive = delivery.download("maty",record)
    dest = DATA/("delivery" if kind=="bridge" else "transport_delivery")
    manifest = delivery.extract(archive,record,dest)
    result = dict(passed=True, strict_fetch_passed=True, all_files_SHA_verified=True, host="maty",kind=kind,
        created_utc=now(), archive=record,destination=str(dest),files=manifest["files"],authority=manifest["authority"])
    save(receipt,result)
    print("ACTUAL_5E10_STRICT_FETCH",kind,len(result["files"]),flush=True)
    return result


def cpu_reconstruct():
    import psutil
    import subprocess
    if (REPORT/"local_cpu_exit.json").exists():
        raise FileExistsError("Never repeat a registered scientific execution")
    bridge = read(REPORT/"maty_bridge_strict_fetch.json")
    assert bridge["passed"] and bridge["authority"]["actual_primary_gamma_total"] == TOTAL
    root, payload = DATA/"delivery", DATA/"payload"
    for n,r in bridge["files"].items():
        if (root/n).stat().st_size!=r["bytes"] or sha(root/n)!=r["sha256"]:
            raise ValueError("Changed accepted 5e10 input")
    if (root/"reconstruction").exists():
        raise FileExistsError("Refuse repeated or parallel reconstruction")
    release = DATA/"local_cpu_release"
    release.mkdir()
    for src,dest in (("local_io_common.py","opengate_ehe_common_20261113.py"),
                     ("opengate_ehe_projector_20261113.py","opengate_ehe_projector_20261113.py"),
                     ("opengate_ehe_recon_cpu_20261113.py","opengate_ehe_recon_cpu_20261113.py")):
        shutil.copy2(payload/src,release/dest)
    frozen = dict(passed=True, scientific_base_release="7f1354f52bc68cfd", source_parent_release="f7231adc45ed8607",
        corrected_transport_release=read(REPORT/"freeze.json")["release"],
        CPU_parent_execution_sha256=read(REPORT/"freeze.json")["reconstruction_CPU_parent_sha256"],
        CPU_source_release=sha(release/"opengate_ehe_recon_cpu_20261113.py")[:16],
        execution_change="input budget gate only; MLEM/response/transpose/S/checkpoint loops unchanged",
        initial_density=1., iterations=200, channels=[440,218], threads=8, image_filter=None,
        actual_imaging_primaries=TOTAL, actual_resume_used=False,
        files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in release.iterdir()})
    save(release/"release.json",frozen)
    save(REPORT/"reconstruction_freeze.json",frozen)
    execution = root/"local_execution"
    execution.mkdir()
    for p in release.iterdir():
        shutil.copy2(p,execution/p.name)
    # Retain the actual-process-family guard used for the completed5e9 run.
    cap=int(min(8*(1<<30),.8*psutil.virtual_memory().available))
    if cap<4*(1<<30) or shutil.disk_usage(root).free<4*(1<<30)/.8+16*(1<<30):
        raise MemoryError("Actual local output/RSS headroom insufficient")
    env=dict(os.environ,OMP_NUM_THREADS="8",MKL_NUM_THREADS="8",OPENBLAS_NUM_THREADS="8",OPENGATE_LOCAL_HOST_CAP_BYTES=str(cap))
    command=[str(LOCAL_ENV),"-u",str(release/"opengate_ehe_recon_cpu_20261113.py"),"--root",str(root),"--inputs",str(root/"inputs")]
    start=time.monotonic();peak=0;last=0
    with (execution/"stdout.log").open("xb") as out,(execution/"stderr.log").open("xb") as err:
        child=subprocess.Popen(command,env=env,stdout=out,stderr=err);proc=psutil.Process(child.pid)
        reg=dict(pid=child.pid,create_time=proc.create_time(),cmdline=proc.cmdline(),controller_pid=os.getpid(),
                 status="RUNNING",execution_kind="LOCAL_CPU",host_bound_bytes=cap,started_utc=now())
        save(REPORT/"local_cpu_registration.json",reg)
        try:
            while child.poll() is None:
                rss=0
                for p in [proc]+proc.children(recursive=True):
                    try:
                        m=p.memory_info();rss+=max(m.rss,getattr(m,"peak_wset",0))
                    except psutil.NoSuchProcess:
                        pass
                peak=max(peak,rss)
                if peak>.8*cap or psutil.virtual_memory().available<.2*cap:
                    raise MemoryError("Process family actual20percent memory margin violated")
                if time.monotonic()-start>86400:
                    raise TimeoutError("24h finite execution cap")
                if time.monotonic()-last>30:
                    print("ACTUAL_5E10_CPU_RUNNING",round(time.monotonic()-start,1),peak,flush=True);last=time.monotonic()
                time.sleep(.5)
            if child.returncode or not read(root/"reconstruction_complete.json")["passed"]:
                raise RuntimeError("Scientific reconstruction exited unsuccessfully")
        except BaseException:
            if child.poll() is None:
                family=proc.children(recursive=True)+[proc]
                for p in family:
                    try:p.terminate()
                    except psutil.NoSuchProcess:pass
                _,alive=psutil.wait_procs(family,timeout=10)
                for p in alive:
                    try:p.kill()
                    except psutil.NoSuchProcess:pass
                child.wait(timeout=20)
            save(REPORT/"local_cpu_exit.json",dict(reg,status="FAILED",returncode=child.returncode,ended_utc=now(),peak_family_rss_bytes=peak))
            raise
    result=dict(reg,status="COMPLETED",passed=True,returncode=0,ended_utc=now(),elapsed_seconds=time.monotonic()-start,peak_family_rss_bytes=peak)
    save(REPORT/"local_cpu_exit.json",result);save(execution/"exit.json",result)
    print("ACTUAL_5E10_CPU_COMPLETE",flush=True)


def finalize():
    transport=read(REPORT/"maty_transport_strict_fetch.json")
    exit=read(REPORT/"local_cpu_exit.json")
    assert transport["passed"] and exit["passed"] and exit["returncode"]==0
    root=DATA/"delivery";qa={};peak=0
    for e in ("440","218"):
        folder=root/"reconstruction"/e;complete=read(folder/"complete.json")
        assert complete["passed"] and complete["checkpoints"]==200
        for n,r in complete["files"].items():
            assert (folder/n).stat().st_size==r["bytes"] and sha(folder/n)==r["sha256"]
        history=read(folder/"history.json")
        assert [r["iteration"] for r in history]==list(range(1,201))
        ll=np.asarray([r["log_likelihood_before_update"] for r in history]);minimum=float(np.diff(ll).min())
        assert minimum>=-1e-8
        with np.load(folder/"sensitivity.npz") as z:fraction=z["fraction_xyz"]
        for i in range(1,201):
            with np.load(folder/f"iteration_{i:04d}.npz") as z:a=z["density_xyz"]
            assert np.isfinite(a).all() and np.all(a>=0) and np.all(a[fraction==0]==0)
        qa[e]=dict(passed=True,iterations=200,log_likelihood_minimum_increment=minimum)
        peak=max(peak,max(r["rss_bytes"] for r in history))
    records={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob("*") if p.is_file()}
    assert not any(n.endswith(".writing") for n in records)
    result=dict(passed=True,strict_fetch_passed=True,all_files_SHA_verified=True,checkpoints=400,files=records,
        delivery_root=str(root),actual_imaging_primaries=TOTAL,actual_primary_vertex_rotation_verified=True,
        imaging_observations_inherited=False,execution_backend="LOCAL_CPU_8_THREADS",actual_resume_used=False,
        actual_scientific_peak_rss_bytes=peak,numerical_QA=qa,
        transport_receipt_sha256=sha(REPORT/"maty_transport_strict_fetch.json"),local_exit_sha256=sha(REPORT/"local_cpu_exit.json"),
        inherited_science_sha256=sha(REPORT/"inherited_science.json"),physical_calibration_passed=False,visual_QA_passed=False,completed_utc=now())
    save(REPORT/"strict_delivery.json",result)
    print("ACTUAL_5E10_STRICT_DELIVERY",len(records),flush=True)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","deploy","submit_imaging","submit_collect","collect","bundle","fetch_bridge","fetch_transport","cpu_reconstruct","finalize"])
    p.add_argument("--root",type=Path);p.add_argument("--inputs",type=Path);p.add_argument("--kind",choices=["bridge","transport"]);p.add_argument("--dependency")
    a=p.parse_args()
    if a.action=="prepare":result=prepare()
    elif a.action=="deploy":result=deploy()
    elif a.action.startswith("submit_"):result=submit(a.action[7:],a.dependency)
    elif a.action=="collect":collect(a.root,a.inputs);result=None
    elif a.action=="bundle":result=bundle(a.root,a.kind)
    elif a.action.startswith("fetch_"):result=fetch(a.action[6:])
    elif a.action=="cpu_reconstruct":cpu_reconstruct();result=None
    else:result=finalize()
    if result is not None:print("CAMPAIGN_JSON "+json.dumps(result))
