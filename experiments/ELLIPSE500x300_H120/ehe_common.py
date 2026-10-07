"""Independent EHE experiment identity, atomic evidence and immutable inputs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = 'ehe_spect_5e9_200'
DATA = HERE / 'generated' / STUDY
REPORT = HERE / 'reports/NEMA_Body_H60' / STUDY
CHANNELS = ('440_SinglePhoton', '218_SinglePhoton_CrossTalkCorrected', '440SinglePlus218Single')
RESPONSES = ('A218', 'A440', 'C440to218')
MATY_HOST = 'maty@192.168.11.1'
MATY_BASE = '/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120/generated/' + STUDY
GPU_BASE = '/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120/generated/' + STUDY
GPU_PYTHON = '/data/home/scxi717/.conda/envs/torch/bin/python'
ENGINE = ROOT / 'Auxiliary_Studies/GPU-Based-System-Matrix-Calculation-for-SPECT-PET-main'
TRUTH = HERE / 'generated/NEMA_Body_H60/truth_3mm.npz'
TRUTH_META = HERE / 'reports/NEMA_Body_H60/manifest.json'
GEOMETRY = HERE / 'generated/compton_energy_probability_v5_5e9_full10000/formal_payload/whole_geometry.npz'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + '.writing')
    with staging.open('wb') as f:
        f.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
        f.flush(); os.fsync(f.fileno())
    os.replace(staging, path)
    if os.name != 'nt':
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def hashes(folder):
    return {p.relative_to(folder).as_posix(): digest(p) for p in sorted(Path(folder).rglob('*')) if p.is_file()}

def verify_files(folder, expected):
    for name, sha in expected.items():
        p = Path(folder) / name
        if not p.is_file() or digest(p) != sha: raise ValueError('Missing/changed immutable file: ' + name)

def allocated_bytes(text, nodes=1):
    matches = re.findall(r'AllocTRES=([^\s]+)', text)
    if len(matches) != 1: raise ValueError('Actual scontrol AllocTRES required')
    m = re.search(r'(?:^|,)mem=([0-9.]+)([KMGT]?)', matches[0])
    if not m: raise ValueError('Actual allocated memory absent')
    unit = m[2] or 'M'
    return int(float(m[1]) * 1024 ** ('KMGT'.index(unit)+1) / nodes)

def policy(mode, iterations, save_step):
    if (mode, iterations, save_step) not in (('validation',10,10), ('formal',200,10)):
        raise ValueError('EHE permits only validation10/save10 or formal200/save10')

def physical_gate(predicted, observed, standard_error, adequate):
    import numpy as np
    p, o, s = map(lambda x: np.asarray(x, float), (predicted, observed, standard_error))
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(~np.isfinite(o)) or np.any(o < 0) or np.any(~np.isfinite(s)) or np.any(s < 0):
        raise ValueError('Invalid physical audit arrays')
    delta = abs(p-o)
    relative = delta / np.maximum(o, 1)
    hold = np.asarray(adequate, bool) & (relative > .10) & (delta > 3*s)
    return relative, hold

def array_write(path, array):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing')
    with tmp.open('wb') as f: array.tofile(f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    if os.name!='nt':
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

def allocation(unlimited_transport=False):
    import subprocess
    if not os.environ.get('SLURM_JOB_ID'): raise ValueError('Actual Slurm allocation required')
    raw=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID']],text=True)
    try:
        denominator=allocated_bytes(raw,int(os.environ.get('SLURM_JOB_NUM_NODES','1')))
        if denominator<=0:raise ValueError('Zero memory allocation')
        kind='actual Slurm AllocTRES'
    except ValueError:
        if not unlimited_transport:raise
        # Maty CPU partitions use UNLIMITED and omit a real memory TRES.
        # This is an operational CPU transport guard, never an imaging certificate.
        info=Path('/proc/meminfo').read_text()
        denominator=int(re.search(r'MemAvailable:\s+(\d+)',info)[1])*1024
        kind='CPU transport physical MemAvailable at start; no Slurm memory allocation'
    return {'job':os.environ['SLURM_JOB_ID'],'scontrol':raw,'host_allocated_bytes':denominator,'denominator_kind':kind,
            'imaging_allocation_certificate':kind=='actual Slurm AllocTRES'}

def resources(alloc):
    import resource
    import subprocess
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    value=dict(rss_peak_bytes=rss,host_allocated_bytes=alloc['host_allocated_bytes'],rss_fraction=rss/alloc['host_allocated_bytes'])
    try:
        import torch
        if torch.cuda.is_initialized():
            total=torch.cuda.get_device_properties(0).total_memory
            value.update(gpu_reserved_peak_bytes=torch.cuda.max_memory_reserved(0),gpu_total_bytes=total,
                         gpu_reserved_fraction=torch.cuda.max_memory_reserved(0)/total)
    except ImportError:pass
    if value['rss_fraction']>.8 or value.get('gpu_reserved_fraction',0)>.8:raise MemoryError('20% actual allocation reserve violated')
    return value

def bounded_process(argv,cwd,seconds,log,alloc,env=None,gpu=False):
    import subprocess,time
    peak=0;gpu_peak=0;gpu_total=0
    started=time.monotonic()
    with Path(log).open('wb') as f:
        process=subprocess.Popen(argv,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,env=env)
        try:
            while process.poll() is None:
                try:
                    status=Path(f'/proc/{process.pid}/status').read_text()
                    peak=max(peak,int(re.search(r'VmHWM:\s+(\d+)',status)[1])*1024)
                except (FileNotFoundError,TypeError):pass
                if gpu:
                    # Slurm CUDA device 0 can be a different physical nvidia-smi index.
                    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory,gpu_uuid','--format=csv,noheader,nounits'],text=True)
                    for row in rows.splitlines():
                        fields=[x.strip() for x in row.split(',')]
                        if len(fields)==3 and fields[0]==str(process.pid):
                            gpu_peak=max(gpu_peak,int(fields[1])*1024**2)
                            total=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits','-i',fields[2]],text=True)
                            gpu_total=int(total.strip())*1024**2
                if peak>.8*alloc['host_allocated_bytes'] or (gpu_total and gpu_peak>.8*gpu_total):raise MemoryError('Resource margin exceeded')
                if time.monotonic()-started>seconds:raise TimeoutError('Measured bounded phase limit exceeded')
                time.sleep(.5)
            if process.returncode:raise RuntimeError(f'Process exit {process.returncode}; inspect {log}')
        except BaseException:
            process.terminate()
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise
    return dict(elapsed_seconds=time.monotonic()-started,rss_peak_bytes=peak,
                host_allocated_bytes=alloc['host_allocated_bytes'],rss_fraction=peak/alloc['host_allocated_bytes'],
                gpu_used_peak_bytes=gpu_peak,gpu_total_bytes=gpu_total,gpu_used_fraction=gpu_peak/gpu_total if gpu_total else 0)
