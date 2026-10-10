"""Independent JSCC actual-5e10, three-output experiment identities."""
import hashlib, json, os, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = 'jscc_geant4_5e10_10000'
DATA = HERE / 'generated' / STUDY
REPORT = HERE / 'reports/NEMA_Body_H60' / STUDY
CPU_PROJECT = '/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928'
CPU_BASE = CPU_PROJECT + '/experiments/ELLIPSE500x300_H120/generated/' + STUDY
GPU_PROJECT = '/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120'
GPU_BASE = GPU_PROJECT + '/generated/' + STUDY
GPU_PYTHON = '/data/home/scxi717/.conda/envs/torch/bin/python'
TOTAL, WORKERS, PER_WORKER, PER_VIEW = 50_000_000_000, 1000, 50_000_000, 50
SEED_BASE, PILOT_SEED = 35100101, 35100001
CHANNELS = ('440_SinglePhoton', '218_SinglePhoton_CrossTalkCorrected', '440_ComptonOnly')
PHASE_CHANNELS = dict(zip(('440_single', '218_corrected', '440_compton'), ((c,) for c in CHANNELS)))
BINARY_SHA = '9de76827814a6ce6f1fa273cbdfc8b353f9fc98bf489507ce5ff94d400fd91e6'
CRYSTAL_SHA = '4f36ae7b95cfbac647885538bd64d09c991162ae1aa4c5eb46f11e65ab1292fb'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20), b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf8'))

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing')
    with tmp.open('wb') as f:
        f.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    if os.name=='posix':
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

def hashes(root):
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(Path(root).rglob('*')) if p.is_file()}

def verify_files(root, files):
    for name,sha in files.items():
        if digest(Path(root)/name)!=sha: raise ValueError('Immutable file changed: '+name)

def registry(path):
    r=read(path)
    if (r['total_primary_photons'],r['workers_per_view'],len(r['jobs']))!=(TOTAL,PER_VIEW,WORKERS):
        raise ValueError('New actual dose/worker registry differs')
    for i,j in enumerate(r['jobs']):
        if tuple(j[k] for k in ('index','view','worker','seed','photons'))!=(i,i//50+1,i%50,SEED_BASE+i,PER_WORKER):
            raise ValueError('Independent worker identity differs')
    return r

def execution_policy(mode):
    if mode not in ('validation','formal'):raise ValueError('Explicit execution mode required')
    return (10,10) if mode=='validation' else (10000,50)

def host_allocated_bytes(text,nodes=8):
    if int(re.search(r'\bNumNodes=(\d+)',text)[1])!=nodes:raise ValueError('Actual node count differs')
    tres=re.search(r'\bAllocTRES=([^\s]+)',text)[1]
    m=re.search(r'(?:^|,)mem=([0-9.]+)([KMGT])(?:,|$)',tres)
    if not m:raise ValueError('Actual allocated memory is missing')
    return int(float(m[1])*1024**('KMGT'.index(m[2])+1)/nodes)

def validate_collection(c):
    if (not c['passed'] or c['study']!=STUDY or c['total_primary_photons']!=TOTAL or
        c['worker_indices']!=list(range(WORKERS)) or c['seeds']!=list(range(SEED_BASE,SEED_BASE+WORKERS)) or
        c['views']!=list(range(1,21)) or sum(c['primary_counts'])!=TOTAL or c['primary_counts'][2]!=0 or
        c['binary_sha256']!=BINARY_SHA or c['crystal_sha256']!=CRYSTAL_SHA):
        raise ValueError('This experiment actual transport identity does not close')
    return c
