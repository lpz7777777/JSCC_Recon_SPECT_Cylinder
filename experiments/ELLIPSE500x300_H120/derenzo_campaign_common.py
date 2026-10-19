"""Identities for four independent, monoenergetic Derenzo acquisitions."""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDY = 'derenzo_equal_regions_mono_5e9_20261016'
DATA = HERE/'generated'/STUDY
REPORT = HERE/'reports/Derenzo'/STUDY
DESIGN = ROOT/'Analysis_Plots/research/derenzo_218_440_design_20261015_equal_regions'
CPU_PROJECT = '/WORK/maty_work/lpz/20250307_JSCCGC_32x64_4layer_SPECT_225Ac/JSCC_SPECT/ELLIPSE500x300_H120_20260928/experiments/ELLIPSE500x300_H120'
GPU_PROJECT = '/data/run01/scxi717/lpz/20250307_JSCCGC_32x32x4_Shield_DiffEne_SPECT_PolarCoor/experiments/ELLIPSE500x300_H120'
CPU_BASE = CPU_PROJECT+'/generated/'+STUDY
GPU_BASE = GPU_PROJECT+'/generated/'+STUDY
GPU_PYTHON = '/data/home/scxi717/.conda/envs/torch/bin/python'
TOTAL, VIEWS, WORKERS, PER_VIEW, PER_WORKER = 5_000_000_000, 20, 200, 10, 25_000_000
DATASETS = ('JSCC_440', 'JSCC_218', 'EHE_440', 'EHE_218')
SEED_BASE = dict(zip(DATASETS, (46100101, 46200101, 46300101, 46400101)))
PILOT_PHOTONS = 100_000
CHANNELS = {'JSCC_440':('440_SinglePhoton','440_SinglePlusCompton'),
            'JSCC_218':('218_SinglePhoton',),'EHE_440':('440_SinglePhoton',),'EHE_218':('218_SinglePhoton',)}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.writing')
    with tmp.open('wb') as f:
        f.write((json.dumps(value,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def hashes(root):
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(Path(root).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}


def verify_files(root,files):
    for name,sha in files.items():
        if digest(Path(root)/name)!=sha:raise ValueError('Immutable file differs: '+name)


def dataset_info(name):
    if name not in DATASETS:raise ValueError('Unknown independent dataset')
    system,energy=name.split('_');energy=int(energy)
    return dict(dataset=name,system=system,energy_keV=energy,bins=10496 if system=='JSCC' else 2312,
                channels=list(CHANNELS[name]),iterations=10000 if system=='JSCC' else 200,
                save_step=50 if system=='JSCC' else 1,seed_base=SEED_BASE[name])


def macro_text(design,view,photons):
    if view not in range(1,21) or photons<=0 or photons>2_000_000_000:raise ValueError('Invalid run partition')
    energy=design['energy_keV']
    lines=['# Exact analytic Derenzo cylinders; vacuum object region; one mono gamma/event; full 4pi',
           '# Equal emission density within all rods: categorical weights are cylinder volumes.',
           '/xcat/clear','/ellipse/clear','/ellipse/centerY -345',f'/ellipse/angle {(view-1)*18}']
    for r in design['rods']:
        values=[energy,r['x_mm'],r['y_mm'],r['z_mm'],r['diameter_mm']/2,
                (r['z_max_mm']-r['z_min_mm'])/2,r['volume_mm3']]
        lines.append('/ellipse/addRod '+' '.join(format(x,'.17g') for x in values))
    lines.append('/run/beamOn '+str(photons))
    return '\n'.join(lines)+'\n'


def validate_registry(registry,dataset):
    info=dataset_info(dataset);jobs=registry['jobs']
    if (registry['study']!=STUDY or registry['dataset']!=dataset or registry['total_primary_photons']!=TOTAL
            or len(jobs)!=WORKERS or registry['views']!=VIEWS or registry['workers_per_view']!=PER_VIEW):
        raise ValueError('Exact independent monoenergy worker registry required')
    for i,j in enumerate(jobs):
        if tuple(j[k] for k in ('index','view','worker','photons','seed','energy_keV'))!=(
                i,i//PER_VIEW+1,i%PER_VIEW,PER_WORKER,info['seed_base']+i,info['energy_keV']):
            raise ValueError('Worker dose/view/energy/seed identity differs')
    return registry


def validate_collection(c,dataset):
    info=dataset_info(dataset);expected=[TOTAL,0,0] if info['energy_keV']==218 else [0,TOTAL,0]
    if (c.get('passed') is not True or c['study']!=STUDY or c['dataset']!=dataset
            or c['total_primary_photons']!=TOTAL or c['primary_counts']!=expected
            or c['bins']!=info['bins'] or c['workers']!=WORKERS or c['views']!=20
            or c['data_kind']!='actual_Geant4_transport' or c['source_solid_angle']!='4pi'):
        raise ValueError('Complete actual 5e9 monoenergetic transport required')
    return c
