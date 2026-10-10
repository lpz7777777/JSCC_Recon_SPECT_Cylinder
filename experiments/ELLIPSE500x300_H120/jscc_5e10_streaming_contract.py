"""Independent three-route/full-input/8x3 streaming execution contract and checkpoints."""
import json, os, re
from pathlib import Path
import numpy as np
import torch
from jscc_5e10_common import *
from run_reconstruction import collect_response

def load_contract(path,mode,iterations,save_step):
    path=Path(path);cfg=read(path);root=path.parent
    if (iterations,save_step)!=execution_policy(mode):raise ValueError('Only full-input validation10 or formal10000/save50')
    if (cfg['study']!=STUDY or cfg['channels']!=list(CHANNELS) or cfg['nodes']!=8 or cfg['gpus_per_node']!=3 or
        cfg['world_size']!=24 or cfg['model']!='continuous_energy' or cfg['iterations']!=10000 or cfg['save_step']!=50 or
        cfg['total_primary_photons']!=TOTAL or cfg['event_policy']!='legacy' or cfg['regularization']!='none' or
        cfg['initial_density']!=1 or cfg['cross_prediction_source']!='440_SinglePhoton_final' or cfg['joint_solver_enabled']):
        raise ValueError('Independent three-route/dose/topology contract differs')
    if cfg.get('response_storage_policy') != 'lossless_zlib_float32_32event_local_disk_v1':
        raise ValueError('Explicit complete streaming response policy required')
    verify_files(root,cfg['files'])
    baseline=read(root/'baseline_contract.json')
    for key in ('factor_manifest_sha256','factor_payload_sha256','whole_geometry_sha256','calibration_release'):
        if cfg[key]!=baseline[key]:raise ValueError('Previously accepted operator identity differs: '+key)
    for n in ('compton_energy_probability_v5.py','compton_event_response.py','torch_active_operator.py','continuous_energy_Sensi_full','transfer_training_summary.json'):
        if cfg['files'][n]!=baseline['files'][n]:raise ValueError('Previously accepted science changed: '+n)
    if sum(cfg['events_per_view'])!=cfg['accepted_events'] or len(cfg['events_per_view'])!=20:
        raise ValueError('New complete event selection does not close')
    validate_collection(read(root/'transport_collection.json'))
    return cfg

def write_checkpoint(output,phase,iteration,frames,geometry,contract_sha,mode):
    limit,step=execution_policy(mode)
    if phase not in PHASE_CHANNELS or set(frames)!=set(PHASE_CHANNELS[phase]) or not 0<iteration<=limit or iteration%step:
        raise ValueError('Three-route durable checkpoint policy differs')
    parent=Path(output)/('checkpoints_'+phase);parent.mkdir(exist_ok=True)
    target=parent/f'checkpoint_{iteration:06d}';temp=parent/f'.checkpoint_{iteration:06d}.partial'
    temp.mkdir(exist_ok=False)
    if target.exists():raise FileExistsError('Never overwrite existing checkpoints')
    outputs={}
    for name,frame in frames.items():
        if frame.numel()!=78920 or not bool(torch.isfinite(frame).all()) or bool((frame<0).any()):raise ValueError('Invalid checkpoint image')
        collect_response(temp,name,frame,None,geometry,0);outputs[name]={}
        for kind in ('active','full'):
            p=temp/f'Image_{name}_{kind}.float32'
            with p.open('r+b') as f:os.fsync(f.fileno())
            outputs[name][kind]=digest(p)
    write(temp/'checkpoint_manifest.json',dict(study=STUDY,mode=mode,phase=phase,iteration=iteration,
        contract_sha256=contract_sha,outputs=outputs))
    temp.rename(target)
    if os.name=='posix':
        fd=os.open(parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)

def verify_checkpoints(result,histories,active,contract_sha,mode):
    limit,step=execution_policy(mode);records=[];inactive=np.ones(132040,bool);inactive[active]=False
    for phase,channels in PHASE_CHANNELS.items():
        p=Path(result)/('checkpoints_'+phase);expected={f'checkpoint_{i:06d}' for i in range(step,limit+1,step)}
        if {x.name for x in p.glob('checkpoint_*') if x.is_dir()}!=expected or list(p.glob('*.partial')):
            raise ValueError('Missing/extra/unfinished checkpoint')
        for frame,i in enumerate(range(step,limit+1,step)):
            folder=p/f'checkpoint_{i:06d}';s=read(folder/'checkpoint_manifest.json')
            if tuple(s[k] for k in ('study','mode','phase','iteration','contract_sha256'))!=(STUDY,mode,phase,i,contract_sha) or set(s['outputs'])!=set(channels):
                raise ValueError('Checkpoint identity differs')
            for channel in channels:
                values={}
                for kind,n in (('active',78920),('full',132040)):
                    f=folder/f'Image_{channel}_{kind}.float32'
                    if f.stat().st_size!=n*4 or digest(f)!=s['outputs'][channel][kind]:raise ValueError('Checkpoint size/SHA differs')
                    values[kind]=np.fromfile(f,'<f4')
                if (not np.array_equal(values['active'],histories[channel][frame]) or
                    not np.array_equal(values['full'][active],values['active']) or np.any(values['full'][inactive]!=0)):
                    raise ValueError('Checkpoint differs from complete history/support')
            records.append(dict(phase=phase,iteration=i,manifest_sha256=digest(folder/'checkpoint_manifest.json')))
    return records

def verify_topology(resources,allocation):
    if len(resources)!=24 or sorted(r['rank'] for r in resources)!=list(range(24)):raise ValueError('All 24 rank identities required')
    allocation_text=Path(allocation).read_text()
    tres=re.search(r'\bAllocTRES=([^\s]+)',allocation_text)[1]
    if re.search(r'(?:^|,)gres/gpu=24(?:,|$)',tres) is None:raise ValueError('Actual allocation must contain 24 GPUs')
    memory=host_allocated_bytes(allocation_text,8);nodes={r['node'] for r in resources}
    if len(nodes)!=8:raise ValueError('Eight actual nodes required')
    for node in nodes:
        group=[r for r in resources if r['node']==node]
        if sorted(r['local_rank'] for r in group)!=list(range(3)) or len({r['gpu_uuid'] for r in group})!=3:
            raise ValueError('Three different actual GPUs/local ranks per node required')
        if sum(r['host_peak_rss_bytes'] for r in group)>.8*memory:
            raise ValueError('Conservative aggregate node RSS leaves less than 20% headroom')
        for r in group:
            if (r['host_allocated_bytes_node']!=memory or r['host_peak_rss_bytes']<=0 or r['total_device_bytes']<=0 or not r['gpu_uuid'] or
                r['peak_reserved_bytes']>.8*r['total_device_bytes'] or r['measured_gpu_used_peak_bytes']>.8*r['total_device_bytes'] or r['gpu_memory_samples']<2):
                raise ValueError('Actual allocation/GPU resource margin differs')
    return memory
