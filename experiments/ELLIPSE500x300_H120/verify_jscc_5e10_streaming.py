"""Read-only complete image/identity/topology/S/fixed-background verification."""
import argparse, hashlib, math, os, time
from pathlib import Path
import numpy as np
from jscc_5e10_common import *
from jscc_5e10_streaming_contract import load_contract,verify_checkpoints,verify_topology

def rel(x,y):return float(np.linalg.norm(np.asarray(x,float)-np.asarray(y,float))/max(np.linalg.norm(np.asarray(y,float)),1e-30))

def verify_selection(input_root,selection_root,manifest):
    for v in range(1,21):
        indices=np.load(selection_root/'selections'/f'{v}.npy')
        rows=np.load(selection_root/'selected_rows'/f'{v}.npy')
        raw=np.loadtxt(input_root/'List'/f'{v}.csv',delimiter=',',usecols=(0,1,2,3),dtype=np.float32,ndmin=2)
        if len(indices)!=manifest['events_per_view'][v-1] or np.any(indices[1:]<=indices[:-1]) or np.any(indices<0) or np.any(indices>=len(raw)):
            raise ValueError('Complete new selection index identity differs')
        if not np.array_equal(rows,raw[indices]):raise ValueError('Cached selected rows are not exact new raw event rows')
        print('ALL_SELECTED_RAW_ROWS_MATCH',v,len(indices),flush=True)

def own_s(raw,g):
    column=np.zeros(132040,np.float64)
    for i in range(0,132040,1024):column[i:i+1024]=raw[i:i+1024].sum(axis=1,dtype=np.float64)
    active=g['active_indices'];f=g['ellipse_fraction'][active]
    return sum(column[g['inverse_rotation'][active,v]]*f for v in range(20))/20

def cross_prediction(raw,g,image):
    active=g['active_indices'];fraction=g['ellipse_fraction'][active]
    scattered=np.zeros((132040,20),np.float64)
    for v in range(20):np.add.at(scattered[:,v],g['inverse_rotation'][active,v],image*fraction/20)
    prediction=np.zeros((10496,20),np.float64)
    for i in range(0,132040,1024):
        prediction+=np.asarray(raw[i:i+1024],np.float64).T@scattered[i:i+1024]
        if i%16384==0:print('READ_ONLY_FIXED_BACKGROUND_PIXELS',i,flush=True)
    return prediction

def verify(result,contract,allocation,mode,input_root,factors):
    began=time.monotonic();iterations,step=execution_policy(mode);cfg=load_contract(contract,mode,iterations,step);root=contract.parent
    run=read(result/'run_manifest.json')
    if (run['study'],run['iterations'],run['save_step'],run['channels'],run['actual_primary_photons'])!=(STUDY,iterations,step,list(CHANNELS),TOTAL):
        raise ValueError('This complete three-route/dose run required')
    if run['contract_sha256']!=digest(contract) or run['helper_sha256']!=cfg['files'] or run['joint_solver_enabled'] or run['regularization']!='none' or run['initial_density']!=1:
        raise ValueError('Executed algorithm/release identity differs')
    for k in ('input_sha256','factor_manifest_sha256','factor_payload_sha256'):
        if run[k]!=cfg[k]:raise ValueError('Executed frozen input identity differs')
    if run['geometry_sha256']!=cfg['whole_geometry_sha256'] or run['sensitivity_sha256']!=cfg['files']['continuous_energy_Sensi_full']:
        raise ValueError('Matched basis/S identity differs')
    coll=validate_collection(read(input_root/'collection.json'));verify_files(input_root,cfg['input_sha256']);verify_files(factors,cfg['factor_payload_sha256'])
    if (run.get('world_size'),run.get('nodes'),run.get('gpus_per_node'),run.get('response_storage_policy')) != (24,8,3,cfg['response_storage_policy']):
        raise ValueError('Actual streaming reconstruction topology/storage differs')
    resources=run['resources'];memory=verify_topology(resources,allocation)
    if run['accepted_compton_events']!=cfg['accepted_events'] or run['events_per_view']!=cfg['events_per_view'] or sum(r['accepted_events'] for r in resources)!=cfg['accepted_events']:
        raise ValueError('Complete event count differs')
    for v,count in enumerate(cfg['events_per_view'],1):
        indices=np.load(root/'selections'/f'{v}.npy');cursor=0
        for r in sorted(resources,key=lambda x:x['rank']):
            part=r['partitions'][v-1];lo=part['first_selected_position'];hi=part['last_selected_position_exclusive']
            if part['view']!=v or lo!=cursor or hi<lo or part['events']!=hi-lo or part['original_rows_sha256']!=hashlib.sha256(indices[lo:hi].astype('<i8').tobytes()).hexdigest():
                raise ValueError('Missing/overlapping or changed actual event partitions')
            cursor=hi
        if cursor!=count:raise ValueError('All view rows must close')
    for r in resources:
        if not np.isfinite(r['minimum_active_event_mass']) or r['minimum_active_event_mass']<=0 or not 0<=r['initial_forward_floor_events']<=r['accepted_events']:
            raise ValueError('Invalid retained-event support diagnostics')
        cache=read(result/f"cache_rank{r['rank']:02d}.json")
        if (not cache['complete'] or cache['policy']!=cfg['response_storage_policy'] or cache['contract_sha256']!=digest(contract) or
                cache['rank']!=r['rank'] or cache['node']!=r['node'] or len(cache['views'])!=20 or
                sum(x['events'] for x in cache['views'])!=r['accepted_events'] or
                any(not x['complete_readback_passed'] or not x['all_generated_blocks_byte_exact_roundtrip'] for x in cache['views'])):
            raise ValueError('Complete lossless response cache identity/readback differs')
        n=r['numerical_checks']
        if (n['sample_events'],n['active_points'],n['full_points'])!=(32,78920,132040) or n['source_truth_used']:raise ValueError('Actual event numerical check dimensions differ')
        for key in ('chunk_relative_L2','rank_weight_relative_L2','forward_adjoint_relative_error','event_constant_relative_error'):
            if not np.isfinite(n[key]) or not 0<=n[key]<=1e-5:raise ValueError('Original event operator numerical check failed')
        if mode=='validation':
            if set(r['single_regression_relative_L2'])!=set(CHANNELS[:2]) or any(not np.isfinite(x) or not 0<=x<=1e-5 for x in r['single_regression_relative_L2'].values()):raise ValueError('Original single MLEM regression failed')
            if not np.isfinite(r['original_compton_regression_relative_L2']) or not 0<=r['original_compton_regression_relative_L2']<=1e-5:raise ValueError('Original Compton branch regression failed')
            for name in ('A218','A440','C440to218'):
                ch=r['single_operator_checks'][name]
                if ch['views']!=20 or len(ch['forward_relative_L2'])!=20 or len(ch['adjoint_relative_L2'])!=20 or max(ch['forward_relative_L2']+ch['adjoint_relative_L2']+[ch['sensitivity_relative_L2']])>1e-5:
                    raise ValueError('All-row/view original forward/transpose/S check failed')
    g=np.load(root/'whole_geometry.npz');active=g['active_indices'];inactive=np.ones(132040,bool);inactive[active]=False;histories={};outputs=[]
    for channel in CHANNELS:
        paths={kind:result/f'Image_{channel}_{kind}.float32' for kind in ('active','full','history')};values={}
        for kind,path in paths.items():
            n=132040 if kind=='full' else 78920*(iterations//step if kind=='history' else 1)
            if path.stat().st_size!=n*4:raise ValueError('Complete image/history size differs')
            values[kind]=np.fromfile(path,'<f4')
            if not np.isfinite(values[kind]).all() or np.any(values[kind]<0):raise ValueError('Nonfinite/negative image')
        history=values['history'].reshape(iterations//step,78920)
        if not np.array_equal(values['active'],history[-1]) or not np.array_equal(values['full'][active],values['active']) or np.any(values['full'][inactive]!=0):raise ValueError('History/last frame/support differs')
        histories[channel]=history;outputs.append(dict(channel=channel,frames=len(history),sha256={k:digest(p) for k,p in paths.items()}))
    allowed={f'Image_{c}_{k}.float32' for c in CHANNELS for k in ('active','full','history')}
    if {p.name for p in result.glob('Image_*.float32')}!=allowed:raise ValueError('Unexpected joint or cross-energy output')
    checkpoints=verify_checkpoints(result,histories,active,digest(contract),mode)
    sensitivity_checks={}
    for energy in (218,440):
        raw=np.memmap(factors/f'{energy}keV_RotateNum20/SysMat_polar','<f4',mode='r',shape=(132040,10496))
        expected=own_s(raw,g);actual=np.fromfile(result/f'S_{energy}.float32','<f4')
        sensitivity_checks[str(energy)]=rel(actual,expected)
        if not np.isfinite(sensitivity_checks[str(energy)]) or sensitivity_checks[str(energy)]>1e-5:raise ValueError('Full matrix own sensitivity CPU verification failed')
        del raw
    expected=np.fromfile(root/'continuous_energy_Sensi_full','<f4')[active]*g['ellipse_fraction'][active]
    if not np.array_equal(np.fromfile(result/'S_Compton.float32','<f4'),expected.astype('<f4')):raise ValueError('Own matched Compton sensitivity changed')
    cross=np.memmap(factors/'440keV_to218win_RotateNum20/SysMat_polar','<f4',mode='r',shape=(132040,10496))
    expected=cross_prediction(cross,g,histories[CHANNELS[0]][-1]);actual=np.fromfile(result/'PredictedCntStat_218_From440.float32','<f4').reshape(10496,20)
    error=rel(actual,expected)
    if not np.isfinite(error) or error>1e-5 or not np.isfinite(actual).all() or np.any(actual<0):raise ValueError('Fixed background is not this actual final440 image forward prediction')
    if mode=='formal':
        authority=read(Path(run['authority_file']))
        if digest(run['authority_file'])!=run['authority_sha256'] or not authority['passed'] or authority['contract_sha256']!=digest(contract):raise ValueError('Actual full-input authority differs')
    return dict(passed=True,study=STUDY,mode=mode,iterations=iterations,save_step=step,contract_sha256=digest(contract),
        run_manifest_sha256=digest(result/'run_manifest.json'),allocation_sha256=digest(allocation),outputs=outputs,checkpoints=checkpoints,
        prediction_sha256=digest(result/'PredictedCntStat_218_From440.float32'),sensitivity_relative_L2=sensitivity_checks,
        fixed_background_relative_L2=error,resources=resources,host_allocated_bytes_node=memory,accepted_events=cfg['accepted_events'],
        actual_primary_photons=TOTAL,verification_source_sha256=digest(__file__),elapsed_seconds=time.monotonic()-began,
        cpu_read_only_verification_is_not_gpu_resource_certificate=True)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('result','contract','allocation','input','factors','receipt'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--mode',choices=('validation','formal'),required=True);a=p.parse_args()
    if a.receipt.exists():raise FileExistsError('Never overwrite an acceptance receipt')
    proof=verify(a.result,a.contract,a.allocation,a.mode,a.input,a.factors);write(a.receipt,proof);print('JSCC_COMPLETE_STRICT_VERIFICATION_PASSED',a.mode,flush=True)
