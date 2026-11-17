"""Bounded local regression/benchmark; never connects to or submits to HPC.

Full production geometry and independent S; manufactured SC/event matrices.
These tests certify the data path on these inputs, NOT a physical reconstruction.
CUDA is optional and guarded. Default is CPU; no precision flags are changed.
"""
import argparse, gc, hashlib, os, tempfile, threading, time
import numpy as np
import psutil
import torch
from common import *
from adapters import *

def tensor_sha(t): return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def rel(a,b): return float(torch.linalg.vector_norm((a-b).double())/torch.linalg.vector_norm(b.double()).clamp_min(1e-30))

class Timer:
    def __init__(self,device): self.device=device
    def sync(self):
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
    def run(self, fn):
        self.sync(); t=time.perf_counter(); out=fn(); self.sync()
        return out,time.perf_counter()-t

class RSS:
    def __enter__(self):
        self.stop=threading.Event(); self.start=self.peak=psutil.Process().memory_info().rss
        def sample():
            while not self.stop.wait(.01): self.peak=max(self.peak,psutil.Process().memory_info().rss)
        self.thread=threading.Thread(target=sample,daemon=True); self.thread.start(); return self
    def __exit__(self,*exc):
        self.peak=max(self.peak,psutil.Process().memory_info().rss);self.stop.set();self.thread.join()

def invariant_tests():
    checks={}
    for group in [1,2,8]:
        blocks=[torch.arange(n*13,dtype=torch.float32).reshape(n,13)+i for i,n in enumerate([0,32,1,32,0,31]*6)]
        with OrderedTransferPool(13,'cpu',group_blocks=group,max_host_staging_bytes=1<<20) as pool:
            for repeat in range(3):
                got=[x.clone() for x in pool.iterate(blocks)]
                assert all(torch.equal(x,y) for x,y in zip(got,blocks)) and len(got)==len(blocks)
            assert list(pool.iterate([]))==[]
            g=pool.iterate(blocks);next(g)
            try: next(pool.iterate(blocks));raise AssertionError('concurrent pool accepted')
            except RuntimeError: pass
            g.close(); assert not pool.busy
            assert all(torch.equal(x,y) for x,y in zip([x.clone() for x in pool.iterate(blocks)],blocks))
        checks['ordered_empty_tail_reuse_group_'+str(group)]=True
    for invalid in [[torch.ones(33,13)],[torch.ones(2,13,dtype=torch.float64)],
                    [torch.ones(2,13).T],[torch.ones(13)]]:
        try: block_groups(invalid);raise AssertionError('invalid input accepted')
        except ValueError: pass
    try: OrderedTransferPool(78920,'cpu',max_host_staging_bytes=100);raise AssertionError('budget')
    except MemoryError: pass
    checks['input_and_budget_guards']=True
    return checks

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--max-host-mib',type=int,default=768)
    p.add_argument('--max-gpu-mib',type=int,default=2048)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.iterations not in [50,100] or not 1<=a.repeats<=5: p.error('Only bounded 50/100 iterations; 1..5 repeats')
    if a.device=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable; no GPU result produced')
    torch.set_num_threads(a.threads); torch.set_grad_enabled(False);torch.manual_seed(35100101)
    mods=frozen(); op=mods['torch_active_operator']; joint=mods['jscc_5e10_joint_mlem']; device=torch.device(a.device)
    timer=Timer(device);checks=invariant_tests()
    estimated_host=500<<20
    if estimated_host>a.max_host_mib*(1<<20) or psutil.virtual_memory().available<estimated_host/0.8:
        raise MemoryError('Bounded fixture RAM/headroom guard failed')
    if device.type=='cuda':
        budget=current_cuda_budget(device,256<<20)
        if (500<<20)>min(budget.available(),a.max_gpu_mib*(1<<20)):
            raise MemoryError('Bounded GPU fixture/headroom guard failed')
        torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    geo=op.ActiveGeometry.from_npz(PAYLOAD/'whole_geometry.npz',device)
    # Manufactured dense positive data, fixed seed, never labelled real B or C.
    B=(torch.rand((4,geo.full_count))+.01).to(device)
    y=(torch.rand((4,20))*30+1).to(device); bg=(torch.rand((4,20))*.03).to(device)
    blocks=[[] for _ in range(20)]
    for v,sizes in [(0,[0]+[32]*9+[5]),(5,[32,1]),(10,[0]),(19,[1])]:
        blocks[v]=[(torch.rand((n,geo.active_count))+.01).contiguous() for n in sizes]
    # S NEVER comes from observed synthetic C. It is the exact independent
    # calibration tensor, intentionally not physically matched to toy data.
    sf=torch.tensor(np.fromfile(PAYLOAD/'continuous_energy_Sensi_full','<f4'),device=device)
    sc=geo.compton_sensitivity(sf)
    delivered_s=torch.tensor(np.fromfile(DELIVERY/'result/S_Compton.float32','<f4').reshape(-1,1),device=device)
    checks['independent_compton_S_bitwise_equal']=torch.equal(sc,delivered_s)
    x=torch.tensor(np.fromfile(DELIVERY/'result/Image_440_SinglePlusCompton_active.float32','<f4').reshape(-1,1),device=device)
    preparation=time.perf_counter()-start
    # Warm-up is reported separately; no compile or precision mode toggles.
    base=op.ViewResponse(B,geo,'none')
    (_,warmup)=timer.run(lambda:(base.matrix(0)@x,op._event_weight(blocks[5],x,device)))
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    states={};rows=[]
    names=['baseline','sc_cache','grouped_sync','grouped_prefetch']
    for name in names:
        gc.collect();owner=PhaseCacheOwner();pool=None
        with RSS() as rss:
            timer.sync(); t=time.perf_counter()
            if name=='baseline': response=op.ViewResponse(B,geo,'none')
            else:
                b=(current_cuda_budget(device,256<<20) if device.type=='cuda' else MemoryBudget(1<<30,0))
                response=owner.open(op,B,geo,budget=b,phase='joint_solve_after_compton_prepare')
            events=blocks
            if name.startswith('grouped'):
                pool=OrderedTransferPool(geo.active_count,device,max_host_staging_bytes=160<<20,
                       device_budget=current_cuda_budget(device,256<<20) if device.type=='cuda' else None,
                       asynchronous=name=='grouped_prefetch',timing=True)
                events=[ViewBlocks(pool,b) for b in blocks]
            timer.sync();setup=time.perf_counter()-t
            S,stime=timer.run(response.sensitivity)
            fp,ftime=timer.run(lambda:op.forward_project(response,x))
            def bp():
                result=torch.zeros_like(x)
                for v in range(20):result+=response.matrix(v).T@y[:,v:v+1]/20
                return result
            back,btime=timer.run(bp)
            eventweight,etime=timer.run(lambda:op._event_weight(events[0],x,device))
            lhs=torch.sum(fp.double()*y.double());rhs=torch.sum(x.double()*back.double())
            adj=float(abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-30))
            assert adj<=1e-5
            if pool: pool.reset_stats()
            first_sc,_=timer.run(lambda:op.single_mlem(response,y,S,1,1,additive_background=bg))
            first_joint,_=timer.run(lambda:joint.joint_mlem(response,y,events,S,sc,1,1))
            if pool:pool.reset_stats()
            runs=[];callbacks=[]
            for repeat in range(a.repeats):
                (im,h),st=timer.run(lambda:op.single_mlem(response,y,S,a.iterations,50,additive_background=bg))
                (jim,jh),jt=timer.run(lambda:joint.joint_mlem(response,y,events,S,sc,a.iterations,50,
                    checkpoint_callback=lambda i,h:callbacks.append((repeat,i,tensor_sha(h[-1])))))
                runs.append(dict(sc_seconds=st,joint_seconds=jt))
            assert [c[1] for c in callbacks]==list(range(50,a.iterations+1,50))*a.repeats
            vals=[S,fp,back,eventweight,first_sc[0],first_joint[0],im,h,jim,jh]
            labels=['S','forward','backprojection','event_weight','first_SC','first_joint','SC_final','SC_history','joint_final','joint_history']
            assert all(bool(torch.isfinite(v).all()) and bool((v>=0).all()) for v in vals)
            full=torch.zeros((geo.full_count,1),device=device)
            full[geo.object_active]=jim
            inactive=torch.ones(geo.full_count,dtype=torch.bool,device=device);inactive[geo.object_active]=False
            assert not bool(full[inactive].any())
            if name=='baseline': states={k:v.detach().cpu().clone() for k,v in zip(labels,vals)}
            else:
                for k,val in zip(labels,vals):
                    checks[name+'_'+k+'_bitwise']=bool(torch.equal(states[k],val.cpu()))
            # Independent full-grid float64 SC S check; original operation once.
            if name=='baseline':
                ref=torch.zeros_like(S,dtype=torch.float64)
                for v in range(20):ref+=(B.double()[:,geo.indices[v]]*geo.fraction.double()).sum(0)[:,None]/20
                checks['SC_S_independent_L2_le_1e-5']=rel(S,ref)<=1e-5
            stats=pool.finish_stats() if pool else None
            rows.append(dict(candidate=name,setup_once_seconds=setup,SC_S_seconds=stime,
                forward_seconds=ftime,backprojection_seconds=btime,event_weight_seconds=etime,
                adjoint_relative_error=adj,runs=runs,
                sc_median_seconds=float(np.median([r['sc_seconds'] for r in runs])),
                joint_median_seconds=float(np.median([r['joint_seconds'] for r in runs])),
                cache_views=getattr(response,'cached_views',0),cache_bytes=getattr(response,'cache_bytes',0),
                transfers=stats,outputs={k:tensor_sha(v) for k,v in zip(labels,vals)},
                rss_start_bytes=rss.start,rss_peak_sampled_bytes=rss.peak,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None))
            if pool:pool.close()
            owner.before_compton_prepare()
    # Fixed-prefix partial cache lifecycle, all 20 matrices, no hidden second cache.
    bytes_view=B.shape[0]*geo.active_count*4
    owner=PhaseCacheOwner();partial=owner.open(op,B,geo,budget=MemoryBudget(10*bytes_view,0),phase='sc_solve',requested_views=7)
    assert 0<partial.cached_views<20
    checks['partial_cache_all_views_bitwise']=all(torch.equal(partial.matrix(v),base.matrix(v)) for v in range(20))
    try:owner.open(op,B,geo,budget=MemoryBudget(1<<30,0),phase='sc_solve');raise AssertionError('overlapping cache')
    except RuntimeError:pass
    owner.before_compton_prepare();checks['cache_released_before_preparation']=partial.closed and not partial.partial and partial.base.cached is None
    # A small isolated durable I/O measurement, NOT the production checkpoint writer.
    with tempfile.TemporaryDirectory(prefix='local_checkpoints_',dir=HERE) as td:
        start=time.perf_counter();arr=states['joint_history'].numpy();nbytes=0
        for i,frame in enumerate(arr):
            temp=Path(td)/f'{i}.partial';target=Path(td)/f'{i}.float32'
            with temp.open('xb') as f: f.write(frame.tobytes());f.flush();os.fsync(f.fileno())
            temp.rename(target);assert target.read_bytes()==frame.tobytes();nbytes+=frame.nbytes
        checkpoint=dict(seconds=time.perf_counter()-start,bytes=nbytes,frames=len(arr),
                        scope='Local active-array atomic rename/fsync microbenchmark; original callback indices/data checked separately')
    result=dict(scope='Manufactured FP32 B/C/counts on full actual geometry, independently calibrated S; not a scientific reconstruction.',
       device=str(device),torch=torch.__version__,cuda_available=torch.cuda.is_available(),
       precision_flags=dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32),
       threads=a.threads,iterations=a.iterations,save_step=50,repeats=a.repeats,
       geometry=dict(full=geo.full_count,active=geo.active_count,views=geo.views),
       event_rows=sum(len(b) for bs in blocks for b in bs),original_blocks=sum(len(bs) for bs in blocks),
       fixture_prepare_seconds=preparation,warmup_seconds=warmup,compile_seconds=0,
       rows=rows,checks=checks,passed=all(checks.values()),checkpoint=checkpoint,
       finite_nonnegative_and_inactive_zero=True,
       cuda_peak_scope='Cumulative through each candidate; historical peak is retained, not an isolated memory-saving measurement.',
       communication='Not measured: single-process local test, no distributed all-reduce',
       baseline_H2D='Only CUDA mode moves original blocks; original_calls/bytes are derivable, baseline total includes transfers.',
       gpu_gain_claimed=False if device.type=='cpu' else None)
    write(a.output or HERE/('bench_'+device.type+'.json'),result)
    assert result['passed'],[k for k,v in checks.items() if not v]
    print('BENCH PASS',len(checks),[(r['candidate'],round(r['joint_median_seconds'],3)) for r in rows])

if __name__=='__main__':main()
