"""Explicit local GPU binding and node-aggregate resource evidence."""
from datetime import timedelta
import os, resource, subprocess, time, threading
import torch
import torch.distributed as dist

_monitor=None

class DeviceMonitor:
    def __init__(self,device):
        prop=torch.cuda.get_device_properties(device);self.uuid=str(getattr(prop,'uuid',''))
        if not self.uuid:
            data=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
            matches=[x.split(',')[1].strip() for x in data.splitlines() if x.split(',')[0].strip()==str(os.getpid())]
            if len(matches)!=1:raise ValueError('Actual process-to-GPU UUID evidence missing')
            self.uuid=matches[0]
        self.peak=0;self.samples=0;self.error=None;self.stop=threading.Event()
        self.sample();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def sample(self):
        data=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,timeout=10)
        values=[x.split(',') for x in data.splitlines() if x.split(',')[0].strip()==self.uuid]
        if len(values)!=1:raise ValueError('Allocated GPU UUID missing from measured memory evidence')
        self.peak=max(self.peak,int(values[0][1].strip())*(1<<20));self.samples+=1
    def loop(self):
        while not self.stop.wait(5):
            try:self.sample()
            except Exception as e:self.error=str(e);return
    def finish(self):
        self.stop.set();self.thread.join(timeout=15);self.sample()
        if self.error:raise ValueError('Actual GPU memory monitoring failed: '+self.error)

def setup():
    global _monitor
    rank,world,local=[int(os.environ[n]) for n in ('RANK','WORLD_SIZE','LOCAL_RANK')]
    if world!=32 or not 0<=local<4 or torch.cuda.device_count()!=4:raise ValueError('Frozen 8-node x4-GPU topology required')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.set_num_threads(10);torch.set_grad_enabled(False)
    if os.environ.get('NCCL_SOCKET_IFNAME')!='bond0':raise ValueError('Registered NCCL interface required')
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    _monitor=DeviceMonitor(device)
    return rank,world,local,device

def resource_record(rank,local,device,started):
    prop=torch.cuda.get_device_properties(device);_monitor.finish()
    return dict(rank=rank,local_rank=local,node=os.environ['SLURMD_NODENAME'],pid=os.getpid(),gpu_uuid=_monitor.uuid,
        host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        host_allocated_bytes_node=int(os.environ['JSCC_HOST_BYTES_NODE']),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        total_device_bytes=prop.total_memory,elapsed_seconds=time.monotonic()-started,
        measured_gpu_used_peak_bytes=_monitor.peak,gpu_memory_samples=_monitor.samples,gpu_memory_sampling_period_seconds=5)

def gather_resources(record):
    result=[None]*32;dist.all_gather_object(result,record);return result
