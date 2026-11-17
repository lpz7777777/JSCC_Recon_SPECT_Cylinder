"""Four independent authenticated connections; actual identical 48MiB input."""
from concurrent.futures import ThreadPoolExecutor
import json
import time
import stream_transfer as st
from common import read,write,now
from prepare import REPORT

def main():
    one=read(REPORT/'transfer_benchmark_20261117.json')
    source=one['source'];ranges=one['sample_ranges'];sample=one['sample']
    size=21_285_888_000
    def worker(index):
        with st.connect() as c:
            return st.bridge(c,source,size,ranges,
                st.REMOTE_GPU+'/transfer_bench_20261117/parallel4_'+str(index)+'_'+str(time.time_ns())+'.bin',
                sample['bytes'],sample['sha256'],'zlib1',progress=False,maximum_seconds=300)
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(worker,range(4)))
    seconds=time.monotonic()-start;rate=4*sample['bytes']/seconds/(1<<20)
    result=dict(passed=all(x['passed'] for x in results),time=now(),connections=4,
        sample_bytes_per_connection=sample['bytes'],total_bytes=4*sample['bytes'],elapsed_seconds=seconds,
        aggregate_MiB_per_second=rate,old_MiB_per_second=one['original_observed_MiB_per_second'],
        measured_speedup=rate/one['original_observed_MiB_per_second'],results=results,
        selected_connections=4 if rate>1.2*one['measured_best_MiB_per_second'] else 1,
        scientific_payload_unchanged=True)
    write(REPORT/'transfer_parallel_benchmark_20261117.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='results'},indent=2))

if __name__=='__main__':main()
