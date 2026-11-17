"""Four bounded 256MiB transfers with ordered fsync merge and strict full SHA.

At most four chunk files (~1GiB) exist in addition to the final matrix. The
original contiguous SFTP prefix is verified and retained. Frozen matrices,
science source, reconstruction and submission logic are unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time
import framed_stream_r4 as st
from transport_retries_r3 import retry_io,TransferIOError
from common import read,write,now
from prepare import REPORT

EMPTY_PRESERVE=r'''from pathlib import Path
import hashlib,json,os,sys,time
c=json.loads(sys.argv[1]);p=Path(c['part']);root=Path(c['parts_root'])
assert p.resolve().parent==root.resolve() and p.name==c['part_name']
assert p.is_file() and p.stat().st_size==0
lock=open(str(p)+'.writer.lock','a+b')
if os.name=='posix':
 import fcntl
 try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  print(json.dumps(dict(busy=True)));sys.exit(0)
assert p.stat().st_size==0 and hashlib.sha256(p.read_bytes()).hexdigest()==c['empty_sha256']
saved=p.with_name(p.name+'.empty_'+str(time.time_ns())+'.preserved')
assert not saved.exists() and saved.resolve().parent==root.resolve()
os.rename(p,saved)
receipt=dict(passed=True,busy=False,original=str(p),preserved=str(saved),bytes=0,
 sha256=c['empty_sha256'],exclusive_writer_lock=(os.name=='posix'),time_epoch=time.time())
with Path(str(saved)+'.json').open('x') as f:json.dump(receipt,f);f.flush();os.fsync(f.fileno())
if os.name=='posix':
 fd=os.open(str(root),os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
print(json.dumps(receipt))
'''


def preserve_empty_range(client,part,parts_root,part_name,digest):
    if digest!='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855':
        raise ValueError('Zero-byte chunk has inconsistent SHA; preserve and stop')
    cfg=dict(part=part,parts_root=parts_root,part_name=part_name,empty_sha256=digest)
    receipt=json.loads(st.rpc(client,EMPTY_PRESERVE.replace("c=json.loads(sys.argv[1])",'c='+repr(cfg)),60))
    if receipt.get('busy'):raise TransferIOError('Zero-byte range still has an active writer')
    if not receipt.get('passed') or receipt.get('bytes')!=0:raise ValueError('Empty range preservation failed')
    folder=REPORT/'transfer_empty_chunk_preservation_r6_20261117';folder.mkdir(exist_ok=True)
    write(folder/(str(time.time_ns())+'.json'),dict(time=now(),**receipt))
    return receipt

MERGE=r'''from pathlib import Path
import hashlib,json,os,sys,time
c=json.loads(sys.argv[1]);p=Path(c['partial']);part=Path(c['part']);root=Path(c['parts_root'])
if part.resolve().parent!=root.resolve():raise ValueError('Temporary range outside authorized parts directory')
if p.resolve().parent!=root.resolve().parent:raise ValueError('Unexpected final matrix directory')
assert part.name==c['part_name'] and part.stat().st_size==c['bytes']
if not p.exists():
 assert c['start']==0
 with p.open('xb') as f:f.flush();os.fsync(f.fileno())
assert p.stat().st_size==c['start'],'Prefix changed; never overwrite successful bytes'
h=hashlib.sha256();n=0
with part.open('rb') as src,p.open('ab') as dst:
 for b in iter(lambda:src.read(4<<20),b''):
  dst.write(b);h.update(b);n+=len(b)
 dst.flush();os.fsync(dst.fileno())
assert n==c['bytes'] and h.hexdigest()==c['sha256']
assert p.stat().st_size==c['start']+c['bytes']
receipt=dict(passed=True,start=c['start'],bytes=n,sha256=h.hexdigest(),prefix_bytes=p.stat().st_size,
 plan_sha256=c['plan_sha256'],fsync=True,merged_epoch=time.time())
record=root/(part.name+'.merged.json')
assert not record.exists()
with record.open('x') as f:json.dump(receipt,f);f.flush();os.fsync(f.fileno())
# Only this verified derived temporary range is removed, after durable merge.
assert part.resolve().parent==root.resolve() and part.name==c['part_name']
part.unlink()
print(json.dumps(receipt))
'''


def source_plan(source,size,expected,offset):
    code=r'''import hashlib,json,os,time
p=SOURCE;size=SIZE;offset=OFFSET;block=256<<20;full=hashlib.sha256();prefix=hashlib.sha256();chunks=[];start=time.monotonic()
with open(p,'rb') as f:
 assert os.fstat(f.fileno()).st_size==size
 left=offset
 while left:
  b=f.read(min(4<<20,left));assert b;left-=len(b);prefix.update(b);full.update(b)
 position=offset
 while position<size:
  length=min(block,size-position);left=length;h=hashlib.sha256()
  while left:
   b=f.read(min(4<<20,left));assert b;left-=len(b);h.update(b);full.update(b)
  chunks.append(dict(start=position,bytes=length,sha256=h.hexdigest()));position+=length
 assert not f.read(1)
assert full.hexdigest()==EXPECTED
print(json.dumps(dict(source=p,bytes=size,sha256=full.hexdigest(),offset=offset,prefix_sha256=prefix.hexdigest(),
 chunks=chunks,block_bytes=block,source_hash_seconds=time.monotonic()-start)))'''.replace('SOURCE',repr(source)).replace('SIZE',str(size)).replace('OFFSET',str(offset)).replace('EXPECTED',repr(expected))
    return json.loads(st.server(code,timeout=900))


def transfer_one(client,source,target,expected,size):
    channel=Path(target).parent.name;began=time.monotonic()
    existing=st.checksum(client,target)
    if existing!='ABSENT':
        if existing!=expected:raise ValueError('Successful matrix destination differs')
        return dict(source=source,target=target,bytes=size,sha256=expected,reused_verified=True)
    partial=target+'.partial';parts_root=str(Path(target).parent).replace('\\','/')+'/.stream_parts_20261117'
    code='from pathlib import Path;import json;p=Path('+repr(partial)+');print(json.dumps(dict(bytes=p.stat().st_size if p.exists() else 0)))'
    offset=json.loads(st.rpc(client,code,60))['bytes']
    if offset>size:raise ValueError('Old partial larger than original matrix')
    planfile=REPORT/('transfer_range_plan_'+channel+'_r6_'+str(offset)+'_20261117.json')
    if planfile.exists():
        plan=read(planfile)
        if (plan['source'],plan['bytes'],plan['sha256'],plan['offset'])!=(source,size,expected,offset):raise ValueError('Previous range plan identity differs')
    else:
        plan=source_plan(source,size,expected,offset);write(planfile,dict(time=now(),**plan))
    if offset and st.checksum(client,partial)!=plan['prefix_sha256']:
        raise ValueError('Original retained prefix mismatch; do not overwrite')
    from common import sha
    plan_sha=sha(planfile)
    policy=read(REPORT/'transfer_parallel_benchmark_20261117.json');workers=policy['selected_connections']
    compression=read(REPORT/'transfer_benchmark_20261117.json')['selected_compression']
    st.rpc(client,'from pathlib import Path;Path('+repr(parts_root)+').mkdir(parents=True,exist_ok=True)')
    chunks=plan['chunks'];merged=[];wire=0;fresh_payload=0
    def worker_once(item):
        name=f"range_{item['start']:012d}_{item['bytes']:09d}.bin";dest=parts_root+'/'+name
        with st.connect() as c:
            state=st.prefix_state(c,dest);digest=state['sha256']
            if digest!='ABSENT':
                if digest==item['sha256']:
                    return dict(part=dest,part_name=name,bytes=item['bytes'],sha256=digest,wire_bytes=0,reused_verified=True)
                prefix=state['bytes']
                if prefix==0:
                    empty_receipt=preserve_empty_range(c,dest,parts_root,name,digest)
                    result=st.bridge(c,source,size,[[item['start'],item['bytes']]],dest,item['bytes'],item['sha256'],
                        compression,progress=False,maximum_seconds=900)
                    result.update(part=dest,part_name=name,reused_verified=False,preserved_empty=empty_receipt)
                    return result
                if not 0<prefix<item['bytes']:raise ValueError('Retained chunk has invalid size; preserve')
                code="import hashlib;f=open("+repr(source)+",'rb');f.seek("+str(item['start'])+");h=hashlib.sha256();left="+str(prefix)+"\nwhile left:\n b=f.read(min(4<<20,left));assert b;left-=len(b);h.update(b)\nprint(h.hexdigest())"
                expected_prefix=st.server(code,timeout=120).strip()
                if digest!=expected_prefix:raise ValueError('Retained source/destination chunk prefixes differ')
                result=st.bridge(c,source,size,[[item['start']+prefix,item['bytes']-prefix]],dest,item['bytes'],item['sha256'],compression,
                    offset=prefix,prefix_sha256=digest,progress=False,maximum_seconds=900)
                result.update(part=dest,part_name=name,reused_verified=True,reused_prefix_bytes=prefix);return result
            result=st.bridge(c,source,size,[[item['start'],item['bytes']]],dest,item['bytes'],item['sha256'],
                compression,progress=False,maximum_seconds=900)
            result.update(part=dest,part_name=name,reused_verified=False);return result
    def retry_log(row):
        folder=REPORT/'transfer_retry_events_r6_20261117';folder.mkdir(exist_ok=True)
        write(folder/(str(time.time_ns())+'.json'),dict(time=now(),channel=channel,**row))
    def worker(item):
        return retry_io(lambda:worker_once(item),label=channel+':'+str(item['start']),attempts=5,logger=retry_log)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={i:pool.submit(worker,c) for i,c in enumerate(chunks[:workers])}
        for i,item in enumerate(chunks):
            result=futures.pop(i).result()
            cfg=dict(partial=partial,part=result['part'],part_name=result['part_name'],parts_root=parts_root,
                start=item['start'],bytes=item['bytes'],sha256=item['sha256'],plan_sha256=plan_sha)
            receipt=json.loads(st.rpc(client,MERGE.replace("c=json.loads(sys.argv[1])",'c='+repr(cfg)),120))
            merged.append(receipt);wire+=result['wire_bytes'];fresh_payload+=item['bytes']-(result.get('reused_prefix_bytes',item['bytes'] if result['wire_bytes']==0 else 0))
            completed=item['start']+item['bytes'];elapsed=time.monotonic()-began
            progress=dict(time=now(),status='PARALLEL_CHUNKS',protocol='length-framed-r6-reconnect',channel=channel,source=source,target=target,
                bytes_complete=completed,bytes_total=size,preserved_prefix_bytes=offset,workers=workers,
                chunks_complete=len(merged),chunks_total=len(chunks),wire_bytes_completed=wire,
                elapsed_seconds=elapsed,effective_MiB_per_second=fresh_payload/max(elapsed,1e-6)/(1<<20),new_payload_bytes_completed=fresh_payload,
                compression=compression,plan_sha256=plan_sha,strict_range_SHA_and_fsync=True)
            write(REPORT/'transfer_stream_progress_20261117.json',progress,replace=True)
            print('PARALLEL_MATRIX_CHUNK',channel,completed,size,round(progress['effective_MiB_per_second'],2),flush=True)
            next_index=i+workers
            if next_index<len(chunks):futures[next_index]=pool.submit(worker,chunks[next_index])
    if st.checksum(client,partial)!=expected:raise ValueError('Final independent uncompressed matrix SHA failed')
    st.rpc(client,'from pathlib import Path;import os;p=Path('+repr(partial)+');q=Path('+repr(target)+');assert p.stat().st_size=='+str(size)+';assert not q.exists();os.rename(p,q)')
    result=dict(passed=True,source=source,target=target,bytes=size,sha256=expected,reused_verified=False,
        preserved_prefix_bytes=offset,preserved_prefix_sha256=plan['prefix_sha256'],workers=workers,
        chunks=len(chunks),elapsed_seconds=time.monotonic()-began,wire_bytes=wire,compression=compression,
        source_range_plan_sha256=plan_sha,all_chunks=merged,full_uncompressed_SHA_verified=True,
        original_scientific_matrix_unchanged=True,max_extra_chunk_storage_bytes=workers*(256<<20))
    write(REPORT/('transfer_complete_'+channel+'_r6_20261117.json'),result)
    return result
