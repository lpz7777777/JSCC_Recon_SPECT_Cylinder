"""Four bounded 256MiB transfers with ordered fsync merge and strict full SHA.

At most four chunk files (~1GiB) exist in addition to the final matrix. The
original contiguous SFTP prefix is verified and retained. Frozen matrices,
science source, reconstruction and submission logic are unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import time
import framed_stream_r2 as st
from common import read,write,now
from prepare import REPORT

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
    plan=source_plan(source,size,expected,offset)
    if offset and st.checksum(client,partial)!=plan['prefix_sha256']:
        raise ValueError('Original retained prefix mismatch; do not overwrite')
    planfile=REPORT/('transfer_range_plan_'+channel+'_r2_20261117.json')
    write(planfile,dict(time=now(),**plan))
    from common import sha
    plan_sha=sha(planfile)
    policy=read(REPORT/'transfer_parallel_benchmark_20261117.json');workers=policy['selected_connections']
    compression=read(REPORT/'transfer_benchmark_20261117.json')['selected_compression']
    st.rpc(client,'from pathlib import Path;Path('+repr(parts_root)+').mkdir(exist_ok=True)')
    chunks=plan['chunks'];merged=[];wire=0
    def worker(item):
        name=f"range_{item['start']:012d}_{item['bytes']:09d}.bin";dest=parts_root+'/'+name
        with st.connect() as c:
            digest=st.checksum(c,dest)
            if digest!='ABSENT':
                if digest==item['sha256']:
                    return dict(part=dest,part_name=name,bytes=item['bytes'],sha256=digest,wire_bytes=0,reused_verified=True)
                prefix=int(st.rpc(c,'from pathlib import Path;print(Path('+repr(dest)+').stat().st_size)',60))
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
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={i:pool.submit(worker,c) for i,c in enumerate(chunks[:workers])}
        for i,item in enumerate(chunks):
            result=futures.pop(i).result()
            cfg=dict(partial=partial,part=result['part'],part_name=result['part_name'],parts_root=parts_root,
                start=item['start'],bytes=item['bytes'],sha256=item['sha256'],plan_sha256=plan_sha)
            receipt=json.loads(st.rpc(client,MERGE.replace("c=json.loads(sys.argv[1])",'c='+repr(cfg)),120))
            merged.append(receipt);wire+=result['wire_bytes']
            completed=item['start']+item['bytes'];elapsed=time.monotonic()-began
            progress=dict(time=now(),status='PARALLEL_CHUNKS',protocol='length-framed-r2',channel=channel,source=source,target=target,
                bytes_complete=completed,bytes_total=size,preserved_prefix_bytes=offset,workers=workers,
                chunks_complete=len(merged),chunks_total=len(chunks),wire_bytes_completed=wire,
                elapsed_seconds=elapsed,effective_MiB_per_second=(completed-offset)/max(elapsed,1e-6)/(1<<20),
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
    write(REPORT/('transfer_complete_'+channel+'_r2_20261117.json'),result)
    return result
