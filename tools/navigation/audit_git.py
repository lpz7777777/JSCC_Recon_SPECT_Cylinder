"""Audit only staged changes or outgoing blobs without displaying secret values."""
from __future__ import annotations
import argparse,json,re,subprocess
from pathlib import Path,PurePosixPath

PATTERNS={
 'private-key':rb'-----BEGIN (?:RSA |DSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----',
 'github-token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b',
 'openai-token':rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b',
 'aws-access-key':rb'\bAKIA[0-9A-Z]{16}\b',
 'encrypted-credential':rb'<SS\s+N="Password">[0-9a-fA-F]{40,}',
 'password-in-url':rb'https?://[^\s/:]+:[^\s/@]+@',
 'literal-secret':rb'''(?im)^\s*["']?(?:password|passwd|passphrase|api[_-]?key|access[_-]?token|secret[_-]?key)["']?\s*[:=]\s*["'][^"'\r\n]{8,}["']''',
}
DENIED={'.png','.pdf','.pptx','.jpg','.jpeg','.gif','.webp','.mp4','.npz','.npy','.pt','.pth','.h5','.hdf5','.mat','.sysmat','.bin','.exe','.dll','.so','.whl','.zip','.gz','.tgz','.tar','.7z','.pyc'}
def git(*args):return subprocess.check_output(['git',*args])

def audit(base=None):
    blobs={};changed=[]
    if base:
        for line in git('rev-list','--objects',base+'..HEAD').splitlines():
            oid,_,path=line.partition(b' ')
            if path:blobs.setdefault(oid.decode(),[]).append(path.decode('utf8'))
    else:
        changed=[p.decode('utf8') for p in git('diff','--cached','--name-only','--diff-filter=ACMR','-z').split(b'\0') if p]
        names=set(changed)
        for row in git('ls-files','--stage','-z').split(b'\0'):
            if not row:continue
            meta,path=row.split(b'\t',1);mode,oid,stage=meta.split();name=path.decode('utf8')
            if name not in names:continue
            if stage!=b'0' or mode not in {b'100644',b'100755'}:raise RuntimeError('Unsupported index mode/stage: '+name)
            blobs.setdefault(oid.decode(),[]).append(name)
    proc=subprocess.run(['git','cat-file','--batch'],input=('\n'.join(blobs)+'\n').encode(),capture_output=True,check=True)
    data=proc.stdout;pos=0;findings=[];sizes=[];total=0
    for oid,names in blobs.items():
        end=data.index(b'\n',pos);header=data[pos:end].split();pos=end+1
        if len(header)!=3:raise RuntimeError('Missing Git object '+oid)
        kind=header[1];size=int(header[2]);body=data[pos:pos+size];pos+=size+1
        if kind!=b'blob':continue
        reasons=[];total+=size;sizes.append(dict(bytes=size,oid=oid,paths=names))
        if size>1048576:reasons.append('new-blob-over-1MiB-needs-individual-review')
        if b'\0' in body:reasons.append('binary-NUL')
        for name in names:
            p=PurePosixPath(name);low=name.lower()
            if p.suffix.lower() in DENIED or 'node_modules' in p.parts:reasons.append('media-data-build-or-dependency')
            if ('.ssh/' in low or p.name.lower() in {'.env','id_rsa','id_ed25519'} or low.endswith(('.pem','.key','.pfx','.p12','.credential.xml','.credentials.json'))):reasons.append('credential-filename')
        reasons += [name for name,pattern in PATTERNS.items() if re.search(pattern,body)]
        if reasons:findings.append(dict(paths=names,bytes=size,reasons=sorted(set(reasons))))
    return dict(passed=not findings,scope=base+'..HEAD' if base else 'staged-changes-only',changed_files=len(changed),blobs_checked=len(sizes),total_blob_bytes=total,largest_blobs=sorted(sizes,key=lambda x:x['bytes'],reverse=True)[:10],findings=findings)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base');p.add_argument('--output',type=Path);a=p.parse_args()
    r=audit(a.base)
    if a.output:a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
    if not r['passed']:raise SystemExit(1)
if __name__=='__main__':main()
