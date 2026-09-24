"""Check staged Git blobs (and optional outgoing history) without printing secrets.

Usage: python tools/security/audit_git_payload.py --base origin/master
This is a conservative size/credential-pattern guard, not a complete secret scanner.
"""
import argparse
import json
from pathlib import PurePosixPath
import re
import subprocess


def git(*args):
    return subprocess.check_output(['git',*args])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',help='Also check new blobs in base..HEAD before pushing')
    parser.add_argument('--max-mib',type=float,default=5)
    args=parser.parse_args()
    blobs={}
    for entry in git('ls-files','--stage','-z').split(b'\0'):
        if not entry:continue
        metadata,path=entry.split(b'\t',1)
        mode,oid,stage=metadata.split()
        if stage!=b'0':raise RuntimeError('Unresolved index stages')
        if mode==b'160000':continue
        blobs.setdefault(oid.decode(),set()).add(path.decode('utf-8'))
    if args.base:
        for line in git('rev-list','--objects',f'{args.base}..HEAD').splitlines():
            oid,_,path=line.partition(b' ')
            if git('cat-file','-t',oid.decode()).strip()==b'blob':
                blobs.setdefault(oid.decode(),set()).add(path.decode('utf-8',errors='replace'))
    patterns={
        'private_key':rb'-----BEGIN (?:RSA |DSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----',
        'github_token':rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b',
        'aws_access_id':rb'\bAKIA[0-9A-Z]{16}\b',
        'encrypted_powershell_credential':rb'<SS\s+N="Password">[0-9a-fA-F]{40,}',
        'credential_in_url':rb'https?://[^\s/:]+:[^\s/@]+@',
    }
    findings=[];largest=[];total=0
    for oid,paths in blobs.items():
        size=int(git('cat-file','-s',oid));total+=size
        largest.append((size,sorted(paths)))
        reasons=[]
        if size>args.max_mib*1024**2:reasons.append('oversize')
        for path in paths:
            name=PurePosixPath(path).name.lower()
            if (name.endswith(('.credential.xml','.pem','.pfx','.p12','.key'))
                or name=='.env' or '.ssh/' in path or name in ('id_rsa','id_ed25519')):
                reasons.append('credential_filename')
        data=git('cat-file','blob',oid)
        reasons += [name for name,pattern in patterns.items() if re.search(pattern,data)]
        if reasons:findings.append({'paths':sorted(paths),'bytes':size,'reasons':sorted(set(reasons))})
    print(json.dumps({'checked_blobs':len(blobs),'total_blob_bytes':total,
        'largest_blobs':sorted(largest,reverse=True)[:5], 'findings':findings},indent=2,ensure_ascii=False))
    raise SystemExit(1 if findings else 0)


if __name__=='__main__':main()
