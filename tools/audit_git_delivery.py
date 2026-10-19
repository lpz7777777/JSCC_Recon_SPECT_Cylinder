"""Audit staged UTF-8 small-text delivery without reading ignored runtime data."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BANNED_SUFFIXES = {'.png','.jpg','.jpeg','.gif','.webp','.pdf','.svg','.npz','.npy','.mat',
                   '.raw','.bin','.float32','.float64','.int32','.int64','.uint32','.uint64',
                   '.zip','.gz','.tar','.7z','.h5','.hdf5','.pt','.pth','.exe','.dll','.so',
                   '.o','.obj','.pdb','.pem','.key','.pfx','.p12'}
SECRET_PATTERNS = [
    re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    re.compile(r'''(?im)(?:password|passwd|api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*["']([^"'\r\n]{8,})["']'''),
]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output_relative = output.relative_to(ROOT).as_posix()
    paths = [p for p in git('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z').decode().split('\0') if p]
    records, errors = [], []
    for name in paths:
        path = Path(name)
        blob = git('show', ':'+name)
        reasons = []
        if path.suffix.lower() in BANNED_SUFFIXES:
            reasons.append('binary/media/array/archive/key extension')
        if any(part.lower() in {'generated','factors','cntstat','list','selected_rows','selections','.ssh'} for part in path.parts):
            reasons.append('runtime observation/matrix/credential directory')
        if path.name.lower().startswith(('.env','id_rsa','id_ed25519','known_hosts')):
            reasons.append('credential filename')
        if len(blob) > 1 << 20:
            reasons.append('over 1 MiB small-text limit')
        try:
            text = blob.decode('utf-8-sig')
            if '\0' in text:
                reasons.append('NUL/binary content')
            if path.suffix == '.json':
                json.loads(text)
            # Scanner source itself contains literal recognition patterns.
            if name != Path(__file__).resolve().relative_to(ROOT).as_posix():
                if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                    reasons.append('possible embedded credential')
        except (UnicodeError, json.JSONDecodeError):
            reasons.append('not valid UTF-8 text/JSON')
        if reasons:
            errors.append(dict(path=name, reasons=reasons))
        if name != output_relative:
            records.append(dict(path=name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest(),
                                git_blob=git('rev-parse', ':'+name).decode().strip()))
    if not records:
        errors.append(dict(path='(index)', reasons=['No non-self staged delivery files']))
    result = dict(passed=not errors, audited_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        base_head=git('rev-parse','HEAD').decode().strip(), staged_nonself_files=len(records),
        total_nonself_bytes=sum(r['bytes'] for r in records), max_file_bytes=max((r['bytes'] for r in records), default=0),
        excluded_self=output_relative, scope='Exact staged blobs; UTF-8 small text only; no new arrays/media/archives/credentials',
        errors=errors, files=records)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
