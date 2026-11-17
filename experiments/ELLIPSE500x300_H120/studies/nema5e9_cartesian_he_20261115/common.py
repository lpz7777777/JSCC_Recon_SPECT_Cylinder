"""Small immutable evidence/output helpers for the isolated Cartesian study."""
import datetime
import hashlib
import json
import os
from pathlib import Path

TAG = 'nema5e9_cartesian_he_20261115'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8<<20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic(path, data, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    tmp = path.with_name(path.name+'.partial')
    with tmp.open('wb' if replace else 'xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write(path, obj, replace=False):
    atomic(path, (json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf8'), replace)


def verify_release(root):
    manifest = read(root/'release_manifest.json')
    for name, expected in manifest['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen release differs: '+name)
    return manifest
