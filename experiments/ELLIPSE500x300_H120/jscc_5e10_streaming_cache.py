"""Lossless, bounded 32-event disk replay; never changes response arithmetic."""
import hashlib
import os
from pathlib import Path
import shutil
import struct
import time
import zlib
import numpy as np
import torch
from jscc_5e10_common import digest, read, write

MAGIC = b'JSCC32F4'
HEADER = struct.Struct('<8sIIQQ32s32s')
MAX_ROWS = 32
POLICY = 'lossless_zlib_float32_32event_local_disk_v1'


def drop_file_cache(fd):
    # Only the descriptor of this experiment's own cache is advised away.
    if hasattr(os, 'posix_fadvise'):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


class CacheWriter:
    def __init__(self, path, columns, reserve_bytes=8 << 30):
        self.path = Path(path)
        self.columns = columns
        self.reserve_bytes = reserve_bytes
        self.partial = self.path.with_suffix('.partial')
        if self.path.exists() or self.partial.exists():
            raise FileExistsError('Preserve existing/partial response cache')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.partial.open('xb')
        self.events = self.blocks = self.raw_bytes = self.encoded_bytes = 0
        self.roundtrip_max_abs = 0.
        self.started = time.monotonic()

    def append(self, tensor):
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype='<f4')
        if (array.ndim != 2 or array.shape[1] != self.columns or
                not 0 < len(array) <= MAX_ROWS or not np.isfinite(array).all() or (array < 0).any()):
            raise ValueError('Complete finite nonnegative 32-event rows required')
        raw = array.tobytes()
        encoded = zlib.compress(raw, 1)
        if zlib.decompress(encoded) != raw:
            raise ValueError('Lossless response storage roundtrip differs')
        if shutil.disk_usage(self.path.parent).free < len(encoded) + HEADER.size + self.reserve_bytes:
            raise OSError('Node-local response cache must retain reserved disk space')
        self.file.write(HEADER.pack(MAGIC, len(array), self.columns, len(raw), len(encoded),
                                   hashlib.sha256(raw).digest(), hashlib.sha256(encoded).digest()))
        self.file.write(encoded)
        # Bound dirty pages as well as Python/torch memory; no full response list.
        self.file.flush()
        self.events += len(array)
        self.blocks += 1
        self.raw_bytes += len(raw)
        self.encoded_bytes += len(encoded) + HEADER.size
        if self.blocks % 64 == 0:
            os.fsync(self.file.fileno())
            drop_file_cache(self.file.fileno())

    def finish(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        drop_file_cache(self.file.fileno())
        self.file.close()
        os.replace(self.partial, self.path)
        if os.name == 'posix':
            fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        receipt = dict(policy=POLICY, path=str(self.path), columns=self.columns,
                       events=self.events, blocks=self.blocks, raw_bytes=self.raw_bytes,
                       encoded_bytes=self.encoded_bytes, sha256=digest(self.path),
                       all_generated_blocks_byte_exact_roundtrip=True,
                       write_seconds=time.monotonic() - self.started)
        # Every complete cache is read back before it may enter a solver.
        cache = DiskBlocks(receipt)
        seen = sum(len(block) for block in cache)
        if seen != self.events:
            raise ValueError('Complete response cache readback count differs')
        receipt['complete_readback_passed'] = True
        return receipt


class DiskBlocks:
    """Re-iterable sequence accepted by the unchanged _event_weight helper."""
    def __init__(self, receipt):
        self.receipt = receipt
        self.path = Path(receipt['path'])
        if receipt['policy'] != POLICY:
            raise ValueError('Unknown response storage policy')
        self.read_passes = 0
        self.read_seconds = 0.

    def verify_file(self):
        if digest(self.path) != self.receipt['sha256']:
            raise ValueError('Frozen response cache file SHA differs')

    def __iter__(self):
        start = time.monotonic()
        events = blocks = raw_total = encoded_total = 0
        with self.path.open('rb', buffering=0) as f:
            try:
                while True:
                    header = f.read(HEADER.size)
                    if not header:
                        break
                    if len(header) != HEADER.size:
                        raise ValueError('Truncated response cache header')
                    magic, rows, columns, raw_n, encoded_n, raw_sha, encoded_sha = HEADER.unpack(header)
                    if (magic != MAGIC or not 0 < rows <= MAX_ROWS or columns != self.receipt['columns'] or
                            raw_n != rows * columns * 4 or encoded_n > raw_n + (1 << 20)):
                        raise ValueError('Response cache block identity/size differs')
                    encoded = f.read(encoded_n)
                    if len(encoded) != encoded_n or hashlib.sha256(encoded).digest() != encoded_sha:
                        raise ValueError('Response cache block encoded SHA differs')
                    decoder = zlib.decompressobj()
                    raw = decoder.decompress(encoded, raw_n + 1)
                    if (len(raw) != raw_n or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or
                            hashlib.sha256(raw).digest() != raw_sha):
                        raise ValueError('Response cache decoded size/SHA differs')
                    array = np.frombuffer(raw, dtype='<f4').reshape(rows, columns).copy()
                    del raw, encoded, decoder
                    if not np.isfinite(array).all() or (array < 0).any():
                        raise ValueError('Response cache contains invalid rows')
                    events += rows
                    blocks += 1
                    raw_total += raw_n
                    encoded_total += HEADER.size + encoded_n
                    yield torch.from_numpy(array)
                    del array
                    if blocks % 64 == 0:
                        drop_file_cache(f.fileno())
            finally:
                drop_file_cache(f.fileno())
        if (events, blocks, raw_total, encoded_total) != tuple(self.receipt[k] for k in
                ('events', 'blocks', 'raw_bytes', 'encoded_bytes')):
            raise ValueError('Missing/extra complete response cache blocks/events')
        self.read_passes += 1
        self.read_seconds += time.monotonic() - start


def attach_cache(manifest_path, expected):
    manifest = read(manifest_path)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError('Existing response cache identity differs: ' + key)
    if not manifest.get('complete') or len(manifest['views']) != 20:
        raise ValueError('All twenty complete cached response views required')
    blocks = [DiskBlocks(receipt) for receipt in manifest['views']]
    for block in blocks:
        block.verify_file()
        if sum(len(x) for x in block) != block.receipt['events']:
            raise ValueError('Existing complete cache event readback differs')
    return manifest, blocks
