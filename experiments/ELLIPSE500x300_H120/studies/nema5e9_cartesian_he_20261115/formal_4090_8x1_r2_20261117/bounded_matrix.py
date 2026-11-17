"""Read-only B mapping with bounded resident pages; never change its values.

Fancy indexing already creates an independent NumPy array. Only after that
copy completes do we drop this process's mapping pages. Shared file data and
the independent result are untouched. A shared slice is left mapped until
the original SC provider copies it and releases its mapping, as before.
"""
import hashlib
import json
import mmap
from pathlib import Path
import tempfile
import numpy as np


class BoundedReadOnlyMatrix:
    def __init__(self, source):
        if not isinstance(source, np.memmap) or source.mode != 'r':
            raise ValueError('A read-only NumPy memmap is required')
        if not hasattr(source._mmap, 'madvise') or not hasattr(mmap, 'MADV_DONTNEED'):
            raise RuntimeError('Bounded production mapping requires Linux MADV_DONTNEED')
        self.source = source
        self.shape, self.dtype = source.shape, source.dtype
        self.copy_reads = 0
        self.max_copied_bytes = 0

    def __getitem__(self, key):
        result = self.source[key]
        if not np.shares_memory(result, self.source):
            # Advanced indexing has finished copying, so no alias is invalidated.
            self.source._mmap.madvise(mmap.MADV_DONTNEED)
            self.copy_reads += 1
            self.max_copied_bytes = max(self.max_copied_bytes, result.nbytes)
        return result


def auxiliary_budget_bytes(geometry, world):
    import resource
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if baseline > 3<<30:
        raise MemoryError('Measured pre-C baseline exceeds bounded-map admission')
    # Peak SC source slice: one read-only mapped slice plus its CPU copy.
    # Include full history, the bounded staging pool, and 2 GiB for CPU runtime,
    # calibration/event arrays, transient copied tiles and allocation overhead.
    local_rows = (10496+world-1)//world
    sc_slice_peak = 2*local_rows*507000*4
    history = 200*geometry.active_count*4
    return baseline+sc_slice_peak+history+(512<<20)+(2<<30)


def mapped_rss(path):
    total = 0
    selected = False
    for line in Path('/proc/self/smaps').read_text().splitlines():
        if '-' in line.split()[0] and line[0] in '0123456789abcdef':
            selected = line.endswith(str(path))
        elif selected and line.startswith('Rss:'):
            total += int(line.split()[1])*1024
    return total


def self_test():
    """Tiny synthetic matrix: exact bytes/operators/history and real Linux RSS."""
    rng = np.random.default_rng(20261117)
    rows = rng.uniform(.1, 1, (128, 4096)).astype('<f4')
    checks = []
    with tempfile.TemporaryDirectory(prefix='cart4090_bounded_map_') as work:
        path = Path(work)/'synthetic.float32'
        path.write_bytes(rows.tobytes())
        before_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        source = np.memmap(path, '<f4', mode='r', shape=rows.shape)
        bounded = BoundedReadOnlyMatrix(source)
        assert float(source.sum()) > 0  # Fault in all pages for the RSS test.
        rss_before = mapped_rss(path)
        indices = np.array([127, 1, 1, 0, 54])
        for start, end in [(0, 17), (17, 1024), (1024, 4096), (4096, 4096)]:
            selected = bounded[indices, start:end]
            assert np.array_equal(selected, rows[indices, start:end])
            assert not np.shares_memory(selected, source)
            checks.append(dict(start=start, end=end, exact=True))
        rss_after = mapped_rss(path)
        assert rss_after <= 64<<10, (rss_before, rss_after)
        shared = bounded[2:5]
        assert np.shares_memory(shared, source) and np.array_equal(shared, rows[2:5])
        del shared
        # Same complete synthetic matrix, forward/transpose/S and short history.
        chosen = np.arange(128)
        actual = bounded[chosen, :]
        x = rng.uniform(.2, 2, (4096, 1)).astype('<f4')
        y = rng.uniform(.2, 2, (128, 1)).astype('<f4')
        assert np.array_equal(actual@x, rows@x)
        assert np.array_equal(actual.T@y, rows.T@y)
        sensitivity = rows.sum(0)[:, None]
        assert np.array_equal(actual.sum(0)[:, None], sensitivity)
        counts = np.ones((128, 1), dtype='<f4')*20
        left = np.ones_like(x);right = left.copy()
        for _ in range(4):
            left *= (actual.T@(counts/(actual@left).clip(1e-12)))/sensitivity
            right *= (rows.T@(counts/(rows@right).clip(1e-12)))/sensitivity
            assert np.array_equal(left, right)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before_sha
        result = dict(passed=True, synthetic_matrix=True, physical_kernel_not_replaced=True,
            indexing_checks=checks, forward_transpose_S_short_history_exact=True,
            source_SHA_unchanged=True, real_Linux_madvise=True,
            mapped_RSS_before_bytes=rss_before,mapped_RSS_after_bytes=rss_after,
            copy_reads=bounded.copy_reads,max_selected_copy_bytes=bounded.max_copied_bytes,
            actual_GPU_resource_gate_still_required=True)
        del bounded
        source._mmap.close()
    print(json.dumps(result))


if __name__ == '__main__':
    import sys
    if sys.argv[1:] != ['--self-test']:
        raise ValueError('Only --self-test is supported')
    self_test()
