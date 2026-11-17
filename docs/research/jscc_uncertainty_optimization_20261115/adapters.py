"""Isolated exact-storage adapters; no physics, solver, or precision changes.

Block views are borrowed until the iterator advances: consumers must enqueue
all use on the captured compute stream before next(). Do not retain the views
or use additional streams. Only the two owned staging buffers are pinned.
"""
from dataclasses import dataclass
import time
import torch


@dataclass(frozen=True)
class MemoryBudget:
    total_bytes: int
    used_bytes: int
    reserved_bytes: int = 0
    historical_peak_bytes: int = 0
    transient_reserve_bytes: int = 0
    limit_fraction: float = .8

    def available(self):
        if not 0 < self.limit_fraction <= .8:
            raise ValueError('At least 20% headroom is required')
        if min(self.total_bytes, self.used_bytes, self.reserved_bytes,
               self.historical_peak_bytes, self.transient_reserve_bytes) < 0:
            raise ValueError('Negative memory measurement')
        return max(0, int(self.total_bytes*self.limit_fraction) -
                   max(self.used_bytes, self.reserved_bytes, self.historical_peak_bytes) -
                   self.transient_reserve_bytes)


def current_cuda_budget(device, transient_reserve_bytes, historical_peak_bytes=0):
    """Measured global device usage plus process reservation, conservative sum."""
    free, total = torch.cuda.mem_get_info(device)
    return MemoryBudget(total, total-free, torch.cuda.memory_reserved(device),
                        max(historical_peak_bytes, torch.cuda.max_memory_reserved(device)),
                        transient_reserve_bytes)


class BoundedViewResponse:
    """Delegate to frozen ViewResponse; cache a fixed prefix, never an LRU.

    All-view fit uses its existing cache='device' directly. Partial capacity
    invokes the identical frozen matrix() once per cached view. No arithmetic
    or view order changes. One object owns one energy/phase and must be closed
    before Compton preparation or a second energy cache is admitted.
    """
    def __init__(self, frozen_module, full_rows, geometry, *, budget,
                 phase, requested_views=None):
        if phase not in ('sc_solve', 'joint_solve_after_compton_prepare'):
            raise ValueError('No SC view cache is allowed in Compton preparation')
        self._module = frozen_module
        self.full_rows, self.geometry = full_rows, geometry
        self.compute_stream = torch.cuda.current_stream(full_rows.device) if full_rows.is_cuda else None
        self.closed = False
        self.bytes_per_view = full_rows.shape[0]*geometry.active_count*full_rows.element_size()
        # index_select AND multiply coexist during construction; keep one
        # additional compact matrix outside cached storage even in all-view mode.
        max_views = max(0, (budget.available()-self.bytes_per_view)//max(1, self.bytes_per_view))
        n = geometry.views if requested_views is None else requested_views
        if not isinstance(n, int) or not 0 <= n <= geometry.views:
            raise ValueError('Invalid requested view count')
        self.cached_views = min(n, int(max_views))
        self.cache_bytes = self.cached_views*self.bytes_per_view
        start = time.perf_counter()
        self.base = frozen_module.ViewResponse(full_rows, geometry,
                         'device' if self.cached_views == geometry.views else 'none')
        self.partial = ({v: self.base.matrix(v) for v in range(self.cached_views)}
                        if self.cached_views != geometry.views else {})
        self.prepare_submit_seconds = time.perf_counter()-start

    def matrix(self, view):
        if self.closed: raise RuntimeError('View cache is closed')
        if self.compute_stream is not None and torch.cuda.current_stream(self.full_rows.device) != self.compute_stream:
            raise RuntimeError('View cache must stay on its owning compute stream')
        if not 0 <= view < self.geometry.views: raise IndexError(view)
        return self.partial[view] if view in self.partial else self.base.matrix(view)

    def sensitivity(self):
        # Run the original implementation with this matrix provider.
        return self._module.ViewResponse.sensitivity(self)

    def close(self):
        if not self.closed:
            if self.compute_stream is not None:
                self.compute_stream.synchronize()
            self.partial.clear(); self.base.cached = None
            self.closed = True

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


class PhaseCacheOwner:
    """Explicit lifetime owner. A second cache cannot silently coexist."""
    def __init__(self): self.active = None
    def open(self, *args, **kwargs):
        if self.active is not None and not self.active.closed:
            raise RuntimeError('Close previous energy/phase cache first')
        self.active = BoundedViewResponse(*args, **kwargs)
        return self.active
    def before_compton_prepare(self):
        if self.active is not None: self.active.close()
        self.active = None


def block_groups(blocks, group_blocks=8, arithmetic_rows=32):
    if group_blocks < 1 or arithmetic_rows != 32:
        raise ValueError('Positive copy grouping and original 32-row arithmetic required')
    if not blocks: return []
    if not isinstance(blocks[0], torch.Tensor) or blocks[0].ndim != 2:
        raise ValueError('Two-dimensional tensor blocks required')
    width = blocks[0].shape[1]
    for b in blocks:
        if (b.device.type != 'cpu' or b.dtype != torch.float32 or b.ndim != 2 or
            b.shape[1] != width or b.shape[0] > arithmetic_rows or not b.is_contiguous() or b.requires_grad):
            raise ValueError('Unchanged contiguous CPU FP32 original blocks required')
    return [tuple(blocks[i:i+group_blocks]) for i in range(0, len(blocks), group_blocks)]


class OrderedTransferPool:
    """Two-buffer transfer aggregation with original GEMV boundaries.

    CUDA: host refill waits for old H2D; copy overwrite waits for old compute;
    compute waits for this H2D. Buffers stay owned until explicit close. Event
    timings are completion timings and collected only after synchronization.
    CPU mode tests packing/order/lifetimes, NOT asynchronous CUDA correctness.
    """
    def __init__(self, columns, device, *, group_blocks=8, max_host_staging_bytes,
                 device_budget=None, asynchronous=True, timing=False):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.columns, self.group_blocks = columns, group_blocks
        if columns < 1 or group_blocks < 1: raise ValueError('Positive dimensions required')
        self.capacity = 32*group_blocks
        self.storage_bytes = 2*self.capacity*columns*4
        if self.storage_bytes > max_host_staging_bytes:
            raise MemoryError('Bounded host staging budget exceeded')
        if self.device.type not in ('cpu', 'cuda'): raise ValueError('CPU or CUDA only')
        if self.device.type == 'cuda':
            if device_budget is None or self.storage_bytes > device_budget.available():
                raise MemoryError('Measured GPU budget including headroom required')
        self.asynchronous, self.timing = asynchronous, timing
        self.busy = self.closed = False
        self.host = [torch.empty((self.capacity, columns), dtype=torch.float32,
                          pin_memory=self.device.type == 'cuda') for _ in range(2)]
        self.gpu = ([torch.empty_like(x, device=self.device) for x in self.host]
                    if self.device.type == 'cuda' else self.host)
        self.copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self.ready = [None, None]; self.done = [None, None]
        self.timings = []; self.reset_stats()

    def reset_stats(self):
        if self.busy: raise RuntimeError('Cannot reset live iterator')
        self.stats = dict(original_blocks=0, original_nonempty_blocks=0,
                          transfer_calls=0, transfer_bytes=0, pack_calls=0,
                          cpu_staging_seconds=0., host_wait_seconds=0.,
                          cpu_copy_seconds=0., h2d_seconds=0.)
        self.timings = []

    def _enqueue(self, group, slot, compute):
        n = sum(b.shape[0] for b in group)
        if self.ready[slot] is not None:
            t = time.perf_counter(); self.ready[slot].synchronize()
            self.stats['host_wait_seconds'] += time.perf_counter()-t
        t = time.perf_counter(); off = 0
        for b in group:
            self.host[slot][off:off+len(b)].copy_(b); off += len(b)
        self.stats['cpu_staging_seconds'] += time.perf_counter()-t
        self.stats['original_blocks'] += len(group)
        self.stats['original_nonempty_blocks'] += sum(len(b)>0 for b in group)
        self.stats['pack_calls'] += int(n>0)
        if self.device.type == 'cuda':
            stream = self.copy_stream if self.asynchronous else compute
            with torch.cuda.stream(stream):
                if self.done[slot] is not None: stream.wait_event(self.done[slot])
                begin = torch.cuda.Event(enable_timing=True) if self.timing else None
                if begin is not None: begin.record(stream)
                if n: self.gpu[slot][:n].copy_(self.host[slot][:n], non_blocking=True)
                ready = torch.cuda.Event(enable_timing=self.timing); ready.record(stream)
            self.ready[slot] = ready
            if begin is not None: self.timings.append((begin, ready))
            self.stats['transfer_calls'] += int(n>0)
            self.stats['transfer_bytes'] += n*self.columns*4
        return n

    def iterate(self, blocks):
        if self.closed or self.busy: raise RuntimeError('Closed or concurrent transfer pool')
        groups = block_groups(blocks, self.group_blocks)
        if any(b.shape[1] != self.columns for b in blocks): raise ValueError('Column mismatch')
        self.busy = True
        compute = torch.cuda.current_stream(self.device) if self.device.type == 'cuda' else None
        try:
            if not groups: return
            self._enqueue(groups[0], 0, compute)
            for g, group in enumerate(groups):
                slot = g % 2
                # Stage next group before yielding current; after the first
                # group, the previous compute is still queued while packing.
                if g+1 < len(groups): self._enqueue(groups[g+1], (g+1)%2, compute)
                if compute is not None: compute.wait_event(self.ready[slot])
                off = 0
                for b in group:
                    if compute is not None and torch.cuda.current_stream(self.device) != compute:
                        raise RuntimeError('Consumer changed compute stream')
                    item = self.gpu[slot][off:off+len(b)]; off += len(b)
                    if compute is not None: item.record_stream(compute)
                    yield item
                if compute is not None:
                    self.done[slot] = torch.cuda.Event(); self.done[slot].record(compute)
        finally:
            # Handles early generator.close/consumer exceptions too: protect
            # ANY outstanding uses before the pool can be reused.
            if compute is not None:
                for slot in range(2):
                    self.done[slot] = torch.cuda.Event(); self.done[slot].record(compute)
            self.busy = False

    def finish_stats(self):
        if self.busy: raise RuntimeError('Finish iterator first')
        if self.device.type == 'cuda':
            # Consumers may have used a non-default stream and then switched
            # back. Synchronize recorded ownership events, not just whichever
            # stream happens to be current at close()/finish_stats().
            for event in self.ready+self.done:
                if event is not None: event.synchronize()
            self.copy_stream.synchronize()
            self.stats['h2d_seconds'] = sum(a.elapsed_time(b)/1000 for a,b in self.timings)
        return dict(self.stats, staging_allocated_bytes=self.storage_bytes,
                    pinned_bytes=self.storage_bytes if self.device.type=='cuda' else 0,
                    device_buffers_bytes=self.storage_bytes if self.device.type=='cuda' else 0,
                    h2d_measured=self.device.type=='cuda' and self.timing)

    def close(self):
        if self.busy: raise RuntimeError('Close active generator before pool')
        if not self.closed:
            self.finish_stats(); self.host.clear(); self.gpu.clear(); self.closed = True
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()


class ViewBlocks:
    """Reusable iterable accepted directly by the frozen _event_weight."""
    def __init__(self, pool, blocks): self.pool, self.blocks = pool, tuple(blocks)
    def __iter__(self): return self.pool.iterate(self.blocks)
    def __len__(self): return len(self.blocks)
