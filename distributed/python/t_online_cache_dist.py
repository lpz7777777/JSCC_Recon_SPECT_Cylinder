from collections import OrderedDict

import numpy as np
import torch

try:
    from distributed.python._path_setup import setup_repo_root
except ImportError:
    try:
        from ._path_setup import setup_repo_root
    except ImportError:
        from _path_setup import setup_repo_root

setup_repo_root()
from process_list_plane_strict import get_compton_backproj_list_single


def chunk_seed(base_seed, global_rank, energy_keV, subset_idx, rotate_idx, chunk_idx):
    return (
        int(base_seed)
        + int(global_rank) * 1000003
        + int(energy_keV) * 1009
        + int(subset_idx) * 9176
        + int(rotate_idx) * 131
        + int(chunk_idx)
    )


def build_list_chunk_plan(list_local, osem_subset_num, list_chunk_events, seed, global_rank, energy_keV):
    rotate_num = len(list_local)
    subset_plan = [[[] for _ in range(rotate_num)] for _ in range(osem_subset_num)]

    for rotate_idx in range(rotate_num):
        subset_chunks = list(torch.chunk(list_local[rotate_idx], osem_subset_num, dim=0))
        for subset_idx, subset_chunk in enumerate(subset_chunks):
            for chunk_idx, list_chunk in enumerate(torch.split(subset_chunk, list_chunk_events, dim=0)):
                if list_chunk.size(0) == 0:
                    continue

                subset_plan[subset_idx][rotate_idx].append(
                    {
                        "key": (int(energy_keV), int(subset_idx), int(rotate_idx), int(chunk_idx)),
                        "seed": chunk_seed(seed, global_rank, energy_keV, subset_idx, rotate_idx, chunk_idx),
                        "list_chunk": list_chunk.contiguous(),
                    }
                )

    return subset_plan


class ComptonChunkCache:
    def __init__(self, max_bytes, pin_memory=True):
        self.max_bytes = int(max_bytes)
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.cache = OrderedDict()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.evictions = 0

    def _tensor_bytes(self, tensor):
        return tensor.nelement() * tensor.element_size()

    def get(self, key):
        tensor = self.cache.get(key)
        if tensor is None:
            self.misses += 1
            return None

        self.cache.move_to_end(key)
        self.hits += 1
        return tensor

    def put(self, key, tensor):
        if self.max_bytes <= 0:
            return False

        if self.pin_memory and tensor.device.type == "cpu" and not tensor.is_pinned():
            tensor = tensor.pin_memory()

        size_bytes = self._tensor_bytes(tensor)
        if size_bytes > self.max_bytes:
            return False

        if key in self.cache:
            old_tensor = self.cache.pop(key)
            self.current_bytes -= self._tensor_bytes(old_tensor)

        while self.current_bytes + size_bytes > self.max_bytes and len(self.cache) > 0:
            _, old_tensor = self.cache.popitem(last=False)
            self.current_bytes -= self._tensor_bytes(old_tensor)
            self.evictions += 1

        self.cache[key] = tensor
        self.current_bytes += size_bytes
        self.puts += 1
        return True

    def stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "entries": len(self.cache),
            "bytes": self.current_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "puts": self.puts,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
        }

    def reset_stats(self):
        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.evictions = 0


class OnlineComptonProvider:
    def __init__(
        self,
        device,
        sysmat_full_gpu,
        detector_gpu,
        coor_polar_gpu,
        delta_r1,
        delta_r2,
        e0,
        ene_resolution,
        ene_threshold_max,
        ene_threshold_min,
        ene_threshold_sum,
        cache,
    ):
        self.device = device
        self.sysmat_full_gpu = sysmat_full_gpu
        self.detector_gpu = detector_gpu
        self.coor_polar_gpu = coor_polar_gpu
        self.delta_r1 = delta_r1
        self.delta_r2 = delta_r2
        self.e0 = e0
        self.ene_resolution = ene_resolution
        self.ene_threshold_max = ene_threshold_max
        self.ene_threshold_min = ene_threshold_min
        self.ene_threshold_sum = ene_threshold_sum
        self.cache = cache

    def _generate_t_cpu(self, record):
        seed = int(record["seed"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        t_chunk, _, _ = get_compton_backproj_list_single(
            self.sysmat_full_gpu,
            self.detector_gpu,
            self.coor_polar_gpu,
            record["list_chunk"].to(self.device, non_blocking=True),
            self.delta_r1,
            self.delta_r2,
            self.e0,
            self.ene_resolution,
            self.ene_threshold_max,
            self.ene_threshold_min,
            self.ene_threshold_sum,
            self.device,
        )
        return t_chunk.contiguous()

    def get_t_block(self, record):
        cached = self.cache.get(record["key"])
        if cached is not None:
            return cached.to(self.device, non_blocking=True)

        t_cpu = self._generate_t_cpu(record)
        self.cache.put(record["key"], t_cpu)
        return t_cpu.to(self.device, non_blocking=True)

    def count_rows_per_rotate(self, subset_plan):
        rotate_num = len(subset_plan[0]) if len(subset_plan) > 0 else 0
        rows_per_rotate = [0 for _ in range(rotate_num)]

        for subset_idx in range(len(subset_plan)):
            for rotate_idx in range(rotate_num):
                for record in subset_plan[subset_idx][rotate_idx]:
                    cached = self.cache.get(record["key"])
                    if cached is None:
                        cached = self._generate_t_cpu(record)
                        self.cache.put(record["key"], cached)
                    rows_per_rotate[rotate_idx] += int(cached.size(0))

        return rows_per_rotate


def load_generation_factors(factor_path, pixel_num, intensity, device):
    detector_gpu = torch.from_numpy(
        np.genfromtxt(f"{factor_path}/Detector.csv", delimiter=",", dtype=np.float32)[:, 1:4]
    ).to(device)
    coor_polar_gpu = torch.from_numpy(
        np.genfromtxt(f"{factor_path}/coor_polar_full.csv", delimiter=",", dtype=np.float32)
    ).to(device)

    sysmat_file_path = f"{factor_path}/SysMat_polar"
    float_size = np.dtype(np.float32).itemsize
    element_count = __import__("os").path.getsize(sysmat_file_path) // float_size
    total_bins = element_count // pixel_num
    sysmat_mmap = np.memmap(sysmat_file_path, dtype=np.float32, mode="r", shape=(pixel_num, total_bins))
    sysmat_full_cpu = np.array(sysmat_mmap.T, dtype=np.float32, copy=True)
    del sysmat_mmap
    sysmat_full_gpu = torch.from_numpy(sysmat_full_cpu).to(device) * intensity
    return sysmat_full_gpu, detector_gpu, coor_polar_gpu
