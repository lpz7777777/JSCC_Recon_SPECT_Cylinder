import torch
import torch.distributed as dist


def log_gpu_memory_usage(stage_label, device):
    if not torch.cuda.is_available():
        if dist.get_rank() == 0:
            print(f"[GPU Memory][{stage_label}] CUDA is not available.")
        return

    torch.cuda.synchronize(device)

    total_bytes = torch.cuda.get_device_properties(device).total_memory
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    free_bytes, total_bytes_mem_get = torch.cuda.mem_get_info(device)
    total_bytes = min(total_bytes, total_bytes_mem_get)
    used_driver_bytes = total_bytes - free_bytes

    stats = {
        "rank": int(dist.get_rank()),
        "local_rank": int(device.index if device.index is not None else torch.cuda.current_device()),
        "device_name": torch.cuda.get_device_name(device),
        "allocated_gb": allocated_bytes / (1024 ** 3),
        "reserved_gb": reserved_bytes / (1024 ** 3),
        "used_driver_gb": used_driver_bytes / (1024 ** 3),
        "total_gb": total_bytes / (1024 ** 3),
        "allocated_ratio": allocated_bytes / total_bytes if total_bytes > 0 else 0.0,
        "reserved_ratio": reserved_bytes / total_bytes if total_bytes > 0 else 0.0,
        "used_driver_ratio": used_driver_bytes / total_bytes if total_bytes > 0 else 0.0,
    }

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, stats)

    if dist.get_rank() == 0:
        print(f"[GPU Memory][{stage_label}]")
        for item in sorted(gathered, key=lambda x: x["rank"]):
            print(
                "  "
                f"rank={item['rank']:>2} "
                f"local_rank={item['local_rank']:>2} "
                f"gpu='{item['device_name']}' "
                f"allocated={item['allocated_gb']:.2f}/{item['total_gb']:.2f}GB ({item['allocated_ratio'] * 100:.1f}%) "
                f"reserved={item['reserved_gb']:.2f}/{item['total_gb']:.2f}GB ({item['reserved_ratio'] * 100:.1f}%) "
                f"driver_used={item['used_driver_gb']:.2f}/{item['total_gb']:.2f}GB ({item['used_driver_ratio'] * 100:.1f}%)"
            )
